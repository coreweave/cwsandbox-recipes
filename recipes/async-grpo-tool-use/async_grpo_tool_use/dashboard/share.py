"""Share the dashboard: a small serverless CPU sandbox serves it at a public HTTPS URL.

The viewer sandbox runs ``viewer.py``: the same page and event stream as the local server,
read-only. The run relays every event to it (``Relay``), scrubbed of secrets and internal
addresses first (``scrub_event``). Late joiners get the whole run. The page keeps serving after
the run ends, until you press Ctrl+C in the run's terminal or the sandbox's lifetime runs out.
"""

from __future__ import annotations

import json
import queue
import re
import secrets
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
SHARE_TAG = "async-grpo-dashboard"
VIEWER_IMAGE = "python:3.13-slim"
VIEWER_PORT = 8080
VIEWER_FILES = (
    "__init__.py",
    "dashboard/__init__.py",
    "dashboard/server.py",
    "dashboard/viewer.py",
    "dashboard/index.html",
)
# The platform closes an HTTPS request after this long; the page's event stream then reconnects
# and resumes from the last event it saw.
REQUEST_TIMEOUT_S = 900
EVENT_TYPES = {"meta", "stage", "step", "phase", "hotload", "step_done", "accuracy", "metric", "log"}
# Only links that need a login of their own are shown to viewers.
LINK_HOSTS = ("https://forge.coreweave.com/",)

REDACTIONS = [
    (re.compile(r"\bBearer\s+\S+", re.I), "Bearer <redacted>"),
    (re.compile(r"\b(\w*(?:TOKEN|KEY|SECRET|PASSWORD)\w*)=\S+", re.I), r"\1=<redacted>"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "<redacted>"),
    # Opaque credentials: long runs of letters and digits (also UUIDs of the run's resources).
    (re.compile(r"\b(?=[\w-]*\d)(?=[\w-]*[A-Za-z])[\w-]{32,}\b"), "<redacted>"),
    (re.compile(r"\b[a-z][a-z0-9+.-]*://\S+", re.I), "<url>"),
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b"), "<ip>"),
    (re.compile(r"\b(?:[a-z0-9-]+\.)+(?:com|net|io|org|ai|cloud|internal|local|svc)\b(?::\d+)?", re.I), "<host>"),
    (re.compile(r"(?<![\w.])/(?:mnt|home|Users|root)/\S*"), "<path>"),
]


def scrub(value: Any, known: list[str]) -> Any:
    """Redact secrets (``known`` values and anything that looks like one), URLs, IP addresses,
    host names and home directories in every string of ``value``."""
    if isinstance(value, str):
        for secret in known:
            value = value.replace(secret, "<redacted>")
        for pattern, replacement in REDACTIONS:
            value = pattern.sub(replacement, value)
        return value
    if isinstance(value, list):
        return [scrub(item, known) for item in value]
    if isinstance(value, dict):
        return {key: scrub(item, known) for key, item in value.items()}
    return value


def scrub_event(event: dict, known: list[str], share_url: str = "") -> dict | None:
    """The event as viewers may see it, or None to drop it. Header links pass when they point at
    ``LINK_HOSTS`` or at the shared page itself."""
    if event.get("type") not in EVENT_TYPES:
        return None
    if event["type"] != "meta":
        return scrub(event, known)
    clean = scrub({key: value for key, value in event.items() if key not in ("links", "share_url")}, known)
    if "links" in event:
        clean["links"] = [link for link in event["links"] if str(link.get("url", "")).startswith(LINK_HOSTS)]
    if share_url and event.get("share_url") == share_url:
        clean["share_url"] = share_url
    return clean


class Relay:
    """Sends every event of a local ``Hub`` to the viewer, the backlog first. A failed POST is
    retried with the same batch, so the viewer never misses or reorders events."""

    def __init__(self, hub, url: str, token: str, known: Callable[[], list[str]], post=None):
        self.share_url = url.rstrip("/")
        self.url = self.share_url + "/publish"
        self.token = token
        self.known = known
        self.post = post or self._post
        self.backlog, self.listener = hub.subscribe()
        self.pending = 0
        self.idle = threading.Event()
        threading.Thread(target=self._pump, daemon=True).start()

    def _post(self, body: bytes) -> None:
        request = urllib.request.Request(self.url, data=body, method="POST", headers={
            "Authorization": f"Bearer {self.token}", "Content-Type": "application/json"})  # fmt: skip
        with urllib.request.urlopen(request, timeout=30) as response:
            response.read()

    def _send(self, events: list[dict]) -> None:
        known = [value for value in self.known() if isinstance(value, str) and len(value) >= 8]
        batch = [clean for clean in (scrub_event(event, known, self.share_url) for event in events) if clean]
        body = json.dumps(batch).encode()
        while True:
            try:
                self.post(body)
                return
            except (urllib.error.URLError, OSError):
                time.sleep(2)

    def _pump(self) -> None:
        for start in range(0, len(self.backlog), 500):
            self._send(self.backlog[start : start + 500])
        while True:
            self.idle.set()
            batch = [self.listener.get()]
            self.idle.clear()
            deadline = time.monotonic() + 0.5
            while len(batch) < 500 and time.monotonic() < deadline:
                try:
                    batch.append(self.listener.get(timeout=max(deadline - time.monotonic(), 0)))
                except queue.Empty:
                    break
            self._send(batch)

    def flush(self, timeout: float = 15) -> bool:
        """Wait until every published event reached the viewer."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.idle.is_set() and self.listener.empty():
                return True
            time.sleep(0.2)
        return False


def viewer_tag(run_tag: str) -> str:
    return f"{run_tag}-dashboard"


def create_viewer(sdk, run_tag: str, hours: float) -> tuple[Any, str, str]:
    """Start the viewer sandbox. Return the sandbox, its public URL and the relay's token. The
    URL is assigned at create time, within seconds; the page answers a few seconds later."""
    token = secrets.token_urlsafe(32)
    files = [{"mount_path": f"/workspace/async_grpo_tool_use/{name}", "file_content": (PACKAGE / name).read_text("utf-8")}
             for name in VIEWER_FILES]  # fmt: skip
    limits = {"cpu": "1", "memory": "1Gi"}
    endpoint = sdk.Endpoint(kind="https", auth="open", request_timeout_seconds=REQUEST_TIMEOUT_S)
    # No run tag: the run's cleanup stops its own sandboxes, and the page outlives the run.
    sandbox = sdk.Sandbox.run(
        "python", "-m", "async_grpo_tool_use.dashboard.viewer",
        container_image=VIEWER_IMAGE,
        mounted_files=files,
        environment_variables={"PYTHONPATH": "/workspace", "PYTHONUNBUFFERED": "1",
                               "SHARE_TOKEN": token, "PORT": str(VIEWER_PORT)},
        resources=sdk.ResourceOptions(requests=limits, limits=limits),
        services=[sdk.Service(port=VIEWER_PORT, name="dashboard", visibility="public", endpoint=endpoint)],
        max_lifetime_seconds=int(hours * 3600),
        tags=[SHARE_TAG, viewer_tag(run_tag)],
    )  # fmt: skip
    deadline = time.monotonic() + 60
    while not sandbox.service_urls and time.monotonic() < deadline:
        time.sleep(1)
        sandbox.get_status()
    url = next((found for _, _, found in sandbox.service_urls if found), "")
    if not url:
        raise RuntimeError(f"the viewer sandbox {sandbox.sandbox_id} got no public URL")
    return sandbox, url.rstrip("/"), token


def wait_until_serving(url: str, timeout: float = 600) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url + "/meta", timeout=10) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(2)
    raise RuntimeError(f"{url} did not answer within {timeout:.0f} s")
