"""Sharing the dashboard: what viewers see, the relay to the viewer sandbox, and the header links."""

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from async_grpo_tool_use import run
from async_grpo_tool_use.dashboard import share
from async_grpo_tool_use.dashboard.server import Hub
from async_grpo_tool_use.dashboard.viewer import make_viewer_handler

# Lines shaped like the launcher, Gym and trainer output of real runs.
SENSITIVE = [
    "[grpo] inference: gateway 09757ea9-7852-4486-9f17-e36018182d1c ready at https://async-grpo-1.ab12cd.gw.cwinference.com",
    "[grpo] trainer: starting GRPO (grpo_async_qwen.yaml) on 8 GPUs, rollouts from https://async-grpo-1.ab12cd.gw.cwinference.com/v1",
    "[grpo] sunk: job 80 submitted (/mnt/data/someone/async-grpo/runs/async-grpo-10831481)",
    "[grpo] storage: temporary access key for s3://gradient-hotloading/cookbook/async-grpo-tool-use/runs/async-grpo-1",
    "[gym] [grpo] tunnel: forwarding to 10.0.12.34:7443",
    "[grpo] could not reach login.cluster.internal.example.com:22",
    "COREWEAVE_API_TOKEN=CW-SECRET-abc AWS_SECRET_ACCESS_KEY=xyz Authorization: Bearer eyJhbGciOi",
    "key AKIAABCDEFGHIJKLMNOP and wandb_v1_1NDDPCz77PTgFyqcIA7pjlge3Iw_fzJCzrn3MZGSHAw37ij",
]
HARMLESS = [
    "[grpo] hot-load v3: 339 tensors, 15.2 GiB; export 12s, upload and load 30s",
    "▶ Skipping reference_logprobs (skip_reference_policy_logprobs_calculation=True)...",
    "[grpo] calibration: details in /work/logs/calibration.json",
    "[grpo] run tag async-grpo-10831481",
    "    • Accuracy: 0.3667",
]


def test_scrub_removes_secrets_addresses_and_home_directories():
    clean = share.scrub(SENSITIVE, known=["CW-SECRET-abc"])
    text = "\n".join(clean)
    for leak in ("cwinference", "/mnt/data", "someone", "gradient-hotloading", "10.0.12.34", "example.com",
                 "CW-SECRET", "xyz", "eyJ", "AKIA", "wandb_v1", "09757ea9"):  # fmt: skip
        assert leak not in text, leak
    assert clean[2] == "[grpo] sunk: job 80 submitted (<path>"
    assert "COREWEAVE_API_TOKEN=<redacted>" in text and "Bearer <redacted>" in text


def test_scrub_keeps_the_lines_the_page_reads():
    assert share.scrub(HARMLESS, known=[]) == HARMLESS
    event = {"type": "step_done", "step": 3, "metrics": {"loss": 0.1}, "timings": {"generation": 60.1}}
    assert share.scrub_event(event, []) == event


def test_scrub_event_drops_unknown_types_and_filters_links():
    assert share.scrub_event({"type": "secret_dump", "line": "x"}, []) is None
    links = [
        {"label": "W&B run", "url": "https://forge.coreweave.com/wandb/team/p/runs/r"},
        {"label": "Agent Lens traces", "url": "https://forge.coreweave.com/wandb/team/p/weave/agents"},
        {"label": "elsewhere", "url": "https://internal.example.com/x"},
    ]
    viewer = "https://8080-abc.ep1.cwsandbox.com"
    clean = share.scrub_event(
        {"type": "meta", "trainer": "SUNK · h100", "links": links, "share_url": viewer}, [], viewer
    )
    assert [link["label"] for link in clean["links"]] == ["W&B run", "Agent Lens traces"]
    assert clean["links"][1]["url"] == links[1]["url"] and clean["share_url"] == viewer
    assert "share_url" not in share.scrub_event({"type": "meta", "share_url": "https://other.example.com"}, [], viewer)


def serve_viewer(token: str) -> tuple[str, Hub, dict]:
    hub, meta = Hub(), {"mode": "live", "shared": True}
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_viewer_handler(hub, meta, token))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_port}", hub, meta


def post(url: str, body: bytes, token: str | None) -> int:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=body, method="POST", headers=headers)) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code


def read_events(url: str, last_id: str | None = None, count: int = 1) -> list[str]:
    """The first ``count`` messages of the event stream, each as its raw text."""
    headers = {"Last-Event-ID": last_id} if last_id else {}
    messages, current = [], []
    with urllib.request.urlopen(urllib.request.Request(url + "/events", headers=headers), timeout=10) as stream:
        for raw in stream:
            line = raw.decode().rstrip("\n")
            if line:
                current.append(line)
                continue
            messages.append("\n".join(current))
            current = []
            if len(messages) == count:
                return messages
    return messages


def test_viewer_is_read_only_without_the_relay_token():
    url, hub, _ = serve_viewer("t0ken")
    assert post(url + "/publish", b'[{"type": "log", "line": "x"}]', None) == 401
    assert post(url + "/publish", b'[{"type": "log", "line": "x"}]', "wrong") == 401
    assert post(url + "/anything", b"{}", "t0ken") == 404
    assert hub.events == []


def test_late_joiners_get_the_backlog_and_reconnects_resume():
    url, hub, meta = serve_viewer("t0ken")
    events = [{"type": "meta", "run_tag": "async-grpo-1"}, {"type": "step", "step": 1, "of": 2}]
    assert post(url + "/publish", json.dumps(events).encode(), "t0ken") == 204
    assert meta["run_tag"] == "async-grpo-1"
    (backlog,) = read_events(url)
    assert backlog.startswith("id: 2\nevent: backlog\n")
    assert [e["type"] for e in json.loads(backlog.split("data: ", 1)[1])] == ["meta", "step"]
    hub.publish({"type": "step", "step": 2, "of": 2})
    (resumed,) = read_events(url, last_id="2")  # no backlog again: the page would reload
    assert resumed.startswith("id: 3\ndata: ") and '"step": 2' in resumed


def test_relay_sends_backlog_then_live_events_scrubbed_and_retries():
    hub = Hub()
    hub.publish({"type": "log", "line": "[grpo] token TOKEN=abcdefgh1234"})
    sent, failures = [], [OSError("down")]

    def fake_post(body: bytes) -> None:
        if failures:
            raise failures.pop()
        sent.extend(json.loads(body))

    relay = share.Relay(hub, "https://viewer", "t", known=lambda: ["supersecretvalue"], post=fake_post)
    hub.publish({"type": "log", "line": "key supersecretvalue"})
    hub.publish({"type": "internal", "line": "dropped"})
    assert relay.flush(timeout=10)
    assert [event["line"] for event in sent] == ["[grpo] token TOKEN=<redacted>", "key <redacted>"]


def test_run_links_point_at_the_run_in_forge():
    env = {"WANDB_API_KEY": "k", "WANDB_ENTITY": "my-team", "WANDB_PROJECT": "async-grpo-tool-use"}
    wandb, page_metrics, traces, evals = run.run_links(env, "async-grpo-1")
    project = "https://forge.coreweave.com/wandb/my-team/async-grpo-tool-use"
    assert wandb == {"label": "W&B run", "url": f"{project}/runs/async-grpo-1"}
    assert page_metrics["url"] == f"{project}/runs/async-grpo-1-dashboard"  # what the page draws, in W&B
    assert traces["url"] == f"{project}/weave/agents" and evals["url"] == f"{project}/weave/evaluations"
    links = (wandb, page_metrics, traces, evals)
    assert all(link["url"].startswith(share.LINK_HOSTS) for link in links)  # viewers see them


def test_no_links_without_a_wandb_key():
    assert run.run_links({"WANDB_ENTITY": "e", "WANDB_PROJECT": "p"}, "async-grpo-1") == []


def test_the_viewer_sandbox_gets_every_file_it_imports():
    for name in share.VIEWER_FILES:
        assert (share.PACKAGE / name).is_file(), name


def test_dotenv_lines(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# comment\n\nCWSANDBOX_API_KEY=abc\nexport SUNK_DATA_DIR='/mnt/data/me'\n# WANDB_API_KEY=\n")
    assert run.read_dotenv(env) == {"CWSANDBOX_API_KEY": "abc", "SUNK_DATA_DIR": "/mnt/data/me"}
    assert run.read_dotenv(tmp_path / "missing") == {}


def test_the_shared_url_comes_back_before_the_page_answers():
    """The URL is printed first, before the bring-up, so it must not wait for the page."""

    class FakeSandbox:
        sandbox_id = "sbx-1"
        service_urls = [("dashboard", 8080, "https://8080-sbx-1.example.com/")]

        def get_status(self):
            raise AssertionError("the URL is already known")

    class FakeSdk:
        Endpoint = Service = ResourceOptions = staticmethod(lambda **kwargs: kwargs)

        class Sandbox:
            @staticmethod
            def run(*command, **kwargs):
                FakeSdk.created = kwargs
                return FakeSandbox()

    sandbox, url, token = share.create_viewer(FakeSdk, "async-grpo-1", hours=2)
    assert url == "https://8080-sbx-1.example.com" and len(token) > 20
    assert FakeSdk.created["max_lifetime_seconds"] == 7200
    assert "async-grpo-1-dashboard" in FakeSdk.created["tags"]
    assert FakeSdk.created["environment_variables"]["SHARE_TOKEN"] == token
