"""The shared dashboard, run inside the viewer sandbox (see ``share.py``).

Serves the page and its event stream like the local server, read-only. ``POST /publish`` with the
relay's bearer token appends events; nothing else can change the page.

Usage: SHARE_TOKEN=... PORT=8080 python -m async_grpo_tool_use.dashboard.viewer
"""

from __future__ import annotations

import hmac
import json
import os
from http.server import ThreadingHTTPServer

from async_grpo_tool_use.dashboard.server import Hub, make_handler

MAX_BODY = 8 << 20


def make_viewer_handler(hub: Hub, meta: dict, token: str):
    base = make_handler(hub, meta)

    class Handler(base):
        def do_POST(self) -> None:
            if self.path != "/publish":
                self.send_error(404)
                return
            given = self.headers.get("Authorization", "")
            if not hmac.compare_digest(given.encode(), f"Bearer {token}".encode()):
                self.send_error(401)
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                self.send_error(413)
                return
            try:
                events = json.loads(self.rfile.read(length) or b"[]")
            except json.JSONDecodeError:
                self.send_error(400)
                return
            for event in events:
                if event.get("type") == "meta":
                    meta.update({key: value for key, value in event.items() if key not in ("type", "t")})
                hub.publish(event)
            self.send_response(204)
            self.end_headers()

    return Handler


def main() -> int:
    token = os.environ["SHARE_TOKEN"]
    hub, meta = Hub(), {"mode": "live", "shared": True}
    server = ThreadingHTTPServer(
        ("0.0.0.0", int(os.environ.get("PORT", "8080"))), make_viewer_handler(hub, meta, token)
    )
    server.daemon_threads = True
    print(f"[viewer] serving on port {server.server_port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
