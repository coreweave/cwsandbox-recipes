"""Operator-facing lifecycle events.

One `[taubench] ...` line per event, on stdout. stdout is the only channel
that survives every runtime this recipe uses: Ray forwards actor stdout to
the driver, the sandbox streams it to the launcher, and SkyPilot tails it.
Plain logging.INFO is dropped unless something configures the root logger,
which the rollout workers never do.

Follow a run with:  grep "\\[taubench\\]"
"""

from __future__ import annotations

import sys
import threading

_lock = threading.Lock()


def log_event(message: str) -> None:
    with _lock:
        print(f"[taubench] {message}", file=sys.stdout, flush=True)


def short_id(sandbox_id: object, length: int = 8) -> str:
    return str(sandbox_id)[:length]
