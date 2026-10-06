"""Stop an owned sandbox on normal exit, errors, SIGINT, and SIGTERM."""

from contextlib import contextmanager
import json
from pathlib import Path
import signal
import sys

from cwsandbox import Sandbox


@contextmanager
def managed_sandbox(output: Path, *command: str, **options):
    sandbox = None
    creating = True
    interrupted = None

    def interrupt(signum, _frame):
        nonlocal interrupted
        interrupted = signum
        # Finish obtaining the ID before unwinding an in-flight create request.
        if not creating:
            raise SystemExit(128 + signum)

    previous = {
        sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        sandbox = Sandbox.run(*command, **options)
        (output / "sandbox-id.txt").write_text(sandbox.sandbox_id + "\n")
        creating = False
        if interrupted is not None:
            raise SystemExit(128 + interrupted)
        yield sandbox
    finally:
        # A second Ctrl-C must not cancel the bounded cleanup request.
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        try:
            if sandbox is not None:
                failed = sys.exc_info()[0] is not None
                cleanup = {"sandbox_id": sandbox.sandbox_id, "stopped": False}
                print(f"Stopping GPU sandbox {sandbox.sandbox_id}...", flush=True)
                try:
                    sandbox.stop(missing_ok=True).result(timeout=60)
                    cleanup["stopped"] = True
                    print("GPU sandbox stopped.", flush=True)
                except Exception:
                    print(
                        "Cleanup could not confirm shutdown. Run: "
                        "uv run python scripts/stop_sandbox.py "
                        f"{sandbox.sandbox_id}",
                        file=sys.stderr,
                        flush=True,
                    )
                    if not failed:
                        raise
                finally:
                    try:
                        (output / "cleanup.json").write_text(
                            json.dumps(cleanup, indent=2) + "\n"
                        )
                    except OSError as exc:
                        print(f"Could not save cleanup status: {exc}", file=sys.stderr)
                        if not failed:
                            raise
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
