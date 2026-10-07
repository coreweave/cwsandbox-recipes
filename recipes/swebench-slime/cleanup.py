"""Stop every running sandbox carrying this recipe's tags.

Scripts stop their own sandboxes on exit; this catches leftovers from an
interrupted run. Each sandbox also has a server-side lifetime cap.
"""

from __future__ import annotations

import argparse
import os

from cwsandbox import Sandbox


def main() -> None:
    default = os.environ.get("SLIME_AGENT_CWSANDBOX_TAGS", "swebench-slime").split(",")[0].strip()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", action="append", help=f"Tag to match (repeatable). Default: {default}")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    tags = args.tag or [default]

    found = Sandbox.list(tags=tags).result()
    for sb in found:
        print(f"{sb.sandbox_id} {sb.status}")
        if not args.dry_run:
            sb.stop(graceful_shutdown_seconds=0, missing_ok=True).result()
    print(f"{'found' if args.dry_run else 'stopped'} {len(found)} sandbox(es) tagged {tags}")


if __name__ == "__main__":
    main()
