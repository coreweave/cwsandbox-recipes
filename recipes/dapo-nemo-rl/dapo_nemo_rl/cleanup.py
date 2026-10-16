"""Stop this recipe's sandboxes: one run by its tag, or every run with --all.

Usage:
    python -m dapo_nemo_rl.cleanup --tag dapo-nemo-rl-1a2b3c4d
    python -m dapo_nemo_rl.cleanup --all
"""

from __future__ import annotations

import argparse
from typing import Any

from dapo_nemo_rl.launch import RECIPE_TAG


def cleanup(tag: str, sdk: Any) -> int:
    sandboxes = sdk.Sandbox.list(tags=[tag]).result()
    for sandbox in sandboxes:
        sandbox.stop(missing_ok=True).result()
        print(f"[dapo] stopped {sandbox.sandbox_id}", flush=True)
    return len(sandboxes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tag", help="run tag printed by the launcher")
    group.add_argument("--all", action="store_true", help=f"every sandbox tagged {RECIPE_TAG}")
    args = parser.parse_args(argv)
    import cwsandbox

    count = cleanup(RECIPE_TAG if args.all else args.tag, cwsandbox)
    print(f"[dapo] {count} sandbox(es) stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
