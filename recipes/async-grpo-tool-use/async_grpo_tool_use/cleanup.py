"""Stop this recipe's sandboxes: one run by its tag, or every run and dashboard with --all.

Usage:
    uv run python -m async_grpo_tool_use.cleanup --tag async-grpo-1a2b3c4d             # its sandboxes
    uv run python -m async_grpo_tool_use.cleanup --tag async-grpo-1a2b3c4d-dashboard   # its dashboard
    uv run python -m async_grpo_tool_use.cleanup --all

Reads CWSANDBOX_API_KEY from .env like the other commands.
"""

from __future__ import annotations

import argparse

from async_grpo_tool_use import env_file
from async_grpo_tool_use.dashboard.share import SHARE_TAG
from async_grpo_tool_use.launch import RECIPE_TAG, stop_tagged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tag", help="run tag printed by the launcher")
    group.add_argument("--all", action="store_true", help=f"every sandbox tagged {RECIPE_TAG} or {SHARE_TAG}")
    args = parser.parse_args(argv)
    env_file.load()
    import cwsandbox

    tags = [RECIPE_TAG, SHARE_TAG] if args.all else [args.tag]
    count = sum(stop_tagged(cwsandbox, tag) for tag in tags)
    print(f"[grpo] {count} sandbox(es) stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
