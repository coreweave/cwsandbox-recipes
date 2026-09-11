"""Reap CPU environment sandboxes for one exact training-run tag."""

from __future__ import annotations

import argparse
import os
import time
from typing import Optional

from verl_taubench.sandbox.env_config import CWSANDBOX_SERVERLESS_AUTH_ENV, resolve_placement_mode
from verl_taubench.sandbox.pool import CwSandboxBackend
from verl_taubench.sandbox.tags import RECIPE_TAG


def cleanup_tag(tag: str, *, max_passes: int = 4, settle_seconds: float = 20.0) -> int:
    """Stop sandboxes carrying ``tag``; empty tags are never broadened.

    A single reap pass misses creates that are still in flight on pool worker
    threads when the trainer exits: they land after the pass listed and then
    live out max_lifetime_seconds unbilled-for-nothing. Sweep repeatedly until
    a pass finds no survivors (or the pass budget runs out).
    """
    if not tag or not tag.strip():
        raise ValueError("cleanup tag must be non-empty")
    backend = CwSandboxBackend(
        tags=[tag],
        placement_mode=resolve_placement_mode(),
    )
    total = 0
    for pass_index in range(max_passes):
        stopped = backend.reap_orphans()
        total += stopped
        # Nothing found: either nothing leaked, or the previous pass already
        # cleared everything. Either way there is nothing left to settle for.
        if stopped == 0:
            break
        if pass_index < max_passes - 1:
            time.sleep(settle_seconds)
    return total


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stop τ-bench CPU sandboxes for one exact run tag."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tag", help="Exact TAUBENCH_ENV_TAG to reap.")
    group.add_argument(
        "--all",
        action="store_true",
        help=(
            f"Reap every sandbox tagged {RECIPE_TAG!r} (this recipe's stable tag). "
            "Use after a crash when the run tag is lost; it does not touch other "
            "workloads' sandboxes."
        ),
    )
    parser.add_argument(
        "--cwsandbox-auth",
        action="store_true",
        help="Ignored; cleanup selects cwsandbox auth from the placement mode.",
    )
    parser.add_argument("--serverless-auth", choices=("wandb", "coreweave"), default=None)
    args = parser.parse_args(argv)
    if args.serverless_auth is not None:
        os.environ[CWSANDBOX_SERVERLESS_AUTH_ENV] = args.serverless_auth
    tag = RECIPE_TAG if args.all else args.tag
    stopped = cleanup_tag(tag)
    print(f"[taubench] cleanup: stopped {stopped} sandbox(es) tagged {tag}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
