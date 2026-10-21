"""Entrypoint of the trainer GPU sandbox (``TRAINER=sandbox``), the counterpart of ``sunk_job``.

1. Forward the Gym sandbox's agent ports 21002-21004 to 127.0.0.1 here, over a Tailcat tunnel.
2. Once the launcher passes the samplers' tunnel addresses, forward each one's vLLM and weight
   receiver too.
3. Run ``run_trainer`` on this sandbox's GPU. ``/job`` is ``/workspace/run``, where the launcher
   writes the files it passes to the trainer.

Usage: /opt/nemo_rl_venv/bin/python -m async_grpo_tool_use.trainer_sandbox [hydra overrides...]
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

from async_grpo_tool_use import tunnel
from async_grpo_tool_use.procs import log, read_env_file, spawn, supervise, wait_any, wait_for_file
from async_grpo_tool_use.sampler import forwards
from async_grpo_tool_use.sunk_job import GYM_PORTS, TRAINER_PYTHON

RUN_DIR = Path("/workspace/run")
EXIT_LINE = "trainer: run_trainer exited with "


def forward_sampler(tailcat: Path) -> None:
    inference = read_env_file(wait_for_file(RUN_DIR / "inference.env"))
    for replica, address in enumerate(inference["SAMPLER_ADDRS"].split(",")):
        tunnel.forward(tailcat, address, forwards(replica, receiver=True), RUN_DIR)


def main(overrides: list[str]) -> int:
    supervise()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if not Path("/job").exists():
        Path("/job").symlink_to(RUN_DIR)
    Path("/work").mkdir(exist_ok=True)
    os.environ.setdefault("REFIT_EXPORT_DIR", "/dev/shm/refit")
    tailcat = tunnel.install(Path("/opt/tailcat"))
    tunnel.forward(tailcat, wait_for_file(RUN_DIR / "gym.addr").read_text().strip(), list(GYM_PORTS), RUN_DIR)
    threading.Thread(target=forward_sampler, args=(tailcat,), daemon=True).start()
    log(f"trainer: sandbox {os.uname().nodename}")
    code = wait_any([spawn([TRAINER_PYTHON, "-m", "async_grpo_tool_use.run_trainer", *overrides])])
    # The sandbox API does not always report an exit code; the launcher reads this line instead.
    log(f"{EXIT_LINE}{code}")
    return code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
