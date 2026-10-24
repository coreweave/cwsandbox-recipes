"""Run one SWE-bench Verified instance through slime's coding-agent sandbox pipeline.

Real slime code paths, unchanged: ``generate.boot_agent_sandbox`` (sandbox,
Node 22 and Claude Code install), ``swe.prepare_workspace``,
``ClaudeCodeHarness.run``, ``swe.git_diff`` and ``swe.run_evaluation`` (fresh
grading sandbox, official swebench grading).

Two stand-ins replace the GPU side, so this runs without a trainer:
- ``stub_anthropic.js`` answers Claude Code inside the agent sandbox in place
  of slime's adapter in front of SGLang.
- The dataset's reference patch is applied as the model's edit, so grading
  should resolve the instance. An empty diff is graded too, as a control.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path

from recipe_common import load_swebench_row, remote_env_info, use_slime_checkout

use_slime_checkout()
os.environ.setdefault("SLIME_AGENT_SANDBOX_BACKEND", "slime_cwsandbox.CWSandbox")

import examples.coding_agent_rl.generate as gen  # noqa: E402
import examples.coding_agent_rl.swe as swe  # noqa: E402
from slime.agent.harness import ClaudeCodeHarness  # noqa: E402
from slime.utils.types import Sample  # noqa: E402

HERE = Path(__file__).resolve().parent
STUB_PORT = 18001
results: list[tuple[str, bool]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}".rstrip(), flush=True)


async def main(instance_id: str, eval_timeout: int) -> None:
    row = load_swebench_row(instance_id)
    sample = Sample(
        prompt=row["problem_statement"], label=instance_id, metadata={"remote_env_info": remote_env_info(row)}
    )
    md = swe.get_metadata(sample, swe.PROTOCOL_SWEBENCH)
    reason = swe.evaluability_check(md)
    check("instance is gradable", reason is None, reason or md["image"])
    if reason:
        raise SystemExit(1)
    workdir = md["workdir"]
    session_id = f"swebench-slime-{instance_id}"

    t0 = time.time()
    async with gen.boot_agent_sandbox(md["image"], instance_id) as sb:
        check("agent sandbox booted, Claude Code installed", True, f"{time.time() - t0:.0f}s")
        await swe.prepare_workspace(sb, workdir, md)
        _, out, _ = await sb.exec(f"stat -c %U {workdir}/PROBLEM_STATEMENT.md", user="agent")
        check("swe.prepare_workspace", out.strip() == "agent")

        await sb.write_file("/tmp/stub_anthropic.js", (HERE / "stub_anthropic.js").read_text())
        await sb.exec("setsid node /tmp/stub_anthropic.js > /tmp/stub.log 2>&1 < /dev/null &", check=True)
        port_open = f"(echo > /dev/tcp/127.0.0.1/{STUB_PORT}) 2>/dev/null"
        await sb.exec(f"for i in $(seq 1 40); do {port_open} && exit 0; sleep 0.25; done; exit 1", check=True)
        t = time.time()
        code = await ClaudeCodeHarness().run(
            sb,
            workdir=workdir,
            session_id=session_id,
            adapter_url=f"http://127.0.0.1:{STUB_PORT}",
            time_budget_sec=300,
            prompt=swe.SWE_PROMPT,
        )
        calls = [json.loads(x) for x in (await sb.read_file("/tmp/stub.jsonl")).splitlines() if x.strip()]
        calls = [c for c in calls if c["url"].startswith("/v1/messages")]
        authed = bool(calls) and all(c["auth"] in (f"Bearer {session_id}", session_id) for c in calls)
        check(
            "Claude Code ran as the agent user against the adapter",
            code == 0 and authed,
            f"exit={code} calls={len(calls)} {time.time() - t:.0f}s",
        )

        await sb.write_file("/tmp/model.diff", row["patch"], user="agent")
        await sb.exec(f"cd {workdir} && git apply /tmp/model.diff", user="agent", check=True)
        diff = await swe.git_diff(sb, workdir)
        check(
            "swe.git_diff captured the edit",
            bool(diff.strip()) and "PROBLEM_STATEMENT" not in diff,
            f"{len(diff.splitlines())} lines",
        )

    t = time.time()
    (patched, applied), (empty, _) = await asyncio.gather(
        swe.run_evaluation(md, diff_text=diff, timeout_sec=eval_timeout),
        swe.run_evaluation(md, diff_text="", timeout_sec=eval_timeout),
    )
    check(
        "grading sandbox: reference patch resolves (reward 1.0)", patched == 1.0 and applied, f"{time.time() - t:.0f}s"
    )
    check("grading sandbox: empty diff does not (reward 0.0)", empty == 0.0)
    print(f"\n{sum(ok for _, ok in results)}/{len(results)} passed")
    if not all(ok for _, ok in results):
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--instance-id", default="astropy__astropy-12907")
    parser.add_argument("--eval-timeout", type=int, default=900)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    for noisy in ("httpx", "httpx2", "huggingface_hub", "grpc", "cwsandbox", "datasets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    asyncio.run(main(args.instance_id, args.eval_timeout))
