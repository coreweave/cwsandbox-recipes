"""Exercise slime's sandbox protocol against CoreWeave Sandboxes through slime's own selector.

Each check uses one sandbox from ``make_sandbox`` (the selector slime's
coding-agent example calls), so a pass here means slime's rollout code will see
the behavior it expects.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile
import time
from pathlib import Path

from recipe_common import use_slime_checkout

use_slime_checkout()
os.environ.setdefault("SLIME_AGENT_SANDBOX_BACKEND", "slime_cwsandbox.CWSandbox")

from slime.agent.sandbox import Sandbox, ensure_agent_user, exec_and_wait, make_sandbox  # noqa: E402

from slime_cwsandbox import CWSandbox  # noqa: E402

results: list[tuple[str, bool]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}".rstrip(), flush=True)


async def main(big: Path) -> None:
    big_sha = hashlib.sha256(big.read_bytes()).hexdigest()
    t0 = time.time()
    sb = make_sandbox("python:3.12")
    check("make_sandbox returns CWSandbox", isinstance(sb, CWSandbox) and isinstance(sb, Sandbox))
    async with sb:
        check("sandbox running", bool(sb.sandbox_id), f"{sb.sandbox_id} in {time.time() - t0:.0f}s")

        code, out, _ = await sb.exec("id -un")
        check("exec as root", code == 0 and out.strip() == "root")

        _, out, _ = await sb.exec('echo "$FOO|$BAR"', env={"FOO": "a b", "BAR": "x=y"})
        check("exec env", out.strip() == "a b|x=y", repr(out.strip()))

        await sb.exec("mkdir -p /workspace/repo", check=True)
        await ensure_agent_user(sb, "/workspace/repo")
        _, out, _ = await sb.exec("id -un; echo $HOME", user="agent")
        check("exec as agent", out.split() == ["agent", "/home/agent"], repr(out.strip()))

        await sb.write_file("/workspace/repo/NOTE.md", "hello\n", user="agent")
        _, out, _ = await sb.exec("stat -c %U /workspace/repo/NOTE.md")
        check("write_file as agent is owned by agent", out.strip() == "agent")

        t = time.time()
        await sb.write_file("/tmp/big.bin", big)
        _, out, _ = await sb.exec("sha256sum /tmp/big.bin | cut -d' ' -f1")
        check(f"stream {big.stat().st_size >> 20} MiB host file", out.strip() == big_sha, f"{time.time() - t:.1f}s")

        check("read_file", await sb.read_file("/workspace/repo/NOTE.md") == "hello\n")
        check("read_file missing returns ''", await sb.read_file("/no/such/file") == "")

        try:
            await sb.exec("exit 7", check=True)
            check("check=True raises", False)
        except RuntimeError as e:
            check("check=True raises", "exit=7" in str(e))

        t = time.time()
        code, out = await exec_and_wait(
            sb,
            cmd="sleep 70; echo detached-ok; (exit 3)",
            user="agent",
            workdir="/workspace/repo",
            time_budget_sec=300,
            tag="check",
            want_output=True,
        )
        check("exec_and_wait detached job", code == 3 and "detached-ok" in out, f"{time.time() - t:.0f}s")
    print(f"\n{sum(ok for _, ok in results)}/{len(results)} passed")
    if not all(ok for _, ok in results):
        raise SystemExit(1)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        blob = Path(tmp) / "big.bin"
        blob.write_bytes(os.urandom(40 << 20))
        asyncio.run(main(blob))
