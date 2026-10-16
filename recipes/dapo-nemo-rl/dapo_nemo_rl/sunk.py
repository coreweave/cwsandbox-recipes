"""SSH and Slurm helpers for the SUNK login node. Runs on your machine."""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Sunk:
    host: str  # an ssh host or ~/.ssh/config alias
    key: str | None = None  # ssh private key; uses only this key when set

    def ssh_args(self) -> list[str]:
        args = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20"]
        if self.key:
            args += ["-i", self.key, "-o", "IdentitiesOnly=yes"]
        return args

    def run(self, command: str, *, input: bytes | None = None, check: bool = True) -> str:
        result = subprocess.run([*self.ssh_args(), self.host, command], input=input, capture_output=True, check=False)
        if check and result.returncode != 0:
            raise RuntimeError(f"ssh {self.host} {command!r} failed: {result.stderr.decode(errors='replace')}")
        return result.stdout.decode(errors="replace")

    def write_private(self, path: str, data: bytes) -> None:
        """Write a file only the SUNK user can read (for tokens and tunnel addresses)."""
        self.run(f"umask 077 && cat > {shlex.quote(path)}", input=data)

    def read(self, path: str) -> str | None:
        out = self.run(f"cat {shlex.quote(path)} 2>/dev/null || true")
        return out or None

    def sync(self, sources: list[Path], dest: str) -> None:
        """Copy local directories to ``dest`` on the login node (shared with compute nodes)."""
        self.run(f"mkdir -p {shlex.quote(dest)}")
        ssh = " ".join(shlex.quote(arg) for arg in self.ssh_args())
        subprocess.run(
            ["rsync", "-a", "--delete", "--exclude", "__pycache__", "-e", ssh,
             *[str(source) for source in sources], f"{self.host}:{dest}/"],
            check=True,
        )  # fmt: skip

    def submit(self, sbatch_args: list[str], command: list[str]) -> str:
        wrap = " ".join(shlex.quote(part) for part in command)
        args = " ".join(shlex.quote(arg) for arg in sbatch_args)
        return self.run(f"sbatch --parsable {args} --wrap {shlex.quote(wrap)}").strip().split(";")[0]

    def state(self, job_id: str) -> str:
        """Slurm state of the job, e.g. PENDING, RUNNING, COMPLETED, FAILED."""
        out = self.run(f"sacct -j {job_id} -X -n -P -o State 2>/dev/null | head -1", check=False).strip()
        return out.split()[0] if out else "PENDING"

    def exit_code(self, job_id: str) -> int:
        out = self.run(f"sacct -j {job_id} -X -n -P -o ExitCode | head -1", check=False).strip()
        return int(out.split(":")[0]) if out and out.split(":")[0].isdigit() else 1

    def cancel(self, job_id: str) -> None:
        self.run(f"scancel {job_id}", check=False)

    def tail(self, path: str) -> subprocess.Popen:
        """Follow a log file on SUNK; read lines from ``.stdout``."""
        return subprocess.Popen(
            [*self.ssh_args(), self.host, f"tail -n +1 -F {shlex.quote(path)} 2>/dev/null"],
            stdout=subprocess.PIPE,
            text=True,
        )
