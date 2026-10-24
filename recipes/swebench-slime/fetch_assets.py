"""Download the host-side tarballs slime's Claude Code harness uploads into each sandbox.

- Node 22 linux-x64 runtime, verified against nodejs.org SHASUMS256.txt.
- The Claude Code npm package, via ``npm pack``. Installing it in the sandbox
  pulls the platform binary from the npm registry, so the agent sandbox needs
  outbound access to registry.npmjs.org.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import urllib.request
from pathlib import Path

NODE_INDEX = "https://nodejs.org/dist/latest-v22.x/"


def fetch_node(dest: Path) -> Path:
    with urllib.request.urlopen(NODE_INDEX + "SHASUMS256.txt", timeout=60) as resp:
        sums = resp.read().decode()
    sha, name = next(line.split() for line in sums.splitlines() if line.endswith("-linux-x64.tar.xz"))
    path = dest / name
    if not path.exists():
        tmp = path.with_suffix(".partial")
        urllib.request.urlretrieve(NODE_INDEX + name, tmp)
        tmp.rename(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != sha:
        path.unlink()
        raise SystemExit(f"{name}: sha256 mismatch ({digest} != {sha})")
    return path


def fetch_claude_code(dest: Path, version: str) -> Path:
    out = subprocess.run(
        ["npm", "pack", f"@anthropic-ai/claude-code@{version}", "--pack-destination", str(dest)],
        check=True,
        capture_output=True,
        text=True,
    )
    return dest / out.stdout.strip().splitlines()[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", type=Path, default=Path("assets"))
    parser.add_argument("--claude-code-version", default="2.1.292")
    args = parser.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)
    node = fetch_node(args.dest)
    cli = fetch_claude_code(args.dest, args.claude_code_version)
    print(f"SLIME_AGENT_NODE_TARBALL={node.resolve()}")
    print(f"SLIME_AGENT_CC_TARBALL={cli.resolve()}")


if __name__ == "__main__":
    main()
