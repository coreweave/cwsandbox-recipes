"""Download the model at a pinned revision. Skip the download if it is already done.

The trainer and vLLM both load this copy.

Usage: python -m dapo_nemo_rl.model_prep --repo <hf repo> --revision <sha> --out <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path


def prepare(repo: str, revision: str, out: Path) -> Path:
    from huggingface_hub import snapshot_download

    done = out / ".prepared"
    if done.is_file() and done.read_text().strip() == f"{repo}@{revision}":
        print(f"[dapo] model already at {out}", flush=True)
        return out
    print(f"[dapo] downloading {repo}@{revision[:12]}", flush=True)
    snapshot_download(repo, revision=revision, local_dir=out, max_workers=16)
    done.write_text(f"{repo}@{revision}\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.repo, args.revision, args.out)


if __name__ == "__main__":
    main()
