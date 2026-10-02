"""Install the pinned NeMo RL source without git or OS package installation."""

import argparse
import hashlib
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request


COMMIT = "b1c86a816c5e2b4ca41ece193624a38dc62e6fdf"
URL = f"https://codeload.github.com/NVIDIA-NeMo/RL/tar.gz/{COMMIT}"
SHA256 = "62c4a86e6364fba75d8e8e02839a9723ef4e8a51305b60f320e128cb101ceb51"


def download(url, archive):
    with urllib.request.urlopen(url, timeout=30) as response, archive.open("wb") as out:
        while chunk := response.read(1024 * 1024):
            out.write(chunk)
            print(f"Downloaded {out.tell()} bytes", flush=True)


def download_with_retry(url, archive, *, attempts=3, attempt_seconds=120):
    for attempt in range(1, attempts + 1):
        print(
            f"Downloading NeMo RL {COMMIT}: attempt {attempt}/{attempts} "
            f"(deadline {attempt_seconds}s)",
            flush=True,
        )
        try:
            # A separate process bounds DNS, connection, and streaming time together.
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--download",
                    url,
                    str(archive),
                ],
                check=True,
                timeout=attempt_seconds,
            )
            return
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
            archive.unlink(missing_ok=True)
            print(f"Download attempt {attempt} failed: {exc}", flush=True)
            if attempt == attempts:
                raise RuntimeError(
                    "NeMo RL source download exhausted its retries"
                ) from exc


def extract_verified(archive, target):
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != SHA256:
        raise ValueError(f"NeMo RL archive SHA-256 mismatch: {digest}")
    print(f"Verified NeMo RL archive SHA-256: {digest}", flush=True)
    with tarfile.open(archive, "r:gz") as source:
        members = source.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or not path.parts
                or path.parts[0] != f"RL-{COMMIT}"
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError(f"Unsafe archive member: {member.name}")
        # Copy only regular files and directories, after validating every member.
        for member in members:
            output = target.joinpath(*PurePosixPath(member.name).parts)
            if member.isdir():
                output.mkdir(parents=True, exist_ok=True)
            else:
                output.parent.mkdir(parents=True, exist_ok=True)
                with source.extractfile(member) as data, output.open("wb") as out:
                    shutil.copyfileobj(data, out)
                output.chmod(member.mode & 0o777)


def install(destination):
    if destination.exists():
        raise FileExistsError(f"Source destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        staging = Path(temporary)
        archive = staging / "source.tar.gz"
        download_with_retry(URL, archive)
        extract_verified(archive, staging)
        (staging / f"RL-{COMMIT}").rename(destination)
    print(f"Installed NeMo RL source at {destination}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("/opt/nemo-rl"))
    parser.add_argument("--download", nargs=2, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.download:
        download(args.download[0], Path(args.download[1]))
    else:
        install(args.destination)
