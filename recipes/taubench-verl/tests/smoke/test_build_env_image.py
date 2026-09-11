"""Exercise image build/test/publish ordering without a Docker daemon."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("push,fail_test", [(False, False), (True, False), (True, True)])
def test_image_is_tested_before_optional_publication(tmp_path, push, fail_test):
    log = tmp_path / "docker-calls.jsonl"
    docker = tmp_path / "docker"
    docker.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['DOCKER_TEST_LOG'], 'a') as f:\n"
        "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if sys.argv[1] == 'run' and os.environ['FAIL_IMAGE_TEST'] == '1':\n"
        "    sys.exit(9)\n"
    )
    docker.chmod(0o755)
    args = ["bash", str(ROOT / "scripts/build_env_image.sh"), "registry.example/taubench-env:test"]
    if push:
        args.append("--push")
    result = subprocess.run(args, cwd=tmp_path, env={
        **os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "DOCKER_TEST_LOG": str(log), "FAIL_IMAGE_TEST": str(int(fail_test)),
    }, capture_output=True, text=True)
    assert result.returncode == (9 if fail_test else 0), result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert calls[0][:2] == ["buildx", "build"]
    assert "linux/amd64" in calls[0]
    assert "--load" in calls[0]
    assert calls[0][-1] == str(ROOT / "docker/taubench-env")
    assert calls[1][0] == "run"
    assert "--network" in calls[1] and "none" in calls[1]
    assert [call for call in calls if call[0] == "push"] == (
        [["push", "registry.example/taubench-env:test"]] if push and not fail_test else []
    )
