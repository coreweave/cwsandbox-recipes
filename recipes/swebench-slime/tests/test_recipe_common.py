from types import SimpleNamespace

import pytest

from recipe_common import SWEBENCH_FIELDS, pick_gpu_type, remote_env_info, swebench_image


def test_swebench_image_matches_harness_naming():
    assert swebench_image("astropy__astropy-12907") == "swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest"


def test_remote_env_info_carries_grader_fields_image_and_workdir():
    row = {key: f"v-{key}" for key in SWEBENCH_FIELDS} | {"instance_id": "a__b-1", "patch": "ignored"}
    info = remote_env_info(row)
    assert info["image"] == "swebench/sweb.eval.x86_64.a_1776_b-1:latest"
    assert info["workdir"] == "/testbed"
    assert "patch" not in info
    assert all(key in info for key in SWEBENCH_FIELDS)


def runners(*types_):
    return [SimpleNamespace(supported_gpu_types=t) for t in types_]


def test_pick_gpu_type_prefers_requested_and_defaults_to_first_advertised():
    found = runners((), ("gpu-a",), ("gpu-b", "gpu-a"))
    assert pick_gpu_type(found) == "gpu-a"
    assert pick_gpu_type(found, "gpu-b") == "gpu-b"


def test_pick_gpu_type_rejects_unadvertised_or_missing():
    with pytest.raises(SystemExit, match="not advertised"):
        pick_gpu_type(runners(("gpu-a",)), "B40")
    with pytest.raises(SystemExit, match="No runner"):
        pick_gpu_type(runners((), None))
