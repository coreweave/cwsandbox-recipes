"""Tests for production scale controls wired through environment variables.

No real sandboxes or network. Asserts observable values on constructed pools,
backends, and transports rather than source text.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from helpers import ScriptedSimulator
from omegaconf import OmegaConf

from verl_taubench.sandbox.env_config import (
    TAUBENCH_ENV_IMAGE_ENV,
    TAUBENCH_MAX_CONNECTIONS_ENV,
    TAUBENCH_POOL_SIZE_ENV,
    resolve_backend_image,
    resolve_max_connections,
    resolve_placement_mode,
    resolve_placement_spillover,
    resolve_pool_size,
    resolve_runner_ids,
)
from verl_taubench.tools.sandbox_taubench_tool import SandboxTauBenchTool

REPO_ROOT = Path(__file__).resolve().parents[2]
SANDBOX_TOOL_CONFIG = REPO_ROOT / "config" / "tool_config" / "taubench_sandbox_tool_config.yaml"


def _resolved_backend_config() -> Any:
    cfg = OmegaConf.load(SANDBOX_TOOL_CONFIG)
    OmegaConf.resolve(cfg)
    return cfg.tools[0].config.backend


@pytest.mark.parametrize("cpu,memory", [(None, None), ("6", "12Gi")])
def test_cpu_resources_reach_constructed_backend(monkeypatch, capture_pool_build, cpu, memory):
    for key, value in (("TAUBENCH_ENV_CPU", cpu), ("TAUBENCH_ENV_MEMORY", memory)):
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    cfg = OmegaConf.to_container(OmegaConf.load(SANDBOX_TOOL_CONFIG), resolve=True)
    _, built = capture_pool_build["build"](cfg["tools"][0]["config"])
    assert built["backend"].kwargs["cpu"] == (cpu or "4")
    assert built["backend"].kwargs["memory"] == (memory or "8Gi")


@pytest.mark.parametrize("configured,cli,expected", [
    (False, [], ("4", "8Gi")),
    (True, [], ("6", "12Gi")),
    (True, ["--cpu", "2", "--memory", "4Gi"], ("2", "4Gi")),
])
def test_preflight_uses_same_resource_settings(monkeypatch, configured, cli, expected):
    from verl_taubench.sandbox import preflight

    for name, value in (("TAUBENCH_ENV_CPU", "6"), ("TAUBENCH_ENV_MEMORY", "12Gi")):
        if configured:
            monkeypatch.setenv(name, value)
        else:
            monkeypatch.delenv(name, raising=False)
    observed = []

    async def record_probe(args):
        observed.append((args.cpu, args.memory))
        return 0

    monkeypatch.setattr(preflight, "_run", record_probe)
    assert preflight.main(cli) == 0
    assert observed == [expected]


def test_tool_config_domain_split_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAUBENCH_DOMAIN", "airline")
    monkeypatch.setenv("TAUBENCH_TASK_SPLIT", "test")
    backend = _resolved_backend_config()
    assert backend.domain == "airline"
    assert backend.task_split == "test"


def test_tool_config_domain_split_defaults_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TAUBENCH_DOMAIN", raising=False)
    monkeypatch.delenv("TAUBENCH_TASK_SPLIT", raising=False)
    backend = _resolved_backend_config()
    assert backend.domain == "retail"
    assert backend.task_split == "train"


def test_tool_config_propagates_generated_environment_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TAUBENCH_ENV_TAG", "verl-taubench-env-run-123")
    backend = _resolved_backend_config()
    assert list(backend.tags) == ["verl-taubench-env-run-123"]


def test_tool_config_uses_serverless_placement(monkeypatch: pytest.MonkeyPatch) -> None:
    # Clusters without Gateway API reject HTTPS-endpoint creates on CKS with a
    # non-spillover error, so the CPU pool goes straight to serverless (where
    # cks_then_serverless spillover does not apply and must resolve to None).
    monkeypatch.delenv("CWSANDBOX_PLACEMENT_MODE", raising=False)
    monkeypatch.delenv("CWSANDBOX_PLACEMENT_SPILLOVER", raising=False)
    backend = _resolved_backend_config()
    assert backend.placement_mode == "serverless"
    assert backend.placement_spillover is None


def _minimal_config(*, pool_size: int = 8, max_connections: int | None = None) -> dict:
    pool: dict[str, Any] = {"size": pool_size}
    if max_connections is not None:
        pool["max_connections"] = max_connections
    return {
        "backend": {
            "domain": "retail",
            "task_split": "train",
            "container_image": "python:3.11",
        },
        "pool": pool,
    }


@pytest.fixture
def capture_pool_build(monkeypatch):
    """Record kwargs passed to CwSandboxBackend and HttpxTransport."""
    captured: dict[str, Any] = {}

    class RecordingTransport:
        instances: list["RecordingTransport"] = []

        def __init__(self, *, timeout: float = 30.0, max_connections: int = 512):
            self.timeout = timeout
            self.max_connections = max_connections
            RecordingTransport.instances.append(self)

        async def request(self, *args, **kwargs):
            raise NotImplementedError

        async def aclose(self) -> None:
            return None

    class RecordingBackend:
        instances: list["RecordingBackend"] = []

        def __init__(self, **kwargs: Any):
            self.kwargs = kwargs
            self.preload_calls = 0
            RecordingBackend.instances.append(self)

        def preload_sdk(self) -> None:
            self.preload_calls += 1

        async def provision(self):
            raise NotImplementedError

        async def destroy(self, handle) -> None:
            return None

    RecordingTransport.instances = []
    RecordingBackend.instances = []

    monkeypatch.setattr(
        "verl_taubench.tools.sandbox_taubench_tool.HttpxTransport",
        RecordingTransport,
    )
    monkeypatch.setattr(
        "verl_taubench.tools.sandbox_taubench_tool.CwSandboxBackend",
        RecordingBackend,
    )

    def build(config: dict | None = None) -> tuple[SandboxTauBenchTool, dict[str, Any]]:
        tool = SandboxTauBenchTool(config=config or _minimal_config())
        return tool, {
            "transport": RecordingTransport.instances[-1],
            "backend": RecordingBackend.instances[-1],
            "pool": tool.pool,
        }

    captured["build"] = build
    return captured


@pytest.fixture
def recording_internal_pools(monkeypatch):
    """Replace only cloud/network boundaries while exercising real tool routing."""

    class RecordingTransport:
        def __init__(self, **kwargs: Any):
            self.kwargs = kwargs

        async def aclose(self) -> None:
            return None

    class RecordingBackend:
        def __init__(self, **kwargs: Any):
            self.kwargs = kwargs
            self.preload_calls = 0

        def preload_sdk(self) -> None:
            self.preload_calls += 1

    class RecordingClient:
        async def spec(self) -> dict[str, Any]:
            return {"wiki": "", "rules": [], "tools_info": []}

    class RecordingPool:
        instances: list["RecordingPool"] = []

        def __init__(self, backend: RecordingBackend, transport: RecordingTransport, **kwargs: Any):
            self.backend = backend
            self.transport = transport
            self.kwargs = kwargs
            self.prewarm_calls = 0
            self.acquire_calls = 0
            self.releases: list[tuple[Any, bool]] = []
            self.close_calls = 0
            self.closed = False
            RecordingPool.instances.append(self)

        async def prewarm(self) -> None:
            self.prewarm_calls += 1
            await asyncio.sleep(0)

        async def acquire(self, task_index: int, episode_id: str) -> Any:
            self.acquire_calls += 1
            return SimpleNamespace(
                task_index=task_index,
                episode_id=episode_id,
                metadata={
                    "instruction": f"instruction-{task_index}",
                    "domain": self.backend.kwargs["domain"],
                },
                client=RecordingClient(),
            )

        async def release(self, lease: Any, *, healthy: bool = True) -> None:
            self.releases.append((lease, healthy))

        async def aclose(self) -> None:
            self.close_calls += 1
            self.closed = True

    RecordingPool.instances = []
    monkeypatch.setattr(
        "verl_taubench.tools.sandbox_taubench_tool.HttpxTransport",
        RecordingTransport,
    )
    monkeypatch.setattr(
        "verl_taubench.tools.sandbox_taubench_tool.CwSandboxBackend",
        RecordingBackend,
    )
    monkeypatch.setattr(
        "verl_taubench.tools.sandbox_taubench_tool.SandboxPool",
        RecordingPool,
    )
    return RecordingPool


# -- env_config unit tests -------------------------------------------------


def test_resolve_pool_size_from_env(monkeypatch) -> None:
    monkeypatch.setenv(TAUBENCH_POOL_SIZE_ENV, "128")
    assert resolve_pool_size(64) == 128


def test_resolve_pool_size_falls_back_to_config(monkeypatch) -> None:
    monkeypatch.delenv(TAUBENCH_POOL_SIZE_ENV, raising=False)
    assert resolve_pool_size(32) == 32


def test_resolve_pool_size_rejects_zero(monkeypatch) -> None:
    monkeypatch.setenv(TAUBENCH_POOL_SIZE_ENV, "0")
    with pytest.raises(ValueError, match="positive integer"):
        resolve_pool_size(8)


def test_resolve_pool_size_rejects_invalid(monkeypatch) -> None:
    monkeypatch.setenv(TAUBENCH_POOL_SIZE_ENV, "not-a-number")
    with pytest.raises(ValueError, match="positive integer"):
        resolve_pool_size(8)


def test_resolve_max_connections_scales_to_pool_size(monkeypatch) -> None:
    monkeypatch.delenv(TAUBENCH_MAX_CONNECTIONS_ENV, raising=False)
    assert resolve_max_connections(600) == 600
    assert resolve_max_connections(128) == 512
    assert resolve_max_connections(8) == 512


def test_resolve_max_connections_env_override(monkeypatch) -> None:
    monkeypatch.setenv(TAUBENCH_MAX_CONNECTIONS_ENV, "1024")
    assert resolve_max_connections(64) == 1024


def test_resolve_max_connections_clamps_low_env_override(monkeypatch) -> None:
    monkeypatch.setenv(TAUBENCH_MAX_CONNECTIONS_ENV, "16")
    assert resolve_max_connections(64) == 512
    assert resolve_max_connections(600) == 600


def test_resolve_max_connections_rejects_zero(monkeypatch) -> None:
    monkeypatch.setenv(TAUBENCH_MAX_CONNECTIONS_ENV, "0")
    with pytest.raises(ValueError, match="positive integer"):
        resolve_max_connections(64)


def test_resolve_backend_image_from_env(monkeypatch) -> None:
    monkeypatch.setenv(TAUBENCH_ENV_IMAGE_ENV, "registry.example/taubench:1.0")
    image, install = resolve_backend_image("python:3.11")
    assert image == "registry.example/taubench:1.0"
    assert install is False


def test_resolve_backend_image_default_installs(monkeypatch) -> None:
    monkeypatch.delenv(TAUBENCH_ENV_IMAGE_ENV, raising=False)
    image, install = resolve_backend_image("python:3.11")
    assert image == "python:3.11"
    assert install is True


def test_resolve_placement_mode_defaults_to_serverless(monkeypatch) -> None:
    monkeypatch.delenv("CWSANDBOX_PLACEMENT_MODE", raising=False)
    assert resolve_placement_mode() == "serverless"
    # The trainer opts into CKS explicitly; env still wins over that.
    assert resolve_placement_mode("cks") == "cks"


def test_resolve_placement_spillover_defaults_to_cks_then_serverless(monkeypatch) -> None:
    monkeypatch.delenv("CWSANDBOX_PLACEMENT_SPILLOVER", raising=False)
    assert resolve_placement_spillover() == "cks_then_serverless"


def test_resolve_placement_spillover_is_none_under_serverless(monkeypatch) -> None:
    """Regression: the SDK rejects cks_then_serverless with placement_mode=serverless
    ("requires placement_mode=cks"), so serverless must resolve to no spillover."""
    monkeypatch.delenv("CWSANDBOX_PLACEMENT_SPILLOVER", raising=False)
    assert resolve_placement_spillover(placement_mode="serverless") is None
    # A stale explicit cks_then_serverless (e.g. copied from .env.example) is
    # dropped rather than failing every create.
    monkeypatch.setenv("CWSANDBOX_PLACEMENT_SPILLOVER", "cks_then_serverless")
    assert resolve_placement_spillover(placement_mode="serverless") is None
    # Compatible explicit values still pass through.
    monkeypatch.setenv("CWSANDBOX_PLACEMENT_SPILLOVER", "strict")
    assert resolve_placement_spillover(placement_mode="serverless") == "strict"


def test_resolve_runner_ids_from_env(monkeypatch) -> None:
    monkeypatch.setenv("CWSANDBOX_RUNNER_IDS", "cluster-a, cluster-b")
    assert resolve_runner_ids() == ["cluster-a", "cluster-b"]


# -- tool pool construction ------------------------------------------------


def test_build_pool_env_size_override(capture_pool_build, monkeypatch) -> None:
    monkeypatch.setenv(TAUBENCH_POOL_SIZE_ENV, "96")
    _, built = capture_pool_build["build"](_minimal_config(pool_size=8))
    assert built["pool"].size == 96


def test_build_pool_config_size_when_env_unset(capture_pool_build, monkeypatch) -> None:
    monkeypatch.delenv(TAUBENCH_POOL_SIZE_ENV, raising=False)
    _, built = capture_pool_build["build"](_minimal_config(pool_size=24))
    assert built["pool"].size == 24
    assert built["backend"].preload_calls == 1


def test_build_pool_invalid_env_size_raises(capture_pool_build, monkeypatch) -> None:
    monkeypatch.setenv(TAUBENCH_POOL_SIZE_ENV, "0")
    with pytest.raises(ValueError, match="positive integer"):
        capture_pool_build["build"](_minimal_config())


def test_build_pool_env_image_overrides_and_disables_install(
    capture_pool_build, monkeypatch
) -> None:
    monkeypatch.setenv(TAUBENCH_ENV_IMAGE_ENV, "registry.example/taubench:baked")
    _, built = capture_pool_build["build"](_minimal_config())
    assert built["backend"].kwargs["container_image"] == "registry.example/taubench:baked"
    assert built["backend"].kwargs["install_tau_bench"] is False


def test_build_pool_normal_mode_installs_tau_bench(capture_pool_build, monkeypatch) -> None:
    monkeypatch.delenv(TAUBENCH_ENV_IMAGE_ENV, raising=False)
    _, built = capture_pool_build["build"](_minimal_config())
    assert built["backend"].kwargs["container_image"] == "python:3.11"
    assert built["backend"].kwargs["install_tau_bench"] is True


def test_build_pool_max_connections_scales_with_pool_size(
    capture_pool_build, monkeypatch
) -> None:
    monkeypatch.delenv(TAUBENCH_MAX_CONNECTIONS_ENV, raising=False)
    monkeypatch.delenv(TAUBENCH_POOL_SIZE_ENV, raising=False)
    _, built = capture_pool_build["build"](_minimal_config(pool_size=600))
    assert built["transport"].max_connections == 600


def test_build_pool_max_connections_respects_env_override(
    capture_pool_build, monkeypatch
) -> None:
    monkeypatch.setenv(TAUBENCH_MAX_CONNECTIONS_ENV, "2048")
    _, built = capture_pool_build["build"](_minimal_config(pool_size=64))
    assert built["transport"].max_connections == 2048


@pytest.mark.asyncio
async def test_internal_tool_uses_keyed_train_and_validation_pools(
    recording_internal_pools,
) -> None:
    tool = SandboxTauBenchTool(
        config=_minimal_config(pool_size=2),
        simulator=ScriptedSimulator(replies=["ok"]),
    )
    assert len(recording_internal_pools.instances) == 1, "default train pool must build eagerly"

    train_id, _ = await tool.create(domain="retail", task_split="train", task_index=1)
    test_id, _ = await tool.create(domain="retail", task_split="test", task_index=2)

    train_pool, test_pool = recording_internal_pools.instances
    assert train_pool.backend.kwargs["task_split"] == "train"
    assert test_pool.backend.kwargs["task_split"] == "test"
    assert tool.instance_registry[train_id]["pool"] is train_pool
    assert tool.instance_registry[test_id]["pool"] is test_pool
    assert train_pool.prewarm_calls == 1
    assert test_pool.prewarm_calls == 1

    train_lease = tool.instance_registry[train_id]["lease"]
    test_lease = tool.instance_registry[test_id]["lease"]
    await tool.release(train_id)
    await tool.release(test_id)
    assert train_pool.releases == [(train_lease, True)]
    assert test_pool.releases == [(test_lease, True)]


@pytest.mark.asyncio
async def test_concurrent_first_use_builds_and_prewarms_keyed_pool_once(
    recording_internal_pools,
) -> None:
    tool = SandboxTauBenchTool(
        config=_minimal_config(pool_size=2),
        simulator=ScriptedSimulator(replies=["ok"]),
    )

    created = await asyncio.gather(
        tool.create(domain="retail", task_split="test", task_index=1),
        tool.create(domain="retail", task_split="test", task_index=2),
    )

    test_pools = [
        pool
        for pool in recording_internal_pools.instances
        if pool.backend.kwargs["task_split"] == "test"
    ]
    assert len(test_pools) == 1
    assert test_pools[0].prewarm_calls == 1
    for instance_id, _ in created:
        await tool.release(instance_id)


@pytest.mark.asyncio
async def test_close_wins_before_first_use_pool_build(
    recording_internal_pools,
) -> None:
    tool = SandboxTauBenchTool(
        config=_minimal_config(pool_size=2),
        simulator=ScriptedSimulator(replies=["ok"]),
    )
    default_pool = recording_internal_pools.instances[0]

    await tool._pools_lock.acquire()
    close_task = asyncio.create_task(tool.aclose())
    await asyncio.sleep(0)
    create_task = asyncio.create_task(
        tool.create(domain="retail", task_split="test", task_index=1)
    )
    await asyncio.sleep(0)
    tool._pools_lock.release()

    await close_task
    with pytest.raises(RuntimeError, match="closed"):
        await create_task

    assert recording_internal_pools.instances == [default_pool]
    assert default_pool.closed is True
    assert default_pool.acquire_calls == 0
    assert ("retail", "test") not in tool._pools


@pytest.mark.asyncio
async def test_close_wins_while_default_pool_prewarm_waits(
    recording_internal_pools,
) -> None:
    tool = SandboxTauBenchTool(
        config=_minimal_config(pool_size=2),
        simulator=ScriptedSimulator(replies=["ok"]),
    )
    default_pool = recording_internal_pools.instances[0]

    await tool._pools_lock.acquire()
    close_task = asyncio.create_task(tool.aclose())
    await asyncio.sleep(0)
    create_task = asyncio.create_task(
        tool.create(domain="retail", task_split="train", task_index=1)
    )
    await asyncio.sleep(0)
    tool._pools_lock.release()

    await close_task
    with pytest.raises(RuntimeError, match="closed"):
        await create_task

    assert default_pool.closed is True
    assert default_pool.prewarm_calls == 0
    assert default_pool.acquire_calls == 0


@pytest.mark.asyncio
async def test_tool_aclose_closes_each_owned_pool_once(recording_internal_pools) -> None:
    tool = SandboxTauBenchTool(
        config=_minimal_config(pool_size=2),
        simulator=ScriptedSimulator(replies=["ok"]),
    )
    instance_id, _ = await tool.create(
        domain="retail", task_split="test", task_index=1
    )
    await tool.release(instance_id)

    await tool.aclose()
    await tool.aclose()

    assert len(recording_internal_pools.instances) == 2
    assert [pool.close_calls for pool in recording_internal_pools.instances] == [1, 1]
