import asyncio
from types import SimpleNamespace

import pytest
from cwsandbox import AuthStrategy

import demo
from warm_pool import PROBE


def test_cks_with_wandb_fails_before_network_access(monkeypatch):
    monkeypatch.setattr(demo, "AUTH", AuthStrategy.WANDB)
    monkeypatch.setattr(
        demo,
        "SandboxDefaults",
        lambda **kwargs: SimpleNamespace(
            placement_mode="cks",
            **{key: value for key, value in kwargs.items() if key != "placement_mode"},
        ),
    )

    def unexpected_session(**kwargs):
        pytest.fail("invalid authentication reached Session creation")

    monkeypatch.setattr(demo, "Session", unexpected_session)
    with pytest.raises(ValueError, match="CKS placement requires AuthStrategy.COREWEAVE_API_KEY"):
        asyncio.run(demo.main())


def test_on_demand_has_one_probe_and_two_workload_commands(monkeypatch):
    created = []

    class FakeSandbox:
        def __init__(self):
            self.sandbox_id = str(len(created))
            self.commands = []

        async def exec(self, command, **kwargs):
            self.commands.append(command)
            return SimpleNamespace(stdout="fresh workspace")

        async def stop(self, **kwargs):
            pass

    class FakeSession:
        def __init__(self, **kwargs):
            pass

        def sandbox(self):
            sandbox = FakeSandbox()
            created.append(sandbox)
            return sandbox

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr(demo, "Session", FakeSession)
    monkeypatch.setattr(demo, "REQUESTS", 1)
    monkeypatch.setattr(demo, "POOL_SIZE", 1)
    monkeypatch.setattr(demo, "CONCURRENCY", 1)
    asyncio.run(demo.main())
    assert created[0].commands == [
        PROBE,
        ["python", "-c", demo.WORKLOAD],
        ["test", "-f", "/tmp/request-state"],
    ]
