"""Model code never shares a sandbox with the grader or with a later rollout (tir_app.py).

The Gym server's dependencies (NeMo-Gym, FastAPI, Weave, Agent Lens) live in the Gym sandbox
image, not in this project, so the few names tir_app.py imports from them are stubbed here.
"""

import asyncio
import json
import sys
import types
from pathlib import Path

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]


def _stub(name: str, **attrs) -> None:
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules.setdefault(name, module)


class _Server(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    def model_post_init(self, context) -> None:
        pass


_stub("nemo_gym")
_stub(
    "nemo_gym.base_resources_server",
    BaseResourcesServerConfig=BaseModel, BaseRunRequest=BaseModel, BaseVerifyRequest=BaseModel,
    BaseVerifyResponse=BaseModel, BaseSeedSessionRequest=BaseModel, BaseSeedSessionResponse=BaseModel,
    SimpleResourcesServer=_Server,
)  # fmt: skip
_stub("nemo_gym.sandbox", AsyncSandbox=object, SandboxSpec=dict)
_stub("nemo_gym.server_utils", SESSION_ID_KEY="session_id")
_stub("fastapi", FastAPI=object, Request=object)
_stub("fastapi.responses", PlainTextResponse=str)
_stub("cwsandbox_provider", register=lambda: None)
_stub("forge", Forge=types.SimpleNamespace(from_env=lambda: None), Cell=tuple)
sys.path.insert(0, str(ROOT / "gym_env" / "math_sandbox_eval"))
import app  # noqa: E402
import tir_app  # noqa: E402


class FakeSandbox:
    def __init__(self, name: str) -> None:
        self.name, self.commands, self.uploads, self.stopped = name, [], [], False

    async def upload(self, local: str, remote: str) -> None:
        self.uploads.append((Path(local).read_text(), remote))

    async def exec(self, command: str, timeout_s: float):
        self.commands.append(command)
        stdout = json.dumps([{"reward": 1.0, "extracted": "2"}])
        return types.SimpleNamespace(return_code=0, stdout=stdout, stderr="")

    async def stop(self) -> None:
        self.stopped = True


def _server(monkeypatch) -> tir_app.MathTirServer:
    graders = iter(FakeSandbox(f"grader-{i}") for i in range(100))

    async def new_grader(self):
        return next(graders)

    monkeypatch.setattr(app.MathSandboxEvalServer, "_new_sandbox", new_grader)
    config = tir_app.MathTirConfig(sandbox={"cwsandbox": {}}, grader_pool_size=2)
    return tir_app.MathTirServer(config=config)


def test_the_answer_is_graded_outside_the_sandbox_that_ran_the_model_code(monkeypatch):
    server = _server(monkeypatch)
    episode_sandbox = FakeSandbox("episode")
    graded = asyncio.run(server._grade_answer("2", "The answer is \\boxed{2}."))
    assert graded == {"reward": 1.0, "extracted": "2"}
    assert not episode_sandbox.commands and not episode_sandbox.uploads
    graders = [server._graders.get_nowait() for _ in range(2)]
    [grader] = [g for g in graders if g.commands]
    assert "verify_answer.py" in grader.commands[0]
    [(cases, _)] = grader.uploads  # the grader receives the answers as data, never code to run
    assert json.loads(cases) == [{"expected": "2", "response": "The answer is \\boxed{2}."}]


def test_a_sandbox_that_ran_model_code_is_stopped_not_reused(monkeypatch):
    server = _server(monkeypatch)
    sandbox = FakeSandbox("episode")
    server._free, server._busy, server._size = asyncio.Queue(), 1, 1
    asyncio.run(server._release(tir_app.Episode(sandbox)))
    assert sandbox.stopped and server._free.empty()
    assert (server._busy, server._size, server._stopped) == (0, 0, 1)  # the pool starts a fresh one
