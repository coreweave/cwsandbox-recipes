"""NeMo-Gym resources server that grades math answers inside CoreWeave sandboxes.

Used for evaluation (AIME-2024 validation) and verifier calibration. It keeps a small
pool of CPU sandboxes, created through Gym's ``AsyncSandbox`` API with the ``cwsandbox``
provider. Each grading request uploads the cases to one sandbox and runs
``verify_answer.py`` (math-verify) there.

Routes:

* ``POST /verify``: Gym's standard verify call; grades one rollout.
* ``POST /calibrate``: grades a list of ``{"expected", "response"}`` cases and returns
  one ``{"reward", "extracted"}`` per case. No model is involved.
"""

import asyncio
import json
import tempfile
import uuid
from pathlib import Path
from typing import Any, Optional

from cwsandbox_provider import register
from fastapi import FastAPI
from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.sandbox import AsyncSandbox, SandboxSpec
from pydantic import BaseModel, PositiveInt

VERIFIER = Path(__file__).with_name("verify_answer.py")
REMOTE_VERIFIER = "/opt/verify_answer.py"


class MathSandboxEvalConfig(BaseResourcesServerConfig):
    sandbox: dict[str, Any]  # Gym provider config, e.g. {"cwsandbox": {"runner_id": ...}}
    pool_size: PositiveInt = 4
    image: str = "python:3.12-slim"
    setup_command: str = "pip install --quiet --no-cache-dir math-verify==0.8.0"
    cpu: float = 2
    memory_mib: int = 4096
    exec_timeout_s: float = 300


class MathEvalRunRequest(BaseRunRequest):
    question: str
    expected_answer: str


class MathEvalVerifyRequest(MathEvalRunRequest, BaseVerifyRequest):
    pass


class MathEvalVerifyResponse(BaseVerifyResponse):
    expected_answer: str
    extracted_answer: Optional[str]


class CalibrationCase(BaseModel):
    expected: str
    response: str


class CalibrateRequest(BaseModel):
    cases: list[CalibrationCase]


class MathSandboxEvalServer(SimpleResourcesServer):
    config: MathSandboxEvalConfig

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        register()
        self._pool: asyncio.Queue | None = None
        self._pool_lock = asyncio.Lock()

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/calibrate")(self.calibrate)
        return app

    async def _new_sandbox(self) -> AsyncSandbox:
        spec = SandboxSpec(
            image=self.config.image,
            resources={"cpu": self.config.cpu, "memory_mib": self.config.memory_mib},
        )
        sandbox = await AsyncSandbox(self.config.sandbox, spec).start()
        setup = await sandbox.exec(self.config.setup_command, timeout_s=self.config.exec_timeout_s)
        if setup.return_code != 0:
            await sandbox.stop()
            raise RuntimeError(f"sandbox setup failed: {setup.stderr}")
        await sandbox.upload(VERIFIER, REMOTE_VERIFIER)
        return sandbox

    async def _sandboxes(self) -> asyncio.Queue:
        """Create the sandbox pool on first use."""
        async with self._pool_lock:
            if self._pool is None:
                sandboxes = await asyncio.gather(*(self._new_sandbox() for _ in range(self.config.pool_size)))
                self._pool = asyncio.Queue()
                for sandbox in sandboxes:
                    self._pool.put_nowait(sandbox)
                print(f"[math_sandbox_eval] {len(sandboxes)} grading sandboxes ready", flush=True)
            return self._pool

    async def _grade(self, cases: list[dict[str, str]]) -> list[dict[str, Any]]:
        pool = await self._sandboxes()
        sandbox = await pool.get()
        try:
            remote = f"/tmp/cases-{uuid.uuid4().hex}.json"
            with tempfile.NamedTemporaryFile("w", suffix=".json") as fh:
                json.dump(cases, fh)
                fh.flush()
                await sandbox.upload(fh.name, remote)
            result = await sandbox.exec(f"python3 {REMOTE_VERIFIER} {remote}", timeout_s=self.config.exec_timeout_s)
            if result.return_code != 0:
                raise RuntimeError(f"verify_answer.py failed: {result.stderr}")
            return json.loads(result.stdout)
        finally:
            pool.put_nowait(sandbox)

    async def verify(self, body: MathEvalVerifyRequest) -> MathEvalVerifyResponse:
        text = "".join(
            part.text
            for item in body.response.output
            if item.type == "message"
            for part in item.content
            if part.type == "output_text"
        )
        [graded] = await self._grade([{"expected": body.expected_answer, "response": text}])
        return MathEvalVerifyResponse(
            **body.model_dump(), reward=graded["reward"], extracted_answer=graded["extracted"]
        )

    async def calibrate(self, body: CalibrateRequest) -> dict[str, Any]:
        cases = [case.model_dump() for case in body.cases]
        chunks = [cases[i : i + 50] for i in range(0, len(cases), 50)]
        graded = await asyncio.gather(*(self._grade(chunk) for chunk in chunks))
        return {"results": [result for chunk in graded for result in chunk]}


if __name__ == "__main__":
    MathSandboxEvalServer.run_webserver()
