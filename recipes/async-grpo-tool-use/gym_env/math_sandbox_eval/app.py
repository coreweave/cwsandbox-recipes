"""NeMo-Gym resources server that grades math answers inside CoreWeave sandboxes.

The base of the episode server (``tir_app.py``), and the verifier calibration route. It keeps a pool of CPU sandboxes, created
through Gym's ``AsyncSandbox`` API with the ``cwsandbox`` provider. Each grading request
uploads the cases to one sandbox and runs ``verify_answer.py`` (math-verify) there.

The pool starts at ``pool_size`` sandboxes. With ``max_pool_size`` above it, the pool adds a
sandbox whenever a request finds none free, and stops extra sandboxes that sat idle for
``idle_timeout_s``, so its size follows the rollout bursts. Setting ``pool_size`` equal to
``max_pool_size`` keeps every sandbox warm between bursts.

Routes:

* ``POST /verify``: Gym's standard verify call; grades one rollout.
* ``POST /calibrate``: grades a list of ``{"expected", "response"}`` cases and returns
  one ``{"reward", "extracted"}`` per case. No model is involved.
"""

import asyncio
import json
import os
import tempfile
import time
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
    max_pool_size: Optional[PositiveInt] = None  # default: pool_size (a fixed pool)
    idle_timeout_s: float = 20
    image: str = "python:3.12-slim"
    setup_command: str = "pip install --quiet --no-cache-dir math-verify==0.8.0"
    cpu: float = 2
    memory_mib: int = 4096
    exec_timeout_s: float = 300


class MathEvalRunRequest(BaseRunRequest):
    question: str
    expected_answer: str
    split: str = "train"  # "eval" for AIME-2024 validation rows (dapo_data.py)


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
        self._free: asyncio.Queue | None = None  # (sandbox, time it was last released)
        self._pool_lock = asyncio.Lock()
        self._size = 0  # sandboxes started or starting
        self._busy = 0
        self._waiting = 0
        self._graded = 0
        self._correct = 0
        self._started = 0
        self._stopped = 0
        self._start_s = 0.0  # summed time to start and set up a sandbox
        self._acquires = 0
        self._cells = 0  # code cells run by tool-using episodes (tir_app.py)
        self._acquire_s = 0.0  # summed time requests spent getting a sandbox
        self._min = self.config.pool_size
        self._max = max(self.config.max_pool_size or self.config.pool_size, self.config.pool_size)
        self._dashboard = os.environ.get("DASHBOARD_METRICS") == "1"

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/calibrate")(self.calibrate)
        return app

    async def _new_sandbox(self) -> AsyncSandbox:
        spec = SandboxSpec(
            image=self.config.image,
            resources={"cpu": self.config.cpu, "memory_mib": self.config.memory_mib},
        )
        began = time.monotonic()
        sandbox = await AsyncSandbox(self.config.sandbox, spec).start()
        setup = await sandbox.exec(self.config.setup_command, timeout_s=self.config.exec_timeout_s)
        if setup.return_code != 0:
            await sandbox.stop()
            raise RuntimeError(f"sandbox setup failed: {setup.stderr}")
        await sandbox.upload(VERIFIER, REMOTE_VERIFIER)
        self._started += 1
        self._start_s += time.monotonic() - began
        return sandbox

    async def _pool(self) -> asyncio.Queue:
        """Create the first ``pool_size`` sandboxes on first use."""
        async with self._pool_lock:
            if self._free is None:
                self._size = self.config.pool_size
                sandboxes = await asyncio.gather(*(self._new_sandbox() for _ in range(self.config.pool_size)))
                self._free = asyncio.Queue()
                for sandbox in sandboxes:
                    self._free.put_nowait((sandbox, time.monotonic()))
                print(f"[math_sandbox_eval] {len(sandboxes)} grading sandboxes ready", flush=True)
                asyncio.create_task(self._maintain())
                if self._dashboard:
                    asyncio.create_task(self._report())
            return self._free

    async def _acquire(self) -> AsyncSandbox:
        began = time.monotonic()
        try:
            return await self._take()
        finally:
            self._acquires += 1
            self._acquire_s += time.monotonic() - began

    async def _take(self) -> AsyncSandbox:
        free = await self._pool()
        try:
            return free.get_nowait()[0]
        except asyncio.QueueEmpty:
            pass
        if self._size < self._max:
            self._size += 1
            try:
                return await self._new_sandbox()
            except Exception:
                self._size -= 1
                raise
        self._waiting += 1
        try:
            return (await free.get())[0]
        finally:
            self._waiting -= 1

    async def _add_free(self) -> None:
        try:
            sandbox = await self._new_sandbox()
        except Exception as exc:  # noqa: BLE001 - a failed warm start only leaves the pool smaller
            self._size -= 1
            print(f"[math_sandbox_eval] warm start failed: {exc}", flush=True)
            return
        self._free.put_nowait((sandbox, time.monotonic()))

    async def _maintain(self) -> None:
        """Every 2 seconds: stop sandboxes above the maximum, stop extra sandboxes idle for
        ``idle_timeout_s``, and start sandboxes up to the minimum."""
        while True:
            await asyncio.sleep(2)
            keep = []
            while not self._free.empty():
                sandbox, released = self._free.get_nowait()
                idle = time.monotonic() - released > self.config.idle_timeout_s
                if self._size > self._max or (idle and self._size > self._min):
                    self._size -= 1
                    self._stopped += 1
                    asyncio.create_task(sandbox.stop())
                else:
                    keep.append((sandbox, released))
            for item in keep:
                self._free.put_nowait(item)
            while self._size < self._min:
                self._size += 1
                asyncio.create_task(self._add_free())

    async def _report(self) -> None:
        """One ``@@metric`` line per second for the live dashboard (``DASHBOARD_METRICS=1``)."""
        last = None
        while True:
            await asyncio.sleep(1)
            row = {
                "k": "env", "sandboxes": self._size, "busy": self._busy, "waiting": self._waiting,
                "graded": self._graded, "correct": self._correct, "started": self._started, "stopped": self._stopped,
                "min": self._min, "max": self._max, "start_s": round(self._start_s, 1),
                "acquires": self._acquires, "acquire_s": round(self._acquire_s, 1), "cells": self._cells,
            }  # fmt: skip
            if row != last:
                print("@@metric " + json.dumps({"ts": time.time(), **row}), flush=True)
                last = row

    async def _grade(self, cases: list[dict[str, str]]) -> list[dict[str, Any]]:
        sandbox = await self._acquire()
        self._busy += 1
        try:
            remote = f"/tmp/cases-{uuid.uuid4().hex}.json"
            with tempfile.NamedTemporaryFile("w", suffix=".json") as fh:
                json.dump(cases, fh)
                fh.flush()
                await sandbox.upload(fh.name, remote)
            result = await sandbox.exec(f"python3 {REMOTE_VERIFIER} {remote}", timeout_s=self.config.exec_timeout_s)
            if result.return_code != 0:
                raise RuntimeError(f"verify_answer.py failed: {result.stderr}")
            graded = json.loads(result.stdout)
            self._graded += len(graded)
            self._correct += sum(1 for row in graded if row["reward"] > 0)
            return graded
        finally:
            self._busy -= 1
            self._free.put_nowait((sandbox, time.monotonic()))

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
