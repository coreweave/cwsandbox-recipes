"""NeMo-Gym resources server for tool-integrated math reasoning: one CoreWeave sandbox per episode.

The model (Qwen2.5-Math, TIR format) writes ```python blocks; ``token_ids_model.py`` turns them
into ``execute_python`` calls, and Gym's ``simple_agent`` sends each call here. Each rollout holds
one sandbox from ``/seed_session`` until ``/verify``:

1. ``/seed_session`` takes a sandbox from the pool for this Gym session (a cookie per rollout).
   When the pool is at its maximum, the rollout waits here, before any generation.
2. ``/execute_python`` writes the code to ``/work/cell_<n>.py`` in the episode's sandbox and runs
   it with a time limit. Each cell runs on its own, as in Qwen2.5-Math's own TIR evaluation;
   files written in ``/work`` stay for the later cells of the episode.
3. ``/verify`` grades the final answer with math-verify in the same sandbox, then empties
   ``/work`` and returns the sandbox to the pool.

So the number of sandboxes in use follows the number of rollouts in flight. The pool, its
``pool_size``/``max_pool_size`` bounds and the ``@@metric`` rows
come from ``MathSandboxEvalServer`` (``app.py``); here ``busy`` counts sandboxes held by
episodes and ``waiting`` counts rollouts waiting for one. ``/calibrate`` works as in ``app.py``.
"""

import asyncio
import json
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Optional

import forge
import tir
from app import (
    REMOTE_VERIFIER,
    MathEvalVerifyRequest,
    MathEvalVerifyResponse,
    MathSandboxEvalConfig,
    MathSandboxEvalServer,
)
from cwsandbox.exceptions import SandboxTimeoutError, SandboxUnavailableError
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from nemo_gym.base_resources_server import BaseSeedSessionRequest, BaseSeedSessionResponse
from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.server_utils import SESSION_ID_KEY
from pydantic import BaseModel, PositiveInt

WORKDIR = "/work"


class MathTirConfig(MathSandboxEvalConfig):
    cpu: float = 1
    memory_mib: int = 2048
    cell_timeout_s: PositiveInt = 15
    # math-verify can hang on a pathological expression; one such answer must not hold a whole batch.
    grade_timeout_s: PositiveInt = 20
    max_output_chars: PositiveInt = 512
    episode_timeout_s: float = 1800  # a rollout that never reached /verify gives its sandbox back
    # The cwsandbox SDK blocks, so every sandbox call takes a thread while it runs. asyncio's
    # default of about 32 threads would queue hundreds of episodes behind a few calls.
    threads: PositiveInt = 1024


class ExecutePythonRequest(BaseModel):
    code: str


class MathTirVerifyResponse(MathEvalVerifyResponse):
    cells: int  # code cells the episode ran


@dataclass
class Episode:
    sandbox: AsyncSandbox
    cells: int = 0
    last_used: float = field(default_factory=time.monotonic)
    started: float = field(default_factory=time.time)
    timeline: list[forge.Cell] = field(default_factory=list)  # for the Agent Lens trace


class MathTirServer(MathSandboxEvalServer):
    config: MathTirConfig

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._episodes: dict[str, Episode] = {}
        self._examples = 0
        self._last_example = 0.0
        self._forge = forge.Forge.from_env()

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/execute_python")(self.execute_python)
        return app

    async def _new_sandbox(self) -> AsyncSandbox:
        sandbox = await super()._new_sandbox()
        await sandbox.exec(f"mkdir -p {WORKDIR}", timeout_s=60)
        return sandbox

    async def _pool(self) -> asyncio.Queue:
        """Start the first ``pool_size`` sandboxes concurrently and hand each out as soon as it is
        ready. A failed start only leaves the pool smaller; ``_maintain`` starts a replacement."""
        async with self._pool_lock:
            if self._free is None:
                asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(self.config.threads))
                self._free = asyncio.Queue()
                self._size = self.config.pool_size
                for _ in range(self.config.pool_size):
                    asyncio.create_task(self._add_free())
                print(f"[math_tir_sandbox] starting {self.config.pool_size} episode sandboxes", flush=True)
                asyncio.create_task(self._maintain())
                asyncio.create_task(self._reap())
                if self._forge:
                    asyncio.create_task(self._flush_evals())
                if self._dashboard:
                    asyncio.create_task(self._report())
            return self._free

    async def _hold(self) -> Episode:
        sandbox = await self._acquire()
        self._busy += 1
        return Episode(sandbox)

    async def _episode(self, session_id: str) -> Episode:
        episode = self._episodes.get(session_id)
        if episode is None:
            episode = self._episodes[session_id] = await self._hold()
        episode.last_used = time.monotonic()
        return episode

    async def _release(self, episode: Episode) -> None:
        """Empty the episode's work directory and return the sandbox to the pool. A sandbox that
        cannot be reset is stopped; ``_maintain`` starts another up to the minimum."""
        try:
            reset = await episode.sandbox.exec(f"rm -rf {WORKDIR} && mkdir -p {WORKDIR}", timeout_s=60)
            reusable = reset.return_code == 0
        except Exception as exc:  # noqa: BLE001 - the sandbox may be gone; any failure means do not reuse it
            print(f"[math_tir_sandbox] reset failed, stopping the sandbox: {exc}", flush=True)
            reusable = False
        self._busy -= 1
        if reusable:
            self._free.put_nowait((episode.sandbox, time.monotonic()))
            return
        self._size -= 1
        self._stopped += 1
        asyncio.create_task(episode.sandbox.stop())

    async def _reap(self) -> None:
        """Give back the sandboxes of rollouts that stopped before /verify (a failed model call,
        a cancelled run)."""
        while True:
            await asyncio.sleep(30)
            now = time.monotonic()
            for session_id, episode in list(self._episodes.items()):
                if now - episode.last_used > self.config.episode_timeout_s:
                    self._episodes.pop(session_id, None)
                    asyncio.create_task(self._release(episode))

    async def seed_session(self, request: Request, body: BaseSeedSessionRequest) -> BaseSeedSessionResponse:
        await self._episode(request.session[SESSION_ID_KEY])
        return BaseSeedSessionResponse()

    async def execute_python(self, request: Request, body: ExecutePythonRequest) -> PlainTextResponse:
        episode = await self._episode(request.session[SESSION_ID_KEY])
        episode.cells += 1
        self._cells += 1
        cell = f"{WORKDIR}/cell_{episode.cells}.py"
        began = time.time()
        with tempfile.NamedTemporaryFile("w", suffix=".py") as fh:
            fh.write(body.code)
            fh.flush()
            await episode.sandbox.upload(fh.name, cell)
        limit = self.config.cell_timeout_s
        result = await episode.sandbox.exec(f"cd {WORKDIR} && timeout {limit} python3 {cell}", timeout_s=limit + 30)
        episode.last_used = time.monotonic()
        output = tir.render_output(
            result.stdout, result.stderr, result.return_code, limit, self.config.max_output_chars
        )
        episode.timeline.append(forge.Cell(body.code, output, began, time.time()))
        return PlainTextResponse(output)

    async def _grade_in(self, sandbox: AsyncSandbox, expected: str, response: str) -> dict[str, Any]:
        remote = f"/tmp/cases-{time.monotonic_ns()}.json"
        with tempfile.NamedTemporaryFile("w", suffix=".json") as fh:
            json.dump([{"expected": expected, "response": response}], fh)
            fh.flush()
            await sandbox.upload(fh.name, remote)
        limit = self.config.grade_timeout_s
        try:
            result = await sandbox.exec(f"timeout {limit} python3 {REMOTE_VERIFIER} {remote}", timeout_s=limit + 30)
        except SandboxTimeoutError:
            result = None
        if result is None or result.return_code == tir.TIMEOUT_RETURN_CODE:
            print(f"[math_tir_sandbox] grading ran over {limit} s; reward 0", flush=True)
            self._graded += 1
            return {"reward": 0.0, "extracted": None}
        if result.return_code != 0:
            raise RuntimeError(f"verify_answer.py failed: {result.stderr}")
        [graded] = json.loads(result.stdout)
        self._graded += 1
        self._correct += graded["reward"] > 0
        return graded

    def _report_example(self, body: MathEvalVerifyRequest, output: list, graded: dict, episode: Episode) -> None:
        """An ``@@metric`` row for the dashboard's examples: the first eight episodes that ran code,
        then one every 10 s, which fills the page's list of 30 within a step or two. Any reward:
        the page shows wrong answers too."""
        due = self._examples < 8 or time.monotonic() - self._last_example >= 10
        if not (self._dashboard and due and episode.cells):
            return
        self._examples += 1
        self._last_example = time.monotonic()
        handle = getattr(episode.sandbox, "_handle", None)  # AsyncSandbox keeps the provider's id here
        row = tir.episode_example(
            body.responses_create_params.model_dump()["input"], output, body.expected_answer, graded["reward"],
            graded["extracted"], getattr(handle, "sandbox_id", None),
        )  # fmt: skip
        print("@@metric " + json.dumps({"ts": time.time(), **row}), flush=True)

    async def verify(self, request: Request, body: MathEvalVerifyRequest) -> MathTirVerifyResponse:
        episode = self._episodes.pop(request.session[SESSION_ID_KEY], None) or await self._hold()
        try:
            output = [item.model_dump() for item in body.response.output]
            answer = tir.episode_answer(output)
            graded = {"reward": 0.0, "extracted": None}
            if answer is not None:
                try:
                    graded = await self._grade_in(episode.sandbox, body.expected_answer, answer)
                except SandboxUnavailableError:
                    # A serverless sandbox's stream sometimes resets; one retry gets through.
                    await asyncio.sleep(1)
                    graded = await self._grade_in(episode.sandbox, body.expected_answer, answer)
            self._report_example(body, output, graded, episode)
            if self._forge:
                asyncio.create_task(self._log_to_forge(body, output, answer, graded, episode))
        finally:
            asyncio.create_task(self._release(episode))
        return MathTirVerifyResponse(
            **body.model_dump(), reward=graded["reward"], extracted_answer=graded["extracted"], cells=episode.cells
        )

    async def _log_to_forge(
        self, body: MathEvalVerifyRequest, output: list, answer: Optional[str], graded: dict, episode: Episode
    ) -> None:
        """Trace the rollout in Agent Lens (all evaluation rollouts, a sample of training ones) and
        add evaluation rollouts to the current Weave evaluation. Logging must never fail a rollout."""
        messages = body.responses_create_params.model_dump()["input"]
        system = next((m["content"] for m in messages if m.get("role") == "system"), "")
        try:
            if self._forge.wants(body.split):
                self._forge.trace(
                    session_id=uuid.uuid4().hex[:12], split=body.split, question=body.question, system=system,
                    output=output, cells=episode.timeline, started=episode.started, ended=time.time(),
                    answer=answer or "", expected=body.expected_answer, extracted=graded["extracted"],
                    reward=graded["reward"],
                )  # fmt: skip
            if body.split == "eval":
                row = {"question": body.question, "expected": body.expected_answer, "answer": answer or "",
                       "extracted": graded["extracted"], "cells": episode.cells, "reward": graded["reward"]}  # fmt: skip
                await asyncio.to_thread(self._forge.add_eval, row)
        except Exception as exc:  # noqa: BLE001 - observability only
            print(f"[forge] logging failed: {exc}", flush=True)

    async def _flush_evals(self) -> None:
        """Log a validation round that stopped short of its expected size."""
        while True:
            await asyncio.sleep(10)
            try:
                await asyncio.to_thread(self._forge.flush_idle)
            except Exception as exc:  # noqa: BLE001 - observability only
                print(f"[forge] logging failed: {exc}", flush=True)


if __name__ == "__main__":
    MathTirServer.run_webserver()
