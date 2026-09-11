"""τ-bench environment server. Runs **inside** a CoreWeave sandbox.

This module is uploaded verbatim into the sandbox (``Sandbox.write_file``) and
started with ``Sandbox.exec``. It therefore depends on **stdlib + tau_bench only**
and must never import ``verl_taubench``.

Design
------
The sandbox hosts the one component of τ-bench with real per-episode state: the
retail/airline JSON database plus the tool API that mutates it. Reward is computed
here by diffing that database against the goal state.

Two things deliberately stay outside the sandbox:

* **The user simulator** is an LLM endpoint, not environment code, so it runs on
  the inference fleet. This server never invokes ``env.user``.
* **The policy** runs on the vLLM/SGLang fleet.

Because the simulator is remote, ``respond`` actions are *recorded* here but not
executed. That recording is mandatory, not cosmetic: τ-bench's
``Env.calculate_reward`` scans ``env.actions`` for ``respond`` actions to verify
that every string in ``task.outputs`` was actually said to the user. Dropping them
would silently zero the reward on every task that has required outputs.

Episode lifecycle
-----------------
``POST /reset`` constructs a brand-new ``Env`` for ``task_index``. Construction
alone is a full reset: ``Env.__init__`` reloads the database and clears
``actions``. We never call ``env.reset()``, which would block on the in-sandbox
user simulator.

Endpoints
---------
=========================  ==================================================
``GET  /health``           liveness + which episode is loaded
``POST /reset``            ``{"task_index": int}`` -> task metadata
``POST /step``             ``{"action_name": str, "action_kwargs": dict}``
``POST /reward``           terminal reward (destructive; cached)
=========================  ==================================================

Auth
----
``ingress_mode="public"`` publishes this server on the internet, so every request
must carry ``Authorization: Bearer <token>`` matching ``--token`` /
``TAUBENCH_ENV_TOKEN``. Running without a token is refused unless
``--insecure`` is passed explicitly.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple

from tau_bench.envs.airline.env import MockAirlineDomainEnv
from tau_bench.envs.retail.env import MockRetailDomainEnv
from tau_bench.types import RESPOND_ACTION_NAME, Action

_DOMAIN_ENV_CLS = {
    "retail": MockRetailDomainEnv,
    "airline": MockAirlineDomainEnv,
}

# Max JSON body we will read from a single request (1 MiB). Tool arguments are
# small; anything larger is a bug or an attack.
_MAX_BODY_BYTES = 1024 * 1024


class EpisodeError(Exception):
    """Raised for client errors (bad task index, no episode loaded, ...)."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


class EpisodeState:
    """Holds the single in-flight τ-bench episode for this sandbox.

    One sandbox hosts one episode at a time. All mutation is guarded by a lock so
    a stray concurrent request cannot interleave two episodes' actions.
    """

    def __init__(self, domain: str, task_split: str, partial_credit: float = 0.0):
        if domain not in _DOMAIN_ENV_CLS:
            raise ValueError(f"unknown domain {domain!r}; choose from {sorted(_DOMAIN_ENV_CLS)}")
        self.domain = domain
        self.task_split = task_split
        # Weight for graded credit on failed episodes; 0 keeps upstream
        # tau-bench's strict binary reward.
        self.partial_credit = max(0.0, min(1.0, float(partial_credit)))
        self.lock = threading.Lock()
        self.env: Any = None
        self.task_index: Optional[int] = None
        self.done: bool = False
        self.episode_id: Optional[str] = None
        self._terminal_reward: Optional[Dict[str, Any]] = None

    # -- lifecycle ---------------------------------------------------------

    def reset(self, task_index: int, episode_id: Optional[str] = None) -> Dict[str, Any]:
        """Load ``task_index`` into a fresh environment.

        Constructing the Env *is* the reset: ``Env.__init__`` reloads the database
        from disk and sets ``actions = []``. We always pass ``task_index``
        explicitly, which also sidesteps τ-bench's inclusive
        ``random.randint(0, len(tasks))`` off-by-one.
        """
        env_cls = _DOMAIN_ENV_CLS[self.domain]
        # ``user_strategy="human"`` guarantees no LLM client is constructed. The
        # human simulator only reads stdin, and we never call into it.
        env = env_cls(
            user_strategy="human",
            user_model="",
            user_provider=None,
            task_split=self.task_split,
            task_index=0,
        )
        if not (0 <= task_index < len(env.tasks)):
            raise EpisodeError(
                f"task_index={task_index} out of range for {self.domain}/{self.task_split} "
                f"(tasks={len(env.tasks)})"
            )
        env.task_index = task_index
        env.task = env.tasks[task_index]
        env.actions = []

        self.env = env
        self.task_index = task_index
        self.done = False
        self.episode_id = episode_id
        self._terminal_reward = None

        return {
            "task_index": task_index,
            "domain": self.domain,
            "task_split": self.task_split,
            "episode_id": episode_id,
            "instruction": env.task.instruction,
            "outputs": list(env.task.outputs),
            "num_tasks": len(env.tasks),
            # NOTE: deliberately no `data_hash` here. `get_data_hash()` hashes the
            # whole domain database (~28 ms) and sits on the acquire critical
            # path, for a value no caller uses.
        }

    def _require_env(self) -> Any:
        if self.env is None:
            raise EpisodeError("no episode loaded; POST /reset first", status=409)
        return self.env

    # -- stepping ----------------------------------------------------------

    def step(self, action_name: str, action_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        env = self._require_env()
        if self.done:
            # calculate_reward has already run and reloaded the database, so any
            # further step would score against a replayed ground-truth state.
            raise EpisodeError("episode is finished; POST /reset to start another", status=409)
        action = Action(name=action_name, kwargs=action_kwargs)

        if action_name == RESPOND_ACTION_NAME:
            # Record but do NOT execute: the user simulator is remote. This entry
            # is what `calculate_reward` scans for `task.outputs` matching.
            env.actions.append(action)
            return {
                "observation": "",
                "reward": 0.0,
                "done": False,
                "source": RESPOND_ACTION_NAME,
                "recorded_only": True,
            }

        response = env.step(action)
        done = bool(response.done)
        reward = float(response.reward)
        if done:
            # A terminate tool fired. `Env.step` already ran `calculate_reward`,
            # which is destructive, so cache the result and refuse further steps.
            self.done = True
            reward_info = getattr(response.info, "reward_info", None)
            self._terminal_reward = {
                "reward": reward,
                "source": "terminate_tool",
                # Same shape as the /reward path: the RewardResult's `info`
                # field, not the whole RewardResult (which also carries the
                # entire ground-truth action list).
                "info": _serialize(getattr(reward_info, "info", reward_info)),
            }
        return {
            "observation": response.observation,
            "reward": reward,
            "done": done,
            "source": getattr(response.info, "source", None) or action_name,
            "recorded_only": False,
        }

    # -- reward ------------------------------------------------------------

    # Credit for calling the right tool at all; the rest is earned by getting
    # its arguments right. Exact-match-only scoring is nearly as sparse as the
    # binary reward, which is what left GRPO with no in-group variance.
    _TOOL_NAME_CREDIT = 0.5

    @staticmethod
    def _kwargs_similarity(gt_kwargs: Dict[str, Any], agent_kwargs: Dict[str, Any]) -> float:
        """Fraction of the ground-truth arguments the agent got exactly right."""
        if not gt_kwargs:
            return 1.0
        matched = sum(
            1 for key, value in gt_kwargs.items() if key in agent_kwargs and agent_kwargs[key] == value
        )
        return matched / len(gt_kwargs)

    @classmethod
    def _action_progress(cls, agent_actions: list, gt_actions: list) -> float:
        """Graded progress over the ground-truth tool actions, in [0, 1].

        Each ground-truth action scores the best match among the agent's
        unclaimed actions: 0 when the tool was never called, ``_TOOL_NAME_CREDIT``
        for the right tool with wrong arguments, rising to 1.0 as the arguments
        become correct. Each agent action can satisfy at most one ground-truth
        action, so repeating a call cannot inflate the score. Respond actions
        are excluded here; they are graded by ``task.outputs``.
        """
        goal = [a for a in gt_actions if a.name != RESPOND_ACTION_NAME]
        if not goal:
            return 1.0
        candidates = [a for a in agent_actions if a.name != RESPOND_ACTION_NAME]
        claimed: set = set()
        total = 0.0
        for target in goal:
            best_score, best_index = 0.0, None
            for index, candidate in enumerate(candidates):
                if index in claimed or candidate.name != target.name:
                    continue
                score = cls._TOOL_NAME_CREDIT + (1.0 - cls._TOOL_NAME_CREDIT) * cls._kwargs_similarity(
                    dict(target.kwargs), dict(candidate.kwargs)
                )
                if score > best_score:
                    best_score, best_index = score, index
            if best_index is not None:
                claimed.add(best_index)
            total += best_score
        return total / len(goal)

    @staticmethod
    def _output_progress(agent_actions: list, outputs: list) -> Optional[float]:
        """Fraction of required outputs the agent actually told the user.

        Mirrors tau-bench's own substring check in ``calculate_reward``; returns
        None for tasks that require no specific outputs.
        """
        if not outputs:
            return None
        said = " ".join(
            str(a.kwargs.get("content", ""))
            for a in agent_actions
            if a.name == RESPOND_ACTION_NAME
        ).lower().replace(",", "")
        found = sum(1 for output in outputs if str(output).lower() in said)
        return found / len(outputs)

    def reward(self) -> Dict[str, Any]:
        """Terminal reward: DB diff against the goal state, plus output checks.

        ``Env.calculate_reward`` is destructive -- it reloads the database and
        replays the ground-truth actions to derive ``gt_data_hash``. It must run
        at most once per episode, so the result is cached and the episode is
        marked spent.

        With ``partial_credit > 0``, a failed episode earns
        ``partial_credit * progress`` instead of a hard 0, where progress
        averages how far the agent got on the two things tau-bench grades: the
        ground-truth tool actions and the required outputs to the user. GRPO
        needs reward variance INSIDE each group of rollouts; the strict binary
        reward makes whole groups uniform (all-0 or all-1), which zeroes every
        group-relative advantage and freezes the policy. The strict success
        signal is preserved in ``info.success``, and validation pools never use
        partial credit, so reported task success stays comparable.
        """
        if self._terminal_reward is not None:
            return self._terminal_reward

        env = self._require_env()
        # calculate_reward replays the ground-truth actions through the env,
        # appending them to env.actions: snapshot the agent's history first.
        agent_actions = list(env.actions)
        result = env.calculate_reward()
        success = float(result.reward)
        reward = success
        info = _serialize(getattr(result, "info", None)) or {}
        source = "calculate_reward"
        if self.partial_credit > 0:
            action_progress = self._action_progress(agent_actions, env.task.actions)
            output_progress = self._output_progress(
                agent_actions, list(getattr(env.task, "outputs", []) or [])
            )
            components = [action_progress]
            if output_progress is not None:
                components.append(output_progress)
            progress = sum(components) / len(components)
            if success < 1.0:
                reward = self.partial_credit * progress
            info["success"] = success
            info["action_progress"] = action_progress
            if output_progress is not None:
                info["output_progress"] = output_progress
            info["progress"] = progress
            source = "calculate_reward+partial_credit"
        payload = {
            "reward": reward,
            "source": source,
            "info": info,
        }
        self._terminal_reward = payload
        self.done = True
        return payload

    def spec(self) -> Dict[str, Any]:
        """Static domain description: wiki, rules and tool schemas.

        The policy's system prompt is built from this. It is constant for a
        domain, so the client fetches it once per sandbox and caches it rather
        than paying ~17 KB on every reset.
        """
        env = self.env
        if env is None:
            # Cheap throwaway env just to read the static domain description.
            env = _DOMAIN_ENV_CLS[self.domain](
                user_strategy="human",
                user_model="",
                user_provider=None,
                task_split=self.task_split,
                task_index=0,
            )
        return {
            "domain": self.domain,
            "wiki": env.wiki,
            "rules": list(env.rules),
            "tools_info": list(env.tools_info),
        }

    def health(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "domain": self.domain,
            "task_split": self.task_split,
            "task_index": self.task_index,
            "episode_id": self.episode_id,
            "episode_loaded": self.env is not None,
            "done": self.done,
        }


def _serialize(obj: Any) -> Any:
    """Best-effort JSON-safe conversion of a pydantic reward-info object."""
    if obj is None:
        return None
    for attr in ("model_dump", "dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                pass
    if isinstance(obj, (str, int, float, bool, list, dict)):
        return obj
    return str(obj)


def _dispatch(state: EpisodeState, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Route one request. Pure function of (state, path, payload) for testability."""
    if path == "/health":
        return state.health()

    if path == "/spec":
        with state.lock:
            return state.spec()

    if path == "/reset":
        if "task_index" not in payload:
            raise EpisodeError("reset requires 'task_index'")
        try:
            task_index = int(payload["task_index"])
        except (TypeError, ValueError):
            raise EpisodeError(f"task_index must be an int, got {payload['task_index']!r}")
        with state.lock:
            return state.reset(task_index, episode_id=payload.get("episode_id"))

    if path == "/step":
        action_name = payload.get("action_name")
        if not action_name:
            raise EpisodeError("step requires 'action_name'")
        action_kwargs = payload.get("action_kwargs") or {}
        if not isinstance(action_kwargs, dict):
            raise EpisodeError("'action_kwargs' must be an object")
        with state.lock:
            return state.step(str(action_name), action_kwargs)

    if path == "/reward":
        with state.lock:
            return state.reward()

    raise EpisodeError(f"unknown path {path!r}", status=404)


def make_handler(state: EpisodeState, token: Optional[str]) -> type:
    """Build a request handler bound to ``state``."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "taubench-env/1.0"

        def _send(self, status: int, body: Dict[str, Any]) -> None:
            raw = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _authorized(self) -> bool:
            if token is None:
                return True
            supplied = self.headers.get("Authorization", "")
            expected = f"Bearer {token}"
            # Length-independent comparison is not required here (the token is
            # not a password hash), but constant-time keeps it tidy.
            if len(supplied) != len(expected):
                return False
            mismatch = 0
            for a, b in zip(supplied, expected):
                mismatch |= ord(a) ^ ord(b)
            return mismatch == 0

        def _handle(self, payload: Dict[str, Any]) -> None:
            if not self._authorized():
                self._send(401, {"error": "unauthorized"})
                return
            path = self.path.split("?", 1)[0].rstrip("/") or "/health"
            try:
                self._send(200, _dispatch(state, path, payload))
            except EpisodeError as exc:
                self._send(exc.status, {"error": str(exc)})
            except Exception as exc:  # pragma: no cover - defensive
                self._send(
                    500,
                    {
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(limit=5),
                    },
                )

        def do_GET(self) -> None:  # noqa: N802 - stdlib naming
            self._handle({})

        def do_POST(self) -> None:  # noqa: N802 - stdlib naming
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                self._send(400, {"error": "invalid Content-Length"})
                return
            if length > _MAX_BODY_BYTES:
                # Drain the body before replying, otherwise the unread bytes
                # desync the next request on this keep-alive connection.
                remaining = length
                while remaining > 0:
                    chunk = self.rfile.read(min(remaining, 65536))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                self._send(413, {"error": "request body too large"})
                return
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw or b"{}")
            except json.JSONDecodeError as exc:
                self._send(400, {"error": f"invalid JSON: {exc}"})
                return
            if not isinstance(payload, dict):
                self._send(400, {"error": "request body must be a JSON object"})
                return
            self._handle(payload)

        def log_message(self, fmt: str, *args: Any) -> None:
            # Keep sandbox stdout clean; the pool surfaces failures via HTTP status.
            return

    return Handler


def build_server(
    domain: str,
    task_split: str,
    host: str,
    port: int,
    token: Optional[str],
    partial_credit: float = 0.0,
) -> Tuple[ThreadingHTTPServer, EpisodeState]:
    state = EpisodeState(domain=domain, task_split=task_split, partial_credit=partial_credit)
    server = ThreadingHTTPServer((host, port), make_handler(state, token))
    server.daemon_threads = True
    return server, state


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="τ-bench environment server (in-sandbox)")
    parser.add_argument("--domain", default=os.environ.get("TAUBENCH_DOMAIN", "retail"))
    parser.add_argument("--task-split", default=os.environ.get("TAUBENCH_TASK_SPLIT", "test"))
    # Bind 0.0.0.0: CoreWeave ingress forwards to the container IP, so a
    # localhost-bound server is unreachable from outside the sandbox.
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("TAUBENCH_ENV_PORT", "8080")))
    parser.add_argument("--token", default=os.environ.get("TAUBENCH_ENV_TOKEN"))
    parser.add_argument(
        "--partial-credit",
        type=float,
        default=float(os.environ.get("TAUBENCH_PARTIAL_CREDIT", "0") or 0),
        help="Weight for graded credit on failed episodes (0 = strict binary reward).",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Allow running with no bearer token. Never use with ingress_mode='public'.",
    )
    args = parser.parse_args(argv)

    if not args.token and not args.insecure:
        parser.error(
            "refusing to start without --token/TAUBENCH_ENV_TOKEN. "
            "Public ingress exposes this server to the internet; pass --insecure to override."
        )

    server, _ = build_server(
        domain=args.domain,
        task_split=args.task_split,
        host=args.host,
        port=args.port,
        token=args.token or None,
        partial_credit=args.partial_credit,
    )
    print(f"taubench env server listening on {args.host}:{args.port} domain={args.domain}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
