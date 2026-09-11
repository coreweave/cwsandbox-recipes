"""Shared test doubles for the smoke suite."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from verl_taubench.sandbox.simulator import STOP_TOKEN


class ScriptedSimulator:
    """Deterministic stand-in for the remote user simulator.

    Replays ``replies`` in order and emits ``STOP_TOKEN`` once exhausted, so a
    test can end an episode on a chosen turn without an LLM endpoint.
    """

    def __init__(self, replies: Optional[List[str]] = None, opening: str = "Hello, I need help."):
        self.replies = list(replies or [])
        self.opening = opening
        self.calls: List[Dict[str, Any]] = []
        self._cursor: Dict[str, int] = {}

    async def start(self, episode_id: str, instruction: str) -> str:
        self._cursor[episode_id] = 0
        self.calls.append({"episode_id": episode_id, "event": "start", "instruction": instruction})
        return self.opening

    async def respond(self, episode_id: str, agent_message: str) -> str:
        idx = self._cursor.get(episode_id, 0)
        self._cursor[episode_id] = idx + 1
        self.calls.append({"episode_id": episode_id, "event": "respond", "message": agent_message})
        if idx < len(self.replies):
            return self.replies[idx]
        return STOP_TOKEN

    async def finish(self, episode_id: str) -> None:
        self._cursor.pop(episode_id, None)
        self.calls.append({"episode_id": episode_id, "event": "finish"})

