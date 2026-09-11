"""Async client for the in-sandbox τ-bench environment server.

The HTTP transport is behind a :class:`Transport` protocol so the pool, the tool
and the tests can all run without a network or a real sandbox. ``HttpxTransport``
is the production implementation; tests inject an in-process fake that calls
``env_server._dispatch`` directly, which keeps the server's routing logic under
test rather than mocked away.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, runtime_checkable


class EnvServerError(RuntimeError):
    """Non-2xx response from the environment server."""

    def __init__(self, status: int, message: str, path: str = ""):
        super().__init__(f"env server {status} on {path or '<unknown>'}: {message}")
        self.status = status
        self.message = message
        self.path = path


@runtime_checkable
class Transport(Protocol):
    """Minimal request surface the env client needs."""

    async def request(
        self,
        method: str,
        url: str,
        *,
        json_body: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> tuple[int, Dict[str, Any]]:
        """Return ``(status_code, decoded_json_body)``."""
        ...

    async def aclose(self) -> None: ...


class HttpxTransport:
    """Production transport backed by a shared ``httpx.AsyncClient``.

    ``httpx`` is imported lazily so that importing this module (and therefore the
    tool and pool) works on a machine with only the base dependencies installed.
    """

    def __init__(self, *, timeout: float = 30.0, max_connections: int = 512):
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise ImportError(
                "HttpxTransport requires httpx. Install the sandbox extra: "
                'uv pip install -e ".[sandbox]"'
            ) from exc
        self._httpx = httpx
        self._max_connections = max_connections
        self._client = httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(max_connections=max_connections),
        )

    @property
    def max_connections(self) -> int:
        return self._max_connections

    async def request(
        self,
        method: str,
        url: str,
        *,
        json_body: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> tuple[int, Dict[str, Any]]:
        response = await self._client.request(
            method, url, json=json_body, headers=headers, timeout=timeout
        )
        try:
            body = response.json()
        except Exception:
            body = {"error": response.text}
        if not isinstance(body, dict):
            body = {"result": body}
        return response.status_code, body

    async def aclose(self) -> None:
        await self._client.aclose()


class SandboxEnvClient:
    """Typed wrapper over the env server's four endpoints."""

    def __init__(
        self,
        base_url: str,
        transport: Transport,
        *,
        token: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.transport = transport
        self.token = token
        self.timeout = timeout

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    async def _call(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
        *,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        status, body = await self.transport.request(
            method,
            f"{self.base_url}{path}",
            json_body=payload,
            headers=self._headers(),
            timeout=timeout if timeout is not None else self.timeout,
        )
        if status >= 400:
            raise EnvServerError(status, str(body.get("error", body)), path)
        return body

    async def health(self, *, timeout: Optional[float] = None) -> Dict[str, Any]:
        return await self._call("GET", "/health", timeout=timeout)

    async def spec(self, *, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Static domain description (wiki/rules/tool schemas) for prompt building."""
        return await self._call("GET", "/spec", timeout=timeout)

    async def reset(
        self, task_index: int, *, episode_id: Optional[str] = None
    ) -> Dict[str, Any]:
        return await self._call(
            "POST", "/reset", {"task_index": int(task_index), "episode_id": episode_id}
        )

    async def step(self, action_name: str, action_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        return await self._call(
            "POST", "/step", {"action_name": action_name, "action_kwargs": action_kwargs}
        )

    async def reward(self) -> Dict[str, Any]:
        return await self._call("POST", "/reward", {})
