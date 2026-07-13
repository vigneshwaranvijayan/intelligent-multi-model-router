from __future__ import annotations

from typing import Any, Protocol

import httpx

from multirouter.domain import ModelSpec
from multirouter.schemas import ChatCompletionRequest


class BackendError(RuntimeError):
    pass


class BackendClientProtocol(Protocol):
    async def complete(
        self, model: ModelSpec, request: ChatCompletionRequest
    ) -> dict[str, Any]: ...

    async def health(self, model: ModelSpec) -> bool: ...

    async def close(self) -> None: ...


class HTTPBackendClient:
    def __init__(self, timeout_seconds: float = 15.0) -> None:
        timeout = httpx.Timeout(timeout_seconds, connect=min(5.0, timeout_seconds))
        self._client = httpx.AsyncClient(timeout=timeout)

    async def complete(self, model: ModelSpec, request: ChatCompletionRequest) -> dict[str, Any]:
        payload = request.model_dump(exclude={
            "routing_policy",
            "maximum_latency_ms",
            "privacy_level",
            "required_capabilities",
            "include_routing_trace",
        })
        payload["model"] = model.id
        payload["stream"] = False
        try:
            response = await self._client.post(
                f"{model.endpoint}/v1/chat/completions", json=payload
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise BackendError(f"{model.id} request failed: {exc}") from exc
        data = response.json()
        if not isinstance(data, dict) or "choices" not in data:
            raise BackendError(f"{model.id} returned an invalid chat completion")
        return data

    async def health(self, model: ModelSpec) -> bool:
        try:
            response = await self._client.get(f"{model.endpoint}/health", timeout=3.0)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def close(self) -> None:
        await self._client.aclose()
