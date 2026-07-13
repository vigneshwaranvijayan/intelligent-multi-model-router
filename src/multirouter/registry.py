from __future__ import annotations

import asyncio
import time

from multirouter.domain import ModelRuntime, ModelSpec


class ModelRegistry:
    def __init__(self, models: list[ModelSpec]) -> None:
        if not models:
            raise ValueError("At least one model is required")
        self._models = {model.id: model for model in models}
        if len(self._models) != len(models):
            raise ValueError("Model IDs must be unique")
        self._runtime = {model.id: ModelRuntime() for model in models}
        self._lock = asyncio.Lock()

    def all(self) -> list[ModelSpec]:
        return list(self._models.values())

    def get(self, model_id: str) -> ModelSpec:
        try:
            return self._models[model_id]
        except KeyError as exc:
            raise KeyError(f"Unknown model: {model_id}") from exc

    def runtime(self, model_id: str) -> ModelRuntime:
        return self._runtime[model_id]

    def is_available(self, model_id: str) -> bool:
        state = self._runtime[model_id]
        return state.healthy and time.monotonic() >= state.circuit_open_until

    async def begin_request(self, model_id: str) -> None:
        async with self._lock:
            self._runtime[model_id].queue_depth += 1

    async def record_success(self, model_id: str, latency_ms: float) -> None:
        async with self._lock:
            state = self._runtime[model_id]
            state.queue_depth = max(0, state.queue_depth - 1)
            state.healthy = True
            state.consecutive_failures = 0
            alpha = 0.25
            state.ewma_latency_ms = (
                latency_ms
                if state.ewma_latency_ms is None
                else alpha * latency_ms + (1 - alpha) * state.ewma_latency_ms
            )

    async def record_failure(
        self,
        model_id: str,
        *,
        failure_threshold: int,
        reset_seconds: float,
    ) -> None:
        async with self._lock:
            state = self._runtime[model_id]
            state.queue_depth = max(0, state.queue_depth - 1)
            state.consecutive_failures += 1
            if state.consecutive_failures >= failure_threshold:
                state.circuit_open_until = time.monotonic() + reset_seconds

    async def set_health(self, model_id: str, healthy: bool) -> None:
        async with self._lock:
            self._runtime[model_id].healthy = healthy

    def snapshot(self) -> dict[str, dict[str, object]]:
        now = time.monotonic()
        return {
            model.id: {
                "endpoint": model.endpoint,
                "healthy": state.healthy,
                "available": state.healthy and now >= state.circuit_open_until,
                "queue_depth": state.queue_depth,
                "ewma_latency_ms": state.ewma_latency_ms,
                "consecutive_failures": state.consecutive_failures,
                "circuit_open": now < state.circuit_open_until,
                "capabilities": sorted(model.capabilities),
                "privacy_levels": sorted(model.privacy_levels),
            }
            for model in self._models.values()
            if (state := self._runtime[model.id])
        }
