from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from multirouter.analysis import RequestAnalyzer
from multirouter.backend import BackendError
from multirouter.config import AppSettings, ConfigBundle, load_model_specs, load_policies
from multirouter.registry import ModelRegistry
from multirouter.routing.utility import UtilityRouter
from multirouter.service import RoutingService


class FakeBackend:
    def __init__(self, fail_models: set[str] | None = None) -> None:
        self.fail_models = fail_models or set()
        self.calls: list[str] = []

    async def complete(self, model, request) -> dict[str, Any]:
        self.calls.append(model.id)
        if model.id in self.fail_models:
            raise BackendError(f"forced failure for {model.id}")
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": model.id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": f"response from {model.id}"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    async def health(self, model) -> bool:
        return True

    async def close(self) -> None:
        return None


@pytest.fixture
def bundle() -> ConfigBundle:
    config_dir = Path(__file__).parents[1] / "configs"
    settings = AppSettings(
        config_dir=config_dir,
        health_check_interval_seconds=3600,
        failure_threshold=2,
        circuit_reset_seconds=60,
    )
    return ConfigBundle(
        settings=settings,
        models=load_model_specs(config_dir),
        policies=load_policies(config_dir),
    )


@pytest.fixture
def make_service(bundle):
    def factory(*, fail_models: set[str] | None = None):
        registry = ModelRegistry(bundle.models)
        backend = FakeBackend(fail_models)
        service = RoutingService(
            settings=bundle.settings,
            registry=registry,
            policies=bundle.policies,
            router=UtilityRouter(registry),
            backend=backend,
            analyzer=RequestAnalyzer(),
        )
        return service, backend

    return factory
