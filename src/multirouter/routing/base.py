from __future__ import annotations

from abc import ABC, abstractmethod

from multirouter.domain import RequestFeatures, RoutingPlan, RoutingPolicy
from multirouter.registry import ModelRegistry


class Router(ABC):
    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry

    @abstractmethod
    def route(
        self,
        features: RequestFeatures,
        policy: RoutingPolicy,
        *,
        requested_model: str = "auto",
        request_maximum_latency_ms: int | None = None,
    ) -> RoutingPlan:
        raise NotImplementedError
