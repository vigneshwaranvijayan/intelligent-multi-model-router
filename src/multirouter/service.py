from __future__ import annotations

import asyncio
import time
import uuid
from collections import OrderedDict
from typing import Any

from multirouter.analysis import RequestAnalyzer
from multirouter.backend import BackendClientProtocol, BackendError
from multirouter.config import AppSettings
from multirouter.domain import CandidateScore, DecisionRecord, RoutingPolicy
from multirouter.metrics import (
    BACKEND_LATENCY,
    FALLBACKS,
    QUEUE_DEPTH,
    REQUESTS,
    ROUTED,
    ROUTER_LATENCY,
)
from multirouter.registry import ModelRegistry
from multirouter.routing.base import Router
from multirouter.routing.utility import NoEligibleModelError
from multirouter.schemas import ChatCompletionRequest


class DecisionStore:
    def __init__(self, maximum_items: int = 5000) -> None:
        self._items: OrderedDict[str, DecisionRecord] = OrderedDict()
        self._maximum_items = maximum_items
        self._lock = asyncio.Lock()

    async def put(self, decision: DecisionRecord) -> None:
        async with self._lock:
            self._items[decision.request_id] = decision
            self._items.move_to_end(decision.request_id)
            while len(self._items) > self._maximum_items:
                self._items.popitem(last=False)

    async def get(self, request_id: str) -> DecisionRecord | None:
        async with self._lock:
            return self._items.get(request_id)


class RoutingService:
    def __init__(
        self,
        *,
        settings: AppSettings,
        registry: ModelRegistry,
        policies: dict[str, RoutingPolicy],
        router: Router,
        backend: BackendClientProtocol,
        analyzer: RequestAnalyzer | None = None,
        decisions: DecisionStore | None = None,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.policies = policies
        self.router = router
        self.backend = backend
        self.analyzer = analyzer or RequestAnalyzer()
        self.decisions = decisions or DecisionStore()

    def plan(self, request: ChatCompletionRequest):
        policy = self.policies.get(request.routing_policy)
        if policy is None:
            choices = ", ".join(sorted(self.policies))
            raise ValueError(
                f"Unknown routing policy '{request.routing_policy}'. Available: {choices}"
            )
        features = self.analyzer.analyse(request)
        with ROUTER_LATENCY.time():
            return self.router.route(
                features,
                policy,
                requested_model=request.model,
                request_maximum_latency_ms=request.maximum_latency_ms,
            )

    async def complete(self, request: ChatCompletionRequest) -> dict[str, Any]:
        request_id = f"route-{uuid.uuid4().hex}"
        try:
            plan = self.plan(request)
        except (ValueError, KeyError, NoEligibleModelError):
            REQUESTS.labels(policy=request.routing_policy, status="routing_error").inc()
            raise

        decision = DecisionRecord(
            request_id=request_id,
            selected_model=plan.selected_model,
            attempted_models=[],
            policy=plan.policy,
            features=plan.features,
            candidates=plan.candidates,
        )
        await self.decisions.put(decision)

        last_error: str | None = None
        for index, model_id in enumerate(plan.ordered_models):
            model = self.registry.get(model_id)
            decision.attempted_models.append(model_id)
            await self.registry.begin_request(model_id)
            QUEUE_DEPTH.labels(model=model_id).set(self.registry.runtime(model_id).queue_depth)
            started = time.perf_counter()
            try:
                payload = await self.backend.complete(model, request)
            except (BackendError, TimeoutError, OSError) as exc:
                elapsed_ms = (time.perf_counter() - started) * 1000
                await self.registry.record_failure(
                    model_id,
                    failure_threshold=self.settings.failure_threshold,
                    reset_seconds=self.settings.circuit_reset_seconds,
                )
                QUEUE_DEPTH.labels(model=model_id).set(self.registry.runtime(model_id).queue_depth)
                ROUTED.labels(model=model_id, outcome="failure").inc()
                last_error = str(exc)
                decision.error = last_error
                decision.backend_latency_ms = elapsed_ms
                continue

            elapsed_ms = (time.perf_counter() - started) * 1000
            await self.registry.record_success(model_id, elapsed_ms)
            QUEUE_DEPTH.labels(model=model_id).set(self.registry.runtime(model_id).queue_depth)
            BACKEND_LATENCY.labels(model=model_id).observe(elapsed_ms / 1000)
            ROUTED.labels(model=model_id, outcome="success").inc()
            REQUESTS.labels(policy=plan.policy, status="success").inc()

            decision.final_model = model_id
            decision.backend_latency_ms = elapsed_ms
            decision.fallback_used = index > 0
            decision.error = None
            if index > 0:
                FALLBACKS.inc()

            payload["id"] = payload.get("id", f"chatcmpl-{uuid.uuid4().hex}")
            payload["model"] = model_id
            if request.include_routing_trace:
                payload["routing"] = self._decision_payload(decision)
            return payload

        REQUESTS.labels(policy=plan.policy, status="backend_error").inc()
        decision.error = last_error or "All eligible backends failed"
        raise BackendError(decision.error)

    @staticmethod
    def _candidate_payload(candidate: CandidateScore) -> dict[str, Any]:
        return {
            "model": candidate.model_id,
            "eligible": candidate.eligible,
            "utility": candidate.utility,
            "predicted_quality": candidate.predicted_quality,
            "predicted_latency_ms": candidate.predicted_latency_ms,
            "estimated_cost": candidate.estimated_cost,
            "reliability": candidate.reliability,
            "reasons": list(candidate.reasons),
        }

    def simulation_payload(self, request: ChatCompletionRequest) -> dict[str, Any]:
        plan = self.plan(request)
        return {
            "selected_model": plan.selected_model,
            "fallback_order": list(plan.ordered_models),
            "policy": plan.policy,
            "features": {
                "task": plan.features.task,
                "complexity": plan.features.complexity,
                "prompt_tokens": plan.features.prompt_tokens,
                "estimated_output_tokens": plan.features.estimated_output_tokens,
                "privacy_level": plan.features.privacy_level,
                "required_capabilities": sorted(plan.features.required_capabilities),
                "contains_sensitive_data": plan.features.contains_sensitive_data,
            },
            "candidates": [self._candidate_payload(c) for c in plan.candidates],
        }

    def _decision_payload(self, decision: DecisionRecord) -> dict[str, Any]:
        return {
            "request_id": decision.request_id,
            "selected_model": decision.selected_model,
            "final_model": decision.final_model,
            "attempted_models": decision.attempted_models,
            "fallback_used": decision.fallback_used,
            "policy": decision.policy,
            "task": decision.features.task,
            "complexity_score": decision.features.complexity,
            "privacy_level": decision.features.privacy_level,
            "backend_latency_ms": round(decision.backend_latency_ms or 0.0, 3),
            "candidates": [self._candidate_payload(c) for c in decision.candidates],
        }
