from __future__ import annotations

from multirouter.domain import (
    CandidateScore,
    ModelSpec,
    RequestFeatures,
    RoutingPlan,
    RoutingPolicy,
)
from multirouter.routing.base import Router
from multirouter.routing.constraints import eligibility_reasons, rejected_candidate


class NoEligibleModelError(RuntimeError):
    def __init__(self, candidates: list[CandidateScore]) -> None:
        super().__init__("No model satisfies the request constraints")
        self.candidates = candidates


class UtilityRouter(Router):
    def route(
        self,
        features: RequestFeatures,
        policy: RoutingPolicy,
        *,
        requested_model: str = "auto",
        request_maximum_latency_ms: int | None = None,
    ) -> RoutingPlan:
        candidates: list[CandidateScore] = []
        models = self.registry.all()

        if requested_model != "auto":
            models = [self.registry.get(requested_model)]

        for model in models:
            reasons = eligibility_reasons(
                model,
                features,
                policy,
                self.registry,
                request_maximum_latency_ms,
            )
            if reasons:
                candidates.append(rejected_candidate(model.id, reasons))
                continue
            candidates.append(self._score(model, features, policy))

        eligible = [candidate for candidate in candidates if candidate.eligible]
        if not eligible:
            raise NoEligibleModelError(candidates)
        eligible.sort(key=lambda candidate: candidate.utility or float("-inf"), reverse=True)

        return RoutingPlan(
            selected_model=eligible[0].model_id,
            ordered_models=tuple(candidate.model_id for candidate in eligible),
            policy=policy.name,
            features=features,
            candidates=tuple(candidates),
        )

    def _score(
        self,
        model: ModelSpec,
        features: RequestFeatures,
        policy: RoutingPolicy,
    ) -> CandidateScore:
        state = self.registry.runtime(model.id)
        affinity = model.task_affinity.get(features.task, 0.0)
        complexity_penalty = max(0.0, features.complexity - model.quality_prior) * 0.22
        predicted_quality = _clamp(model.quality_prior + affinity - complexity_penalty)

        latency = state.ewma_latency_ms or model.latency_prior_ms
        predicted_latency = latency * (1.0 + min(2.0, state.queue_depth * 0.15))
        estimated_cost = (
            features.prompt_tokens / 1000 * model.input_cost_per_1k
            + features.estimated_output_tokens / 1000 * model.output_cost_per_1k
        )
        reliability = _clamp(1.0 - state.consecutive_failures * 0.18)

        # Normalise terms to roughly [0, 1]. Lower latency and cost become larger scores.
        latency_score = 1.0 / (1.0 + predicted_latency / 1000.0)
        cost_score = 1.0 / (1.0 + estimated_cost * 1000.0)
        quality_shortfall = max(0.0, policy.minimum_quality - predicted_quality)
        utility = (
            policy.quality_weight * predicted_quality
            + policy.latency_weight * latency_score
            + policy.cost_weight * cost_score
            + policy.reliability_weight * reliability
            - quality_shortfall * 2.0
        )

        reasons = (
            f"task affinity={affinity:+.3f}",
            f"queue depth={state.queue_depth}",
            f"quality shortfall={quality_shortfall:.3f}",
        )
        return CandidateScore(
            model_id=model.id,
            eligible=True,
            utility=round(utility, 6),
            predicted_quality=round(predicted_quality, 6),
            predicted_latency_ms=round(predicted_latency, 3),
            estimated_cost=round(estimated_cost, 8),
            reliability=round(reliability, 6),
            reasons=reasons,
        )


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
