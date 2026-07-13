from __future__ import annotations

from multirouter.domain import CandidateScore, RequestFeatures, RoutingPlan, RoutingPolicy
from multirouter.routing.base import Router
from multirouter.routing.constraints import eligibility_reasons, rejected_candidate
from multirouter.routing.utility import NoEligibleModelError


class RuleRouter(Router):
    """Transparent baseline router used for comparison with learned/utility routers."""

    def route(
        self,
        features: RequestFeatures,
        policy: RoutingPolicy,
        *,
        requested_model: str = "auto",
        request_maximum_latency_ms: int | None = None,
    ) -> RoutingPlan:
        candidates: list[CandidateScore] = []
        eligible_ids: list[str] = []

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
            else:
                eligible_ids.append(model.id)
                candidates.append(
                    CandidateScore(
                        model_id=model.id,
                        eligible=True,
                        utility=None,
                        predicted_quality=None,
                        predicted_latency_ms=None,
                        estimated_cost=None,
                        reliability=None,
                        reasons=("eligible under static rules",),
                    )
                )

        if not eligible_ids:
            raise NoEligibleModelError(candidates)

        if requested_model != "auto":
            ordered = eligible_ids
        else:
            ordered = self._preferred_order(features, eligible_ids)

        return RoutingPlan(
            selected_model=ordered[0],
            ordered_models=tuple(ordered),
            policy=policy.name,
            features=features,
            candidates=tuple(candidates),
        )

    def _preferred_order(self, features: RequestFeatures, eligible_ids: list[str]) -> list[str]:
        def priority(model_id: str) -> tuple[int, float]:
            model = self.registry.get(model_id)
            specialist = 0
            if features.task in model.capabilities:
                specialist -= 3
            if features.task == "coding" and "coding" in model.capabilities:
                specialist -= 5
            if features.complexity < 0.35:
                return (specialist, model.input_cost_per_1k)
            return (specialist, -model.quality_prior)

        return sorted(eligible_ids, key=priority)
