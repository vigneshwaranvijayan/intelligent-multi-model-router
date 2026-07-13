from __future__ import annotations

from multirouter.domain import CandidateScore, ModelSpec, RequestFeatures, RoutingPolicy
from multirouter.registry import ModelRegistry


def eligibility_reasons(
    model: ModelSpec,
    features: RequestFeatures,
    policy: RoutingPolicy,
    registry: ModelRegistry,
    request_maximum_latency_ms: int | None,
) -> list[str]:
    reasons: list[str] = []
    if not registry.is_available(model.id):
        reasons.append("model is unhealthy or circuit breaker is open")
    if features.prompt_tokens + features.estimated_output_tokens > model.max_context_tokens:
        reasons.append("context window is too small")
    if features.privacy_level not in model.privacy_levels:
        reasons.append(f"privacy level '{features.privacy_level}' is not allowed")
    missing = features.required_capabilities - model.capabilities
    if missing:
        reasons.append(f"missing capabilities: {', '.join(sorted(missing))}")

    state = registry.runtime(model.id)
    latency = state.ewma_latency_ms or model.latency_prior_ms
    latency *= 1.0 + min(2.0, state.queue_depth * 0.15)
    maximum = request_maximum_latency_ms or policy.maximum_latency_ms
    if maximum is not None and latency > maximum:
        reasons.append(f"predicted latency {latency:.0f} ms exceeds limit {maximum:.0f} ms")
    return reasons


def rejected_candidate(model_id: str, reasons: list[str]) -> CandidateScore:
    return CandidateScore(
        model_id=model_id,
        eligible=False,
        utility=None,
        predicted_quality=None,
        predicted_latency_ms=None,
        estimated_cost=None,
        reliability=None,
        reasons=tuple(reasons),
    )
