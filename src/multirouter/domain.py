from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class PrivacyLevel(StrEnum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    id: str
    endpoint: str
    max_context_tokens: int
    privacy_levels: frozenset[str]
    capabilities: frozenset[str]
    input_cost_per_1k: float
    output_cost_per_1k: float
    quality_prior: float
    latency_prior_ms: float
    task_affinity: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RoutingPolicy:
    name: str
    quality_weight: float
    latency_weight: float
    cost_weight: float
    reliability_weight: float
    minimum_quality: float = 0.0
    maximum_latency_ms: float | None = None


@dataclass(slots=True)
class ModelRuntime:
    healthy: bool = True
    queue_depth: int = 0
    ewma_latency_ms: float | None = None
    consecutive_failures: int = 0
    circuit_open_until: float = 0.0


@dataclass(frozen=True, slots=True)
class RequestFeatures:
    task: str
    complexity: float
    prompt_tokens: int
    estimated_output_tokens: int
    privacy_level: str
    required_capabilities: frozenset[str]
    contains_sensitive_data: bool


@dataclass(frozen=True, slots=True)
class CandidateScore:
    model_id: str
    eligible: bool
    utility: float | None
    predicted_quality: float | None
    predicted_latency_ms: float | None
    estimated_cost: float | None
    reliability: float | None
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RoutingPlan:
    selected_model: str
    ordered_models: tuple[str, ...]
    policy: str
    features: RequestFeatures
    candidates: tuple[CandidateScore, ...]


@dataclass(slots=True)
class DecisionRecord:
    request_id: str
    selected_model: str
    attempted_models: list[str]
    policy: str
    features: RequestFeatures
    candidates: tuple[CandidateScore, ...]
    fallback_used: bool = False
    final_model: str | None = None
    error: str | None = None
    backend_latency_ms: float | None = None
    response_metadata: dict[str, Any] = field(default_factory=dict)
