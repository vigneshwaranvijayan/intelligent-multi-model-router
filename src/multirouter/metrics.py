from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter(
    "multirouter_requests_total",
    "Total routed chat requests",
    ["policy", "status"],
)
ROUTED = Counter(
    "multirouter_model_routes_total",
    "Number of model routing attempts",
    ["model", "outcome"],
)
FALLBACKS = Counter(
    "multirouter_fallbacks_total",
    "Number of requests that used a fallback model",
)
ROUTER_LATENCY = Histogram(
    "multirouter_router_latency_seconds",
    "Time spent analysing and selecting a model",
)
BACKEND_LATENCY = Histogram(
    "multirouter_backend_latency_seconds",
    "Backend inference latency",
    ["model"],
)
QUEUE_DEPTH = Gauge(
    "multirouter_model_queue_depth",
    "Current in-flight request count per model",
    ["model"],
)
FEEDBACK = Histogram(
    "multirouter_feedback_score",
    "User-provided response quality score",
    buckets=(0.0, 0.25, 0.5, 0.75, 0.9, 1.0),
)
