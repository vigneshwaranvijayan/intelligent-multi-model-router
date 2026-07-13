# Intelligent Multi-Model Routing and AI Serving Platform

A runnable AI-serving platform that routes OpenAI-style chat requests across multiple model endpoints using task type, complexity, privacy, predicted quality, latency, cost, health, and live queue pressure.

## What is implemented

- OpenAI-compatible `POST /v1/chat/completions`
- Rule-based and utility-based routers
- Hard constraints for privacy, capabilities, context size, health, and latency
- Explainable routing traces with accepted and rejected candidates
- Automatic fallback when a backend fails
- Lightweight circuit breaker and runtime health state
- Prometheus metrics at `/metrics`
- Routing simulation endpoint that does not run inference
- Three Dockerised mock model services for local development
- Unit and API tests
- YAML model and policy configuration

## Architecture

```text
Client
  -> FastAPI gateway
      -> request analyser
      -> hard constraint filter
      -> rule/utility router
      -> selected model endpoint
      -> fallback candidate(s) on failure
      -> routing decision store + Prometheus metrics
```

## Run locally with Docker

```bash
cp .env.example .env
docker compose up --build
```

Gateway: `http://localhost:8000`

API documentation: `http://localhost:8000/docs`

Prometheus metrics: `http://localhost:8000/metrics`

## Run without Docker

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

# In separate terminals
MODEL_ID=small-general MODEL_PORT=8101 MODEL_LATENCY_MS=120 multirouter-mock
MODEL_ID=medium-general MODEL_PORT=8102 MODEL_LATENCY_MS=300 multirouter-mock
MODEL_ID=code-specialist MODEL_PORT=8103 MODEL_LATENCY_MS=450 multirouter-mock

MULTIROUTER_CONFIG_DIR=configs multirouter
```

## Example request

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Find the bug in this Python function"}],
    "routing_policy": "balanced",
    "maximum_latency_ms": 3000,
    "privacy_level": "internal"
  }' | python -m json.tool
```

The response contains standard chat completion fields plus a `routing` object:

```json
{
  "routing": {
    "selected_model": "code-specialist",
    "task": "coding",
    "complexity_score": 0.49,
    "attempted_models": ["code-specialist"],
    "candidates": []
  }
}
```

## Simulate a routing decision

```bash
curl -s http://localhost:8000/routing/simulate \
  -H 'content-type: application/json' \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Summarise this quarterly report"}],
    "routing_policy": "cost_first"
  }' | python -m json.tool
```

## Useful failure test

Stop the code-specialist service:

```bash
docker compose stop code-model
```

Send a coding request again. The gateway records the failed attempt and falls back to the next eligible model.

## Tests

```bash
pip install -e '.[dev]'
pytest
ruff check .
```

## Replace mocks with real models

Edit `configs/models.yaml` and set each `endpoint` to an OpenAI-compatible server such as vLLM. No routing code needs to change.

## Next engineering milestones

1. Persist traces and feedback in PostgreSQL.
2. Add EWMA latency updates from successful requests.
3. Train a supervised ranking model from benchmark outcomes.
4. Add confidence-gated escalation and contextual-bandit routing.
5. Deploy with Ray Serve or Kubernetes and route using replica-level telemetry.
6. Add Grafana dashboards and failure-injection load tests.

## Licence

MIT
