from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any, AsyncIterator

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from multirouter.analysis import RequestAnalyzer
from multirouter.backend import BackendError, HTTPBackendClient
from multirouter.config import AppSettings, ConfigBundle, load_bundle
from multirouter.metrics import FEEDBACK
from multirouter.registry import ModelRegistry
from multirouter.routing import RuleRouter, UtilityRouter
from multirouter.routing.utility import NoEligibleModelError
from multirouter.schemas import (
    ChatCompletionRequest,
    FeedbackRequest,
    HealthResponse,
    SimulationRequest,
)
from multirouter.service import RoutingService

logger = logging.getLogger(__name__)


def build_service(bundle: ConfigBundle, backend: HTTPBackendClient | None = None) -> RoutingService:
    registry = ModelRegistry(bundle.models)
    router = RuleRouter(registry) if bundle.settings.router == "rule" else UtilityRouter(registry)
    return RoutingService(
        settings=bundle.settings,
        registry=registry,
        policies=bundle.policies,
        router=router,
        backend=backend or HTTPBackendClient(bundle.settings.request_timeout_seconds),
        analyzer=RequestAnalyzer(),
    )


def create_app(
    *,
    bundle: ConfigBundle | None = None,
    service: RoutingService | None = None,
) -> FastAPI:
    effective_bundle = bundle or load_bundle()
    effective_service = service or build_service(effective_bundle)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.health_task = asyncio.create_task(_health_loop(app))
        yield
        app.state.health_task.cancel()
        await asyncio.gather(app.state.health_task, return_exceptions=True)
        await app.state.service.backend.close()

    application = FastAPI(
        title="Intelligent Multi-Model Router",
        version="0.1.0",
        description="SLA-aware and explainable model routing gateway.",
        lifespan=lifespan,
    )
    application.state.service = effective_service

    @application.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        snapshot = request.app.state.service.registry.snapshot()
        any_available = any(bool(item["available"]) for item in snapshot.values())
        return HealthResponse(status="ok" if any_available else "degraded", models=snapshot)

    @application.get("/ready")
    async def ready(request: Request) -> JSONResponse:
        snapshot = request.app.state.service.registry.snapshot()
        if not any(bool(item["available"]) for item in snapshot.values()):
            return JSONResponse(
                status_code=503, content={"status": "not_ready", "models": snapshot}
            )
        return JSONResponse(content={"status": "ready"})

    @application.get("/models")
    async def models(request: Request) -> dict[str, Any]:
        service: RoutingService = request.app.state.service
        return {
            "object": "list",
            "data": [
                {
                    "id": model.id,
                    "object": "model",
                    "max_context_tokens": model.max_context_tokens,
                    "capabilities": sorted(model.capabilities),
                    "privacy_levels": sorted(model.privacy_levels),
                    "runtime": service.registry.snapshot()[model.id],
                }
                for model in service.registry.all()
            ],
        }

    @application.post("/routing/simulate")
    async def simulate(payload: SimulationRequest, request: Request) -> dict[str, Any]:
        try:
            return request.app.state.service.simulation_payload(payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except NoEligibleModelError as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": str(exc),
                    "candidates": [
                        request.app.state.service._candidate_payload(candidate)
                        for candidate in exc.candidates
                    ],
                },
            ) from exc

    @application.post("/v1/chat/completions")
    async def chat_completions(payload: ChatCompletionRequest, request: Request):
        try:
            result = await request.app.state.service.complete(payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except NoEligibleModelError as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": str(exc),
                    "candidates": [
                        request.app.state.service._candidate_payload(candidate)
                        for candidate in exc.candidates
                    ],
                },
            ) from exc
        except BackendError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        if not payload.stream:
            return result
        return StreamingResponse(
            _stream_result(result),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @application.get("/routing/decisions/{request_id}")
    async def decision(request_id: str, request: Request) -> dict[str, Any]:
        record = await request.app.state.service.decisions.get(request_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Routing decision not found")
        payload = asdict(record)
        payload["features"]["required_capabilities"] = sorted(record.features.required_capabilities)
        return payload

    @application.post("/feedback")
    async def feedback(payload: FeedbackRequest) -> dict[str, Any]:
        FEEDBACK.observe(payload.score)
        return {"status": "accepted", "request_id": payload.request_id}

    @application.get("/metrics")
    async def metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return application


async def _stream_result(result: dict[str, Any]) -> AsyncIterator[str]:
    choices = result.get("choices", [])
    content = ""
    if choices:
        content = choices[0].get("message", {}).get("content", "")
    chunk_size = 24
    for index in range(0, len(content), chunk_size):
        chunk = {
            "id": result.get("id"),
            "object": "chat.completion.chunk",
            "model": result.get("model"),
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": content[index : index + chunk_size]},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0)
    final = {
        "id": result.get("id"),
        "object": "chat.completion.chunk",
        "model": result.get("model"),
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "routing": result.get("routing"),
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"


async def _health_loop(app: FastAPI) -> None:
    service: RoutingService = app.state.service
    interval = service.settings.health_check_interval_seconds
    while True:
        for model in service.registry.all():
            try:
                healthy = await service.backend.health(model)
                await service.registry.set_health(model.id, healthy)
            except Exception:  # health checks must never crash the process
                logger.exception("Health check failed for %s", model.id)
                await service.registry.set_health(model.id, False)
        await asyncio.sleep(interval)


app = create_app()


def run() -> None:
    settings = AppSettings()
    logging.basicConfig(level=settings.log_level)
    uvicorn.run("multirouter.api:app", host="0.0.0.0", port=8000, reload=False)
