from __future__ import annotations

import asyncio
import os
import time
import uuid
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str
    content: str


class MockRequest(BaseModel):
    model: str
    messages: list[Message] = Field(min_length=1)
    max_tokens: int | None = None
    temperature: float = 0.2
    stream: bool = False


app = FastAPI(title="Mock OpenAI-Compatible Model")


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": os.getenv("MODEL_ID", "mock-model"),
        "quality": float(os.getenv("MODEL_QUALITY", "0.75")),
    }


@app.post("/v1/chat/completions")
async def complete(request: MockRequest) -> dict[str, Any]:
    latency_ms = float(os.getenv("MODEL_LATENCY_MS", "150"))
    failure_rate = float(os.getenv("MODEL_FAILURE_RATE", "0"))
    prompt = "\n".join(message.content for message in request.messages)

    # Deterministic failure trigger is more useful than randomness in tests and demos.
    if "[force-failure]" in prompt.lower() or failure_rate >= 1.0:
        raise HTTPException(status_code=503, detail="Simulated backend failure")

    await asyncio.sleep(latency_ms / 1000)
    model_id = os.getenv("MODEL_ID", request.model)
    speciality = os.getenv("MODEL_SPECIALITY", "general")
    answer = _answer(model_id, speciality, prompt)
    prompt_tokens = max(1, len(prompt) // 4)
    completion_tokens = max(1, len(answer) // 4)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _answer(model_id: str, speciality: str, prompt: str) -> str:
    preview = " ".join(prompt.split())[:180]
    if speciality == "coding":
        return (
            f"[{model_id}] Coding analysis: inspect shared state, lock ordering, exception paths, "
            f"and add a reproducible test. Request preview: {preview}"
        )
    return f"[{model_id}] Processed request: {preview}"


def run() -> None:
    port = int(os.getenv("MODEL_PORT", "8101"))
    uvicorn.run("multirouter.mock_backend:app", host="0.0.0.0", port=port, reload=False)
