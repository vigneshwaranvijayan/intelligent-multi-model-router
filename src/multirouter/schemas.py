from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["system", "user", "assistant", "tool"]
    content: str

    @field_validator("content")
    @classmethod
    def content_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message content cannot be blank")
        return value


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = "auto"
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=32768)
    stream: bool = False
    routing_policy: str = "balanced"
    maximum_latency_ms: int | None = Field(default=None, ge=1)
    privacy_level: Literal["public", "internal", "confidential"] = "public"
    required_capabilities: list[str] = Field(default_factory=list)
    include_routing_trace: bool = True


class SimulationRequest(ChatCompletionRequest):
    stream: bool = False


class FeedbackRequest(BaseModel):
    request_id: str
    score: float = Field(ge=0.0, le=1.0)
    accepted: bool | None = None
    comment: str | None = Field(default=None, max_length=2000)


class HealthResponse(BaseModel):
    status: str
    models: dict[str, dict[str, Any]]
