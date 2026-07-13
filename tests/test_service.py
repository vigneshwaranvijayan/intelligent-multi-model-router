import pytest

from multirouter.schemas import ChatCompletionRequest, ChatMessage


@pytest.mark.asyncio
async def test_service_returns_routing_trace(make_service):
    service, backend = make_service()
    request = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="Classify this review as positive or negative")]
    )
    result = await service.complete(request)
    assert result["routing"]["final_model"] in {"small-general", "medium-general"}
    assert result["routing"]["attempted_models"] == backend.calls
    assert result["choices"][0]["message"]["content"].startswith("response from")


@pytest.mark.asyncio
async def test_fallback_after_backend_failure(make_service):
    initial_service, _ = make_service()
    request = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="Debug this Python deadlock")]
    )
    plan = initial_service.plan(request)
    first = plan.ordered_models[0]

    service, backend = make_service(fail_models={first})
    result = await service.complete(request)
    assert backend.calls[0] == first
    assert len(backend.calls) >= 2
    assert result["routing"]["fallback_used"] is True
    assert result["routing"]["final_model"] != first
