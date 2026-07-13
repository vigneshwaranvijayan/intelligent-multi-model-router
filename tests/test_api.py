import pytest
from httpx import ASGITransport, AsyncClient

from multirouter.api import create_app


@pytest.mark.asyncio
async def test_api_chat_completion(bundle, make_service):
    service, _ = make_service()
    app = create_app(bundle=bundle, service=service)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": "Summarise this report"}],
                "routing_policy": "balanced",
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert "routing" in body


@pytest.mark.asyncio
async def test_simulation_rejects_unknown_policy(bundle, make_service):
    service, _ = make_service()
    app = create_app(bundle=bundle, service=service)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/routing/simulate",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "routing_policy": "does-not-exist",
            },
        )
    assert response.status_code == 422
