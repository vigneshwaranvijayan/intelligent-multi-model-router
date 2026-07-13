import pytest

from multirouter.analysis import RequestAnalyzer
from multirouter.registry import ModelRegistry
from multirouter.routing.utility import NoEligibleModelError, UtilityRouter
from multirouter.schemas import ChatCompletionRequest, ChatMessage


def test_routes_coding_to_specialist(bundle):
    registry = ModelRegistry(bundle.models)
    router = UtilityRouter(registry)
    request = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="Find the deadlock in this Python service")]
    )
    features = RequestAnalyzer().analyse(request)
    plan = router.route(features, bundle.policies["balanced"])
    assert plan.selected_model == "code-specialist"


def test_confidential_coding_rejects_non_private_specialist(bundle):
    registry = ModelRegistry(bundle.models)
    router = UtilityRouter(registry)
    request = ChatCompletionRequest(
        privacy_level="confidential",
        messages=[ChatMessage(role="user", content="Debug this Python function")],
    )
    features = RequestAnalyzer().analyse(request)
    plan = router.route(features, bundle.policies["balanced"])
    assert plan.selected_model != "code-specialist"
    specialist = next(c for c in plan.candidates if c.model_id == "code-specialist")
    assert specialist.eligible is False
    assert "privacy" in " ".join(specialist.reasons)


def test_impossible_capability_has_explanations(bundle):
    registry = ModelRegistry(bundle.models)
    router = UtilityRouter(registry)
    request = ChatCompletionRequest(
        required_capabilities=["vision"],
        messages=[ChatMessage(role="user", content="Describe the image")],
    )
    features = RequestAnalyzer().analyse(request)
    with pytest.raises(NoEligibleModelError) as captured:
        router.route(features, bundle.policies["balanced"])
    assert len(captured.value.candidates) == 3
    assert all(not candidate.eligible for candidate in captured.value.candidates)
