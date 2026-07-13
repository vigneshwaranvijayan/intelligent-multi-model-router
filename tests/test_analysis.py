from multirouter.analysis import RequestAnalyzer
from multirouter.schemas import ChatCompletionRequest, ChatMessage


def test_detects_coding_task_and_complexity():
    request = ChatCompletionRequest(
        messages=[
            ChatMessage(
                role="user",
                content="Debug this Python race condition in a worker pool",
            )
        ]
    )
    features = RequestAnalyzer().analyse(request)
    assert features.task == "coding"
    assert features.complexity >= 0.48
    assert features.required_capabilities == frozenset({"chat"})


def test_sensitive_data_upgrades_privacy():
    request = ChatCompletionRequest(
        privacy_level="public",
        messages=[ChatMessage(role="user", content="Email alice@example.com with the summary")],
    )
    features = RequestAnalyzer().analyse(request)
    assert features.contains_sensitive_data is True
    assert features.privacy_level == "confidential"
