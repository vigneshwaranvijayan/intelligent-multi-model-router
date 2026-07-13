from __future__ import annotations

import math
import re

from multirouter.domain import RequestFeatures
from multirouter.schemas import ChatCompletionRequest

_CODE = re.compile(
    r"\b(python|javascript|typescript|java|c\+\+|rust|golang|sql|code|bug|debug|"
    r"function|class|api|stack trace|exception|deadlock|race condition)\b",
    re.I,
)
_MATH = re.compile(
    r"\b(calculate|equation|integral|derivative|probability|theorem|proof|matrix|"
    r"algebra|geometry)\b",
    re.I,
)
_SUMMARY = re.compile(r"\b(summarise|summarize|summary|condense|tl;?dr|key points)\b", re.I)
_EXTRACTION = re.compile(r"\b(extract|entities|fields|json schema|structured output|parse)\b", re.I)
_CLASSIFY = re.compile(r"\b(classify|classification|sentiment|label|category|spam)\b", re.I)
_REASONING = re.compile(
    r"\b(analyse|analyze|compare|reason|trade-?off|root cause|step by step|strategy)\b",
    re.I,
)
_SENSITIVE = re.compile(
    r"(?:\b\d{3}-\d{2}-\d{4}\b)|(?:\b(?:\d[ -]*?){13,16}\b)|"
    r"(?:api[_-]?key\s*[:=])|(?:password\s*[:=])|(?:secret\s*[:=])|"
    r"(?:[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})",
    re.I,
)


def estimate_tokens(text: str) -> int:
    # A deterministic approximation suitable for routing before a tokenizer is selected.
    return max(1, math.ceil(len(text) / 4))


class RequestAnalyzer:
    def analyse(self, request: ChatCompletionRequest) -> RequestFeatures:
        prompt = "\n".join(message.content for message in request.messages)
        task = self._task(prompt)
        prompt_tokens = estimate_tokens(prompt)
        output_tokens = request.max_tokens or min(1024, max(128, prompt_tokens // 2))
        complexity = self._complexity(prompt, prompt_tokens, task)
        sensitive = bool(_SENSITIVE.search(prompt))

        # Only caller-declared capabilities are hard constraints. Task detection is a
        # preference signal so a general model can still act as a fallback.
        required = set(request.required_capabilities)
        required.add("chat")

        privacy = request.privacy_level
        if sensitive and privacy == "public":
            privacy = "confidential"

        return RequestFeatures(
            task=task,
            complexity=complexity,
            prompt_tokens=prompt_tokens,
            estimated_output_tokens=output_tokens,
            privacy_level=privacy,
            required_capabilities=frozenset(required),
            contains_sensitive_data=sensitive,
        )

    @staticmethod
    def _task(prompt: str) -> str:
        if _CODE.search(prompt):
            return "coding"
        if _MATH.search(prompt):
            return "mathematics"
        if _SUMMARY.search(prompt):
            return "summarisation"
        if _EXTRACTION.search(prompt):
            return "extraction"
        if _CLASSIFY.search(prompt):
            return "classification"
        if _REASONING.search(prompt):
            return "reasoning"
        return "general"

    @staticmethod
    def _complexity(prompt: str, prompt_tokens: int, task: str) -> float:
        length_component = min(0.45, prompt_tokens / 12000)
        task_component = {
            "classification": 0.10,
            "extraction": 0.18,
            "summarisation": 0.28,
            "general": 0.22,
            "reasoning": 0.45,
            "coding": 0.48,
            "mathematics": 0.52,
        }.get(task, 0.25)
        structure_component = min(0.18, prompt.count("\n") * 0.012)
        difficulty_terms = len(
            re.findall(
                r"\b(complex|advanced|multi-step|optimise|optimize|prove|architecture|"
                r"concurrent)\b",
                prompt,
                re.I,
            )
        )
        lexical_component = min(0.14, difficulty_terms * 0.035)
        total = length_component + task_component + structure_component + lexical_component
        return round(min(1.0, total), 4)
