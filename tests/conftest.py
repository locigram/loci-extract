"""Shared test fixtures."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class _StubChoice:
    content: str = ""

    @property
    def message(self):
        # openai SDK returns .choices[0].message.content — mimic that shape
        return self


@dataclass
class _StubResponse:
    choices: list = field(default_factory=list)


class StubLlmClient:
    """Fake OpenAI client that returns canned responses in order.

    Use:
        client = StubLlmClient(["{\"documents\":[]}"])
        client.chat.completions.create(...)  # returns first canned response
    """

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):  # noqa: D401
        self.calls.append(kwargs)
        if not self._responses:
            return _StubResponse(choices=[_StubChoice(content="{\"documents\": []}")])
        content = self._responses.pop(0)
        choice = type("Choice", (), {})()
        message = type("Message", (), {})()
        message.content = content
        choice.message = message
        return _StubResponse(choices=[choice])
