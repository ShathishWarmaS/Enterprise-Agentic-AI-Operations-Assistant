"""LLM client: missing key, malformed output handling, retry-then-repair."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from app.config import Settings
from app.services.llm import LLMClient, LLMOutputError, LLMUnavailable


class _Shape(BaseModel):
    name: str
    count: int


def test_claude_mode_without_key_is_rejected():
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        Settings(llm_mode="claude", anthropic_api_key=None)


def test_mock_client_refuses_to_call_out():
    client = LLMClient(Settings(llm_mode="mock"))
    assert client.enabled is False
    with pytest.raises(LLMUnavailable):
        client.message(system="s", user="u")


def _client_with_replies(replies: list[str]) -> LLMClient:
    client = LLMClient(Settings(llm_mode="claude", anthropic_api_key="test-key"))

    calls = {"n": 0}

    def fake_call(*, system, user, tools):
        from app.services.llm import LLMReply

        idx = min(calls["n"], len(replies) - 1)
        calls["n"] += 1
        return LLMReply(text=replies[idx])

    client._call = fake_call  # type: ignore[method-assign]
    client._client = object()  # bypass real SDK
    return client


def test_structured_parses_valid_json():
    client = _client_with_replies(['{"name": "db", "count": 3}'])
    out = client.structured(system="s", user="u", model=_Shape)
    assert out == _Shape(name="db", count=3)


def test_structured_repairs_after_one_bad_reply():
    client = _client_with_replies(["not json at all", '{"name": "db", "count": 4}'])
    out = client.structured(system="s", user="u", model=_Shape)
    assert out.count == 4


def test_structured_gives_up_after_second_failure():
    client = _client_with_replies(["nope", "still nope"])
    with pytest.raises(LLMOutputError):
        client.structured(system="s", user="u", model=_Shape)
