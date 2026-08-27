"""LLM access layer.

Everything that talks to Claude goes through `LLMClient`. In `mock` mode the
client raises if an agent tries to call it - mock agents are expected to run
pure-Python logic instead, keeping the two paths honestly separate.

`structured()` is the workhorse: it asks Claude for JSON matching a Pydantic
model, and on malformed output it retries exactly once with the parser error fed
back in. A second failure raises `LLMOutputError` rather than guessing.
"""

from __future__ import annotations

import json
import logging
from typing import TypeVar

from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import LLMMode, Settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Transport / API failure (network, auth, rate limit after retries)."""


class LLMOutputError(ValueError):
    """The model replied, but not with usable content."""


class LLMUnavailable(RuntimeError):
    """Raised when mock mode code path unexpectedly reaches the real client."""


class ToolSpec(BaseModel):
    name: str
    description: str
    input_schema: dict


class LLMReply(BaseModel):
    text: str
    tool_calls: list[dict] = []  # [{"name": str, "input": dict, "id": str}]
    stop_reason: str | None = None


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = None
        if settings.llm_mode is LLMMode.claude:
            self._client = self._build_client()

    @property
    def mode(self) -> str:
        return self._settings.llm_mode.value

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def _build_client(self):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise LLMError("anthropic SDK not installed but LLM_MODE=claude") from exc
        return anthropic.Anthropic(api_key=self._settings.anthropic_api_key)

    def _require_client(self):
        if self._client is None:
            raise LLMUnavailable("LLMClient.mode is 'mock'; agents must use their pure-Python path")
        return self._client

    @retry(
        retry=retry_if_exception_type(LLMError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=8),
        reraise=True,
    )
    def _call(self, *, system: str, user: str, tools: list[ToolSpec] | None) -> LLMReply:
        import anthropic

        client = self._require_client()
        try:
            resp = client.messages.create(
                model=self._settings.anthropic_model,
                max_tokens=self._settings.anthropic_max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                tools=[t.model_dump() for t in tools] if tools else anthropic.NOT_GIVEN,
            )
        except (
            anthropic.APIConnectionError,
            anthropic.RateLimitError,
            anthropic.InternalServerError,
        ) as exc:
            raise LLMError(f"transient Anthropic API error: {exc}") from exc
        except anthropic.APIStatusError as exc:  # 4xx - not retryable
            raise LLMOutputError(f"Anthropic API rejected the request: {exc}") from exc

        text_parts = [b.text for b in resp.content if b.type == "text"]
        tool_calls = [
            {"name": b.name, "input": b.input, "id": b.id}
            for b in resp.content
            if b.type == "tool_use"
        ]
        return LLMReply(
            text="\n".join(text_parts).strip(),
            tool_calls=tool_calls,
            stop_reason=resp.stop_reason,
        )

    def message(self, *, system: str, user: str, tools: list[ToolSpec] | None = None) -> LLMReply:
        return self._call(system=system, user=user, tools=tools)

    def structured(self, *, system: str, user: str, model: type[T]) -> T:
        schema = json.dumps(model.model_json_schema(), indent=2)
        instruction = (
            f"{system}\n\nReply with ONLY a JSON object matching this schema. "
            f"No markdown fences, no prose.\n\nSCHEMA:\n{schema}"
        )
        reply = self._call(system=instruction, user=user, tools=None)
        try:
            return self._parse(reply.text, model)
        except LLMOutputError as first_error:
            logger.warning("structured output invalid, retrying once: %s", first_error)
            repair = (
                f"{user}\n\nYour previous reply could not be parsed: {first_error}\n"
                f"Return corrected JSON only."
            )
            reply = self._call(system=instruction, user=repair, tools=None)
            return self._parse(reply.text, model)

    @staticmethod
    def _parse(text: str, model: type[T]) -> T:
        payload = text.strip()
        if payload.startswith("```"):
            payload = payload.strip("`")
            payload = payload[payload.find("\n") + 1 :] if "\n" in payload else payload
        start, end = payload.find("{"), payload.rfind("}")
        if start == -1 or end == -1:
            raise LLMOutputError(f"no JSON object found in reply: {text[:200]!r}")
        try:
            data = json.loads(payload[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMOutputError(f"reply was not valid JSON: {exc}") from exc
        try:
            return model.model_validate(data)
        except ValidationError as exc:
            raise LLMOutputError(f"JSON did not match {model.__name__}: {exc}") from exc
