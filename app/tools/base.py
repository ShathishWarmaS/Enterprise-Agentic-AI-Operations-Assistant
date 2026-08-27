"""Tool contract shared by the Action agent and the MCP server.

A `Tool` is a name, a human description, a JSON-Schema for its input, and a
`run()` that takes validated arguments and returns a JSON-serialisable dict.
The schema is plain JSON Schema so the same definition works as an Anthropic
tool-use `input_schema` and as an MCP tool definition.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

from pydantic import BaseModel

from app.schemas.agents import ToolCall


class ToolError(RuntimeError):
    """A tool failed in an expected way (bad args, missing table, empty corpus)."""


class Tool(ABC):
    name: str
    description: str
    input_schema: dict

    @abstractmethod
    def run(self, arguments: dict) -> dict:
        """Execute with already-validated arguments."""

    def to_anthropic(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def to_mcp(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }

    def invoke(self, arguments: dict) -> ToolCall:
        """Run and wrap the outcome in a ToolCall trace record. Never raises."""
        started = time.perf_counter()
        try:
            result = self.run(arguments)
            return ToolCall(
                tool=self.name,
                arguments=arguments,
                ok=True,
                result=result,
                duration_ms=_elapsed_ms(started),
            )
        except ToolError as exc:
            return ToolCall(
                tool=self.name,
                arguments=arguments,
                ok=False,
                error=str(exc),
                duration_ms=_elapsed_ms(started),
            )


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def schema_from_model(model: type[BaseModel]) -> dict:
    """JSON Schema for a tool input, with Pydantic's $defs inlined enough for LLMs."""
    return model.model_json_schema()
