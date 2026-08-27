"""Wire concrete tools together and expose them by name.

The registry is the single place the Action agent and the MCP server look up
tools, so the tool set stays identical across both surfaces.
"""

from __future__ import annotations

from app.data.tables import TableStore
from app.retrieval.retriever import Retriever
from app.tools.base import Tool
from app.tools.data_tools import CheckSchemaTool, ComputeMetricsTool, QueryDataTool
from app.tools.incident_tools import DraftIncidentSummaryTool, GenerateChecklistTool
from app.tools.search import SearchDocumentsTool


class ToolRegistry:
    def __init__(self, *, retriever: Retriever, table_store: TableStore) -> None:
        tools: list[Tool] = [
            SearchDocumentsTool(retriever),
            QueryDataTool(table_store),
            ComputeMetricsTool(table_store),
            CheckSchemaTool(table_store),
            DraftIncidentSummaryTool(),
            GenerateChecklistTool(),
        ]
        self._by_name: dict[str, Tool] = {t.name: t for t in tools}

    def get(self, name: str) -> Tool:
        try:
            return self._by_name[name]
        except KeyError:
            raise KeyError(f"unknown tool {name!r}; available: {list(self._by_name)}") from None

    def names(self) -> list[str]:
        return list(self._by_name)

    def all(self) -> list[Tool]:
        return list(self._by_name.values())

    def anthropic_specs(self) -> list[dict]:
        return [t.to_anthropic() for t in self._by_name.values()]

    def mcp_specs(self) -> list[dict]:
        return [t.to_mcp() for t in self._by_name.values()]
