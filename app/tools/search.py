"""`search_documents` - vector search over ingested documents."""

from __future__ import annotations

from app.retrieval.retriever import Retriever
from app.tools.base import Tool, ToolError


class SearchDocumentsTool(Tool):
    name = "search_documents"
    description = (
        "Semantic search over ingested documents (runbooks, postmortems, notes). "
        "Returns the most relevant text chunks with a source filename, a locator "
        "(page or row range), and a similarity score in [0,1]."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "natural-language search query"},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, retriever: Retriever) -> None:
        self._retriever = retriever

    def run(self, arguments: dict) -> dict:
        query = str(arguments.get("query", "")).strip()
        if not query:
            raise ToolError("query is required and must be non-empty")
        top_k = int(arguments.get("top_k", 5))

        result = self._retriever.retrieve(query, top_k=top_k)
        return {
            "query": result.query,
            "confident": result.confident,
            "top_score": result.top_score,
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "filename": c.filename,
                    "locator": c.locator,
                    "score": c.score,
                    "text": c.text,
                }
                for c in result.chunks
            ],
        }
