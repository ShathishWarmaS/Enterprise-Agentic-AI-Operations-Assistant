"""Process-wide singletons, built once at startup.

Keeping construction in one place makes the dependency graph obvious and gives
tests a single seam to override (see tests/conftest.py).
"""

from __future__ import annotations

from functools import lru_cache

from app.config import Settings, get_settings
from app.data.tables import TableStore
from app.retrieval.embeddings import build_embedder
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import build_vector_store
from app.services.ingestion_service import IngestionService
from app.services.jobs import JobQueue
from app.services.llm import LLMClient
from app.services.session_state import SessionStore
from app.tools.registry import ToolRegistry


class Container:
    def __init__(self, settings: Settings) -> None:
        settings.ensure_dirs()
        self.settings = settings
        self.llm = LLMClient(settings)
        self.embedder = build_embedder(settings)
        self.vector_store = build_vector_store(settings, self.embedder)
        self.table_store = TableStore(settings.vector_store_dir)
        self.retriever = Retriever(
            self.vector_store,
            top_k=settings.retrieval_top_k,
            min_score=settings.retrieval_min_score,
        )
        self.tools = ToolRegistry(retriever=self.retriever, table_store=self.table_store)
        self.session_store = SessionStore(settings.redis_url)
        self.job_queue = JobQueue(settings)
        self.ingestion = IngestionService(
            settings=settings,
            vector_store=self.vector_store,
            table_store=self.table_store,
        )

    def orchestrator(self):
        # imported lazily to avoid a cycle (agents import schemas, not services)
        from app.agents.orchestrator import Orchestrator

        return Orchestrator(
            settings=self.settings,
            llm=self.llm,
            retriever=self.retriever,
            table_store=self.table_store,
            tools=self.tools,
        )


@lru_cache
def get_container() -> Container:
    return Container(get_settings())
