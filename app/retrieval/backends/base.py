"""The contract every vector backend implements."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.schemas.documents import Chunk
from app.schemas.retrieval import RetrievedChunk


class VectorBackend(ABC):
    """Persistent similarity index over ingested chunks."""

    @property
    @abstractmethod
    def backend(self) -> str: ...

    @abstractmethod
    def add(self, chunks: list[Chunk]) -> int: ...

    @abstractmethod
    def search(self, query: str, top_k: int) -> list[RetrievedChunk]: ...

    @abstractmethod
    def delete_document(self, document_id: str) -> int: ...

    @abstractmethod
    def clear(self) -> None: ...

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def stats(self) -> dict: ...
