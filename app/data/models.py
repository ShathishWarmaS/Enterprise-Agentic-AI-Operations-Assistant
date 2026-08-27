"""ORM models. Deliberately few tables: documents, sessions, and runs."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.data.database import Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


class DocumentRow(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    filename: Mapped[str] = mapped_column(String(512))
    source_type: Mapped[str] = mapped_column(String(16))
    size_bytes: Mapped[int] = mapped_column(Integer)
    stored_path: Mapped[str] = mapped_column(String(1024))
    status: Mapped[str] = mapped_column(String(16), default="uploaded")
    table_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    chunks: Mapped[int] = mapped_column(Integer, default=0)
    cleaning_report: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class SessionRow(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    kind: Mapped[str] = mapped_column(String(16))  # query | agent | evaluate
    request: Mapped[str] = mapped_column(Text)
    llm_mode: Mapped[str] = mapped_column(String(16))
    response: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
