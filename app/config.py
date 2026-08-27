"""Application configuration, loaded once from the environment.

All tunable behaviour lives here so the rest of the code never reads os.environ
directly. Import `settings` (a module-level singleton) or call `get_settings()`
in FastAPI dependencies.
"""

from __future__ import annotations

import logging
from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMMode(str, Enum):
    mock = "mock"
    claude = "claude"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    llm_mode: LLMMode = LLMMode.mock
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-4-20250514"
    anthropic_max_tokens: int = Field(default=1024, ge=64, le=8192)

    # Storage
    database_url: str = "sqlite:///./storage/app.sqlite3"
    vector_store_dir: Path = Path("./storage/vector")
    upload_dir: Path = Path("./storage/uploads")

    # Redis (optional)
    redis_url: str | None = None

    # Ingestion
    # When true, PDF pages with no embedded text are OCR'd with Tesseract
    # (requires the `ocr` extra and the tesseract binary). Off by default.
    pdf_ocr_fallback: bool = False

    # Retrieval
    chunk_size: int = Field(default=800, ge=200, le=4000)
    chunk_overlap: int = Field(default=120, ge=0, le=1000)
    retrieval_top_k: int = Field(default=5, ge=1, le=50)
    retrieval_min_score: float = Field(default=0.22, ge=0.0, le=1.0)

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    @model_validator(mode="after")
    def _check_claude_config(self) -> Settings:
        if self.llm_mode is LLMMode.claude and not self.anthropic_api_key:
            raise ValueError(
                "LLM_MODE=claude requires ANTHROPIC_API_KEY to be set. "
                "Use LLM_MODE=mock for offline development."
            )
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self

    def ensure_dirs(self) -> None:
        for path in (self.vector_store_dir, self.upload_dir):
            path.mkdir(parents=True, exist_ok=True)
        db = self.database_url
        if db.startswith("sqlite:///") and "./" in db:
            Path(db.replace("sqlite:///", "")).parent.mkdir(parents=True, exist_ok=True)

    def configure_logging(self) -> None:
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper(), logging.INFO),
            format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
