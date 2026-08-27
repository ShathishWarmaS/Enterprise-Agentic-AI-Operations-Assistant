"""A tiny background job queue over a thread pool.

Used by ``POST /documents/ingest`` when async ingestion is requested. Jobs are
persisted to the ``jobs`` table so their status survives the request that
created them; poll ``GET /jobs/{job_id}``.

The synchronous ingest path (seed script, eval harness) does not touch this.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from app.config import Settings
from app.data import repository
from app.data.database import session_scope
from app.data.models import JobRow

logger = logging.getLogger(__name__)


class JobQueue:
    """Wraps a ``ThreadPoolExecutor``; the pool is created on first ``submit``."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pool: ThreadPoolExecutor | None = None

    def _executor(self) -> ThreadPoolExecutor:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(
                max_workers=self._settings.ingest_workers,
                thread_name_prefix="jobq",
            )
        return self._pool

    def submit(self, kind: str, payload: dict, fn: Callable[[], Any]) -> str:
        job_id = "job_" + uuid.uuid4().hex[:16]
        with session_scope() as session:
            repository.create_job(session, job_id=job_id, kind=kind, payload=payload)
        self._executor().submit(self._run, job_id, fn)
        return job_id

    def _run(self, job_id: str, fn: Callable[[], Any]) -> None:
        with session_scope() as session:
            repository.update_job_status(session, job_id, "running")
        try:
            value = fn()
        except Exception as exc:  # noqa: BLE001 - recorded on the row, re-logged with trace
            logger.exception("job %s failed", job_id)
            with session_scope() as session:
                repository.update_job_status(session, job_id, "failed", error=str(exc))
            return
        result = value if isinstance(value, dict) else {"value": value}
        with session_scope() as session:
            repository.update_job_status(session, job_id, "succeeded", result=result)

    def get(self, job_id: str) -> JobRow | None:
        with session_scope() as session:
            row = repository.get_job(session, job_id)
            if row is not None:
                session.expunge(row)
            return row

    def shutdown(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
