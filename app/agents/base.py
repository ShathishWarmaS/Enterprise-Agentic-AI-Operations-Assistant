"""Shared base for agents: holds the LLM client and a small retry helper."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TypeVar

from app.config import Settings
from app.services.llm import LLMClient, LLMError, LLMOutputError

logger = logging.getLogger(__name__)

R = TypeVar("R")


class Agent:
    name: str

    def __init__(self, settings: Settings, llm: LLMClient) -> None:
        self.settings = settings
        self.llm = llm

    @property
    def uses_claude(self) -> bool:
        return self.llm.enabled

    def with_retry(
        self,
        fn: Callable[[], R],
        *,
        attempts: int = 2,
        on_error: str = "operation",
    ) -> tuple[R | None, int, str | None]:
        """Run `fn`, retrying on LLM output/transport errors.

        Returns (result_or_none, retries_used, error_message_or_none). Only the
        error types we know how to recover from are retried; anything else
        propagates so we don't mask real bugs.
        """
        last_error: str | None = None
        for attempt in range(attempts):
            try:
                return fn(), attempt, None
            except (LLMOutputError, LLMError) as exc:
                last_error = f"{on_error} failed: {exc}"
                logger.warning("%s (attempt %d/%d)", last_error, attempt + 1, attempts)
                time.sleep(0.2 * (attempt + 1))
        return None, attempts - 1, last_error
