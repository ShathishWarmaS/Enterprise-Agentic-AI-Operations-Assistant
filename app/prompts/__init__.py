"""System prompts for each agent. Kept as plain module constants so they are
easy to diff and review. Only used when LLM_MODE=claude.
"""

from app.prompts.text import (
    ACTION_SYSTEM,
    DATA_ANALYSIS_SYSTEM,
    PLANNER_SYSTEM,
    QUERY_ANSWER_SYSTEM,
    VALIDATION_SYSTEM,
)

__all__ = [
    "ACTION_SYSTEM",
    "DATA_ANALYSIS_SYSTEM",
    "PLANNER_SYSTEM",
    "QUERY_ANSWER_SYSTEM",
    "VALIDATION_SYSTEM",
]
