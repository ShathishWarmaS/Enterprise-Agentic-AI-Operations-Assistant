"""Incident-response tools: `draft_incident_summary`, `generate_checklist`.

These are deterministic templating tools, not LLM calls. They turn observations
and evidence the agents have already gathered into a consistently structured
summary and an ordered remediation checklist. The checklist rules are a small
keyword-driven playbook, documented inline.
"""

from __future__ import annotations

from app.schemas.agents import ChecklistItem, Citation, IncidentSummary
from app.tools.base import Tool, ToolError

_SEVERITIES = ("low", "medium", "high", "critical")

# scenario keyword -> (extra checklist actions, owner role)
_PLAYBOOK: list[tuple[tuple[str, ...], list[str], str]] = [
    (
        ("rollback", "deploy", "release", "version"),
        [
            "Roll back to the last known-good release",
            "Freeze further deploys until root cause is confirmed",
        ],
        "release-engineer",
    ),
    (
        ("database", "db", "query", "replica", "connection pool"),
        [
            "Check database connection pool saturation and slow-query log",
            "Fail over to a healthy replica if primary is degraded",
        ],
        "database-admin",
    ),
    (
        ("memory", "oom", "leak", "cpu", "throttl"),
        [
            "Capture a heap/CPU profile from an affected instance before restart",
            "Increase resource limits or replica count as a temporary mitigation",
        ],
        "service-owner",
    ),
    (
        ("credential", "secret", "token", "auth", "expired"),
        [
            "Rotate the affected credential and invalidate old sessions",
            "Audit access logs for use of the compromised credential",
        ],
        "security-engineer",
    ),
    (
        ("cache", "redis", "stale"),
        [
            "Flush or warm the affected cache keys",
            "Verify cache TTLs match the data's freshness needs",
        ],
        "service-owner",
    ),
    (
        ("latency", "timeout", "slow", "p99"),
        [
            "Compare upstream/downstream latency to isolate the slow hop",
            "Shed non-critical load (feature flags, rate limits)",
        ],
        "on-call-engineer",
    ),
]


class DraftIncidentSummaryTool(Tool):
    name = "draft_incident_summary"
    description = (
        "Assemble a structured incident summary from observations and evidence "
        "already collected. Does not invent facts - it only formats what it is given."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "severity": {"type": "string", "enum": list(_SEVERITIES)},
            "impact": {"type": "string"},
            "likely_cause": {"type": "string"},
            "observations": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "evidence": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "marker": {"type": "string"},
                        "filename": {"type": "string"},
                        "locator": {"type": "string"},
                        "chunk_id": {"type": "string"},
                    },
                    "required": ["filename", "locator", "chunk_id"],
                },
            },
        },
        "required": ["title", "severity", "impact", "likely_cause", "observations"],
        "additionalProperties": False,
    }

    def run(self, arguments: dict) -> dict:
        severity = arguments["severity"]
        if severity not in _SEVERITIES:
            raise ToolError(f"severity must be one of {_SEVERITIES}")
        observations = [o.strip() for o in arguments["observations"] if o.strip()]
        if not observations:
            raise ToolError("at least one non-empty observation is required")

        evidence = [
            Citation(
                marker=e.get("marker") or f"[{i}]",
                filename=e["filename"],
                locator=e["locator"],
                chunk_id=e["chunk_id"],
            )
            for i, e in enumerate(arguments.get("evidence", []), start=1)
        ]
        summary_text = " ".join(observations)
        incident = IncidentSummary(
            title=arguments["title"].strip(),
            severity=severity,
            summary=summary_text,
            impact=arguments["impact"].strip(),
            likely_cause=arguments["likely_cause"].strip(),
            evidence=evidence,
        )
        return incident.model_dump()


class GenerateChecklistTool(Tool):
    name = "generate_checklist"
    description = (
        "Produce an ordered remediation / deployment checklist for a scenario. "
        "Combines a standard incident-response spine with scenario-specific steps "
        "chosen from a keyword playbook."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "scenario": {"type": "string", "description": "short description of the situation"},
            "severity": {"type": "string", "enum": list(_SEVERITIES)},
        },
        "required": ["scenario", "severity"],
        "additionalProperties": False,
    }

    def run(self, arguments: dict) -> dict:
        scenario = arguments["scenario"].strip().lower()
        severity = arguments["severity"]
        if not scenario:
            raise ToolError("scenario must not be empty")
        if severity not in _SEVERITIES:
            raise ToolError(f"severity must be one of {_SEVERITIES}")

        items: list[ChecklistItem] = []
        order = 1

        def add(action: str, owner: str, blocking: bool = False) -> None:
            nonlocal order
            items.append(
                ChecklistItem(order=order, action=action, owner_role=owner, blocking=blocking)
            )
            order += 1

        high = severity in ("high", "critical")
        add(
            "Declare an incident and open a coordination channel",
            "incident-commander",
            blocking=high,
        )
        add("Post an initial status update to stakeholders", "communications-lead")

        matched = False
        for keywords, actions, owner in _PLAYBOOK:
            if any(k in scenario for k in keywords):
                matched = True
                for action in actions:
                    add(action, owner, blocking=high)
        if not matched:
            add(
                "Identify the smallest change that could have triggered the issue",
                "on-call-engineer",
            )
            add("Apply the lowest-risk mitigation that restores service", "on-call-engineer")

        add(
            "Verify recovery against monitoring and a user-facing check",
            "on-call-engineer",
            blocking=True,
        )
        add("Write a timeline and schedule a blameless postmortem", "service-owner")
        return {
            "scenario": arguments["scenario"],
            "severity": severity,
            "items": [i.model_dump() for i in items],
        }
