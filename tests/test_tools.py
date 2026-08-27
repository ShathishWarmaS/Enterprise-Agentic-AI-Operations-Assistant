"""MCP tools: schema exposure and graceful tool-call failures."""

from __future__ import annotations

from app.data.tables import TableStore
from app.tools.data_tools import QueryDataTool
from app.tools.incident_tools import DraftIncidentSummaryTool, GenerateChecklistTool


def test_tool_specs_are_mcp_and_anthropic_shaped(seeded_container):
    for tool in seeded_container.tools.all():
        anth = tool.to_anthropic()
        mcp = tool.to_mcp()
        assert anth["input_schema"]["type"] == "object"
        assert mcp["inputSchema"] == anth["input_schema"]
        assert mcp["name"] == anth["name"]


def test_query_data_unknown_table_fails_cleanly(tmp_dir):
    tool = QueryDataTool(TableStore(tmp_dir))
    call = tool.invoke({"table": "does_not_exist"})
    assert call.ok is False
    assert "no table" in call.error


def test_query_data_unknown_column_fails_cleanly(seeded_container):
    tool = seeded_container.tools.get("query_data")
    call = tool.invoke({"table": "incidents", "columns": ["not_a_column"]})
    assert call.ok is False
    assert "unknown column" in call.error


def test_search_documents_runs(seeded_container):
    call = seeded_container.tools.get("search_documents").invoke(
        {"query": "rollback payment-service", "top_k": 3}
    )
    assert call.ok and call.result["chunks"]


def test_generate_checklist_matches_playbook():
    call = GenerateChecklistTool().invoke(
        {"scenario": "database connection pool exhausted after deploy", "severity": "high"}
    )
    assert call.ok
    actions = " ".join(i["action"].lower() for i in call.result["items"])
    assert "roll back" in actions or "rollback" in actions
    assert "connection pool" in actions
    assert call.result["items"][-1]["blocking"] is False  # postmortem step, non-blocking


def test_draft_incident_summary_requires_observations():
    call = DraftIncidentSummaryTool().invoke(
        {
            "title": "t",
            "severity": "high",
            "impact": "i",
            "likely_cause": "c",
            "observations": [],
        }
    )
    assert call.ok is False


def test_generate_checklist_rejects_bad_severity():
    call = GenerateChecklistTool().invoke({"scenario": "x", "severity": "spicy"})
    assert call.ok is False
