"""HTTP API: happy path end to end, plus input-validation and 404s."""

from __future__ import annotations

import io
import json
import time


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["llm_mode"] == "mock"
    assert body["chunks"] > 0


def test_upload_ingest_query_flow(client):
    csv = b"host,cpu\nweb-1,0.9\nweb-2,0.4\n"
    up = client.post(
        "/documents/upload",
        files={"file": ("hosts.csv", io.BytesIO(csv), "text/csv")},
    )
    assert up.status_code == 200
    doc_id = up.json()["document"]["document_id"]

    ing = client.post("/documents/ingest", json={"document_id": doc_id})
    assert ing.status_code == 200
    assert ing.json()["results"][0]["table_registered"] is True

    q = client.post("/query", json={"query": "which host has the highest cpu?"})
    assert q.status_code == 200
    assert "session_id" in q.json()


def test_upload_rejects_unsupported_type(client):
    r = client.post(
        "/documents/upload",
        files={"file": ("notes.docx", io.BytesIO(b"junk"), "application/octet-stream")},
    )
    assert r.status_code == 422


def test_ingest_requires_a_target(client):
    assert client.post("/documents/ingest", json={}).status_code == 400


def test_ingest_unknown_document(client):
    r = client.post("/documents/ingest", json={"document_id": "deadbeef"})
    assert r.status_code == 422


def test_query_validation(client):
    assert client.post("/query", json={"query": ""}).status_code == 422


def test_agent_run_and_session_fetch(client):
    r = client.post(
        "/agent/run",
        json={"request": "payments gateway returning 401s that become 5xx - what do we do?"},
    )
    assert r.status_code == 200
    payload = r.json()
    assert payload["result"]["validation"]["passed"] in (True, False)
    sid = payload["session_id"]

    got = client.get(f"/sessions/{sid}")
    assert got.status_code == 200
    assert got.json()["kind"] == "agent"


def test_unknown_session_is_404(client):
    assert client.get("/sessions/nope").status_code == 404


def test_async_ingest_returns_202_and_job_reaches_succeeded(client):
    csv = b"svc,rps\napi,120\nweb,80\n"
    up = client.post(
        "/documents/upload",
        files={"file": ("async_svc.csv", io.BytesIO(csv), "text/csv")},
    )
    doc_id = up.json()["document"]["document_id"]

    ing = client.post("/documents/ingest", json={"document_id": doc_id, "async": True})
    assert ing.status_code == 202
    jobs = ing.json()["jobs"]
    assert jobs and jobs[0]["document_id"] == doc_id
    job_id = jobs[0]["job_id"]

    for _ in range(200):
        body = client.get(f"/jobs/{job_id}").json()
        if body["status"] in {"succeeded", "failed"}:
            break
        time.sleep(0.05)
    assert body["status"] == "succeeded"
    assert body["result"]["chunks_created"] >= 1


def test_unknown_job_is_404(client):
    assert client.get("/jobs/job_deadbeef").status_code == 404


def test_agent_run_stream_emits_steps_then_result(client):
    from app.schemas.agents import AgentRunResult

    events = []
    with client.stream(
        "POST",
        "/agent/run/stream",
        json={"request": "payment-service is throwing 5xx after a deploy; next steps?"},
    ) as resp:
        assert resp.status_code == 200
        event = None
        for line in resp.iter_lines():
            if line.startswith("event: "):
                event = line[len("event: ") :]
            elif line.startswith("data: "):
                events.append((event, json.loads(line[len("data: ") :])))

    kinds = [e[0] for e in events]
    assert kinds.count("step") >= 3
    assert kinds[-1] == "result"
    AgentRunResult.model_validate(events[-1][1])


def test_evaluate_endpoint(client):
    body = client.post("/evaluate", json={}).json()
    assert body["cases"] >= 5
    for key in (
        "retrieval_relevance",
        "citation_presence",
        "groundedness",
        "pass_rate",
    ):
        assert 0.0 <= body[key] <= 1.0
