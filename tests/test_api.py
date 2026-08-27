"""HTTP API: happy path end to end, plus input-validation and 404s."""

from __future__ import annotations

import io


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
