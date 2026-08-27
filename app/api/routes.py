"""HTTP routes. Handlers stay thin: validate, delegate to a service, shape the
response. Domain errors map to 4xx; anything unexpected propagates to the
500 handler in main.py.
"""

from __future__ import annotations

import functools
import json
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from app.api.schemas import (
    AgentRunRequest,
    AgentRunResponse,
    EvaluateRequest,
    HealthResponse,
    IngestAcceptedResponse,
    IngestJobRef,
    IngestRequest,
    IngestResponse,
    JobResponse,
    QueryRequest,
    QueryResponse,
    SessionResponse,
    UploadResponse,
)
from app.data import repository
from app.data.database import session_scope
from app.schemas.evaluation import EvalSummary
from app.services.container import Container, get_container
from app.services.evaluation import load_cases, run_evaluation
from app.services.ingestion_service import IngestionError

router = APIRouter()

MAX_UPLOAD_BYTES = 20 * 1024 * 1024


def container_dep() -> Container:
    return get_container()


@router.get("/health", response_model=HealthResponse, tags=["meta"])
def health(container: Container = Depends(container_dep)) -> HealthResponse:
    stats = container.vector_store.stats()
    return HealthResponse(
        llm_mode=container.llm.mode,
        vector_backend=stats["backend"],
        session_backend=container.session_store.backend_name,
        documents=stats["documents"],
        chunks=stats["chunks"],
        tables=container.table_store.list_tables(),
    )


@router.post("/documents/upload", response_model=UploadResponse, tags=["documents"])
async def upload_document(
    file: UploadFile = File(...),
    container: Container = Depends(container_dep),
) -> UploadResponse:
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"file exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit")
    try:
        doc = container.ingestion.save_upload(filename=file.filename or "upload", content=content)
    except IngestionError as exc:
        raise HTTPException(422, str(exc)) from exc
    return UploadResponse(document=doc)


@router.post(
    "/documents/ingest",
    response_model=IngestResponse | IngestAcceptedResponse,
    tags=["documents"],
)
def ingest_documents(
    body: IngestRequest,
    container: Container = Depends(container_dep),
):
    if not body.ingest_all and not body.document_id:
        raise HTTPException(400, "provide document_id or set ingest_all=true")

    with session_scope() as session:
        if body.ingest_all:
            ids = [row.id for row in repository.list_documents(session) if row.status != "ingested"]
        else:
            ids = [body.document_id]  # type: ignore[list-item]

    if not ids:
        raise HTTPException(404, "no matching documents to ingest")

    if container.settings.ingest_async or body.async_:
        jobs: list[IngestJobRef] = []
        for document_id in ids:
            job_id = container.job_queue.submit(
                "ingest",
                {"document_id": document_id},
                functools.partial(_run_ingest_job, container, document_id),
            )
            jobs.append(IngestJobRef(document_id=document_id, job_id=job_id))
        return JSONResponse(
            status_code=202,
            content=IngestAcceptedResponse(jobs=jobs).model_dump(),
        )

    results = []
    for document_id in ids:
        try:
            results.append(container.ingestion.ingest(document_id))
        except IngestionError as exc:
            raise HTTPException(422, str(exc)) from exc
    return IngestResponse(results=results)


@router.get("/jobs/{job_id}", response_model=JobResponse, tags=["documents"])
def get_job(job_id: str, container: Container = Depends(container_dep)) -> JobResponse:
    row = container.job_queue.get(job_id)
    if row is None:
        raise HTTPException(404, f"no job {job_id!r}")
    return JobResponse(
        job_id=row.id,
        kind=row.kind,
        status=row.status,
        result=row.result,
        error=row.error,
        created_at=row.created_at.isoformat() if row.created_at else "",
        updated_at=row.updated_at.isoformat() if row.updated_at else "",
    )


@router.post("/query", response_model=QueryResponse, tags=["rag"])
def query(
    body: QueryRequest,
    container: Container = Depends(container_dep),
) -> QueryResponse:
    orchestrator = container.orchestrator()
    try:
        answer, _ = orchestrator.retrieval.answer_query(body.query, top_k=body.top_k)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    session_id = f"q_{uuid.uuid4().hex[:12]}"
    _persist_session(container, session_id, "query", body.query, answer.model_dump(mode="json"))
    return QueryResponse(session_id=session_id, answer=answer)


@router.post("/agent/run", response_model=AgentRunResponse, tags=["agents"])
def agent_run(
    body: AgentRunRequest,
    container: Container = Depends(container_dep),
) -> AgentRunResponse:
    orchestrator = container.orchestrator()
    session_id = f"a_{uuid.uuid4().hex[:12]}"
    try:
        result = orchestrator.run(body.request, session_id=session_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    _persist_session(container, session_id, "agent", body.request, result.model_dump(mode="json"))
    return AgentRunResponse(session_id=session_id, result=result)


@router.post("/agent/run/stream", tags=["agents"])
def agent_run_stream(
    body: AgentRunRequest,
    container: Container = Depends(container_dep),
) -> StreamingResponse:
    orchestrator = container.orchestrator()
    session_id = f"a_{uuid.uuid4().hex[:12]}"

    def event_stream():
        result_payload: dict | None = None
        try:
            for kind, payload in orchestrator.run_streaming(body.request, session_id=session_id):
                if kind == "result":
                    result_payload = payload
                yield f"event: {kind}\ndata: {json.dumps(payload)}\n\n"
        except ValueError as exc:
            yield f"event: error\ndata: {json.dumps({'detail': str(exc)})}\n\n"
            return
        if result_payload is not None:
            _persist_session(container, session_id, "agent", body.request, result_payload)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/evaluate", response_model=EvalSummary, tags=["eval"])
def evaluate(
    body: EvaluateRequest,
    container: Container = Depends(container_dep),
) -> EvalSummary:
    path = Path(body.cases_path) if body.cases_path else None
    try:
        cases = load_cases(path) if path else load_cases()
    except FileNotFoundError as exc:
        raise HTTPException(404, f"eval cases not found: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    summary = run_evaluation(container, cases)
    session_id = f"e_{uuid.uuid4().hex[:12]}"
    _persist_session(container, session_id, "evaluate", "eval run", summary.model_dump(mode="json"))
    return summary


@router.get("/sessions/{session_id}", response_model=SessionResponse, tags=["sessions"])
def get_session(session_id: str, container: Container = Depends(container_dep)) -> SessionResponse:
    with session_scope() as session:
        row = repository.get_session_record(session, session_id)
        if row is None:
            cached = container.session_store.fetch(session_id)
            if cached is None:
                raise HTTPException(404, f"no session {session_id!r}")
            return SessionResponse(**cached)
        return SessionResponse(
            session_id=row.id,
            kind=row.kind,
            request=row.request,
            llm_mode=row.llm_mode,
            created_at=row.created_at.isoformat() if row.created_at else "",
            response=row.response,
        )


def _run_ingest_job(container: Container, document_id: str) -> dict:
    return container.ingestion.ingest(document_id).model_dump(mode="json")


def _persist_session(
    container: Container, session_id: str, kind: str, request: str, response: dict
) -> None:
    payload = {
        "session_id": session_id,
        "kind": kind,
        "request": request,
        "llm_mode": container.llm.mode,
        "response": response,
    }
    with session_scope() as session:
        repository.save_session(
            session,
            session_id=session_id,
            kind=kind,
            request=request,
            llm_mode=container.llm.mode,
            response=response,
        )
    container.session_store.put(session_id, {**payload, "created_at": ""})
