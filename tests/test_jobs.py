"""Background job queue: success and failure paths."""

from __future__ import annotations

import time


def _wait(job_queue, job_id, *, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        row = job_queue.get(job_id)
        if row is not None and row.status in {"succeeded", "failed"}:
            return row
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


def test_job_succeeds_running_real_ingest(seeded_container):
    csv = b"host,cpu\nweb-1,0.9\nweb-2,0.4\nweb-3,0.7\n"
    doc = seeded_container.ingestion.save_upload(filename="jobq_hosts.csv", content=csv)

    job_id = seeded_container.job_queue.submit(
        "ingest",
        {"document_id": doc.document_id},
        lambda: seeded_container.ingestion.ingest(doc.document_id).model_dump(mode="json"),
    )
    row = _wait(seeded_container.job_queue, job_id)

    assert row.status == "succeeded"
    assert row.error is None
    assert row.result["chunks_created"] >= 1


def test_job_fails_and_records_error(seeded_container):
    def boom():
        raise RuntimeError("kaboom")

    job_id = seeded_container.job_queue.submit("ingest", {}, boom)
    row = _wait(seeded_container.job_queue, job_id)

    assert row.status == "failed"
    assert row.result is None
    assert "kaboom" in row.error
