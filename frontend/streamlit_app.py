"""Streamlit UI for the Operations Assistant.

Talks only to the FastAPI service over HTTP (API_URL, default localhost:8000).
Run:  streamlit run frontend/streamlit_app.py
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
client = httpx.Client(base_url=API_URL, timeout=120.0)

st.set_page_config(page_title="Ops Assistant", page_icon="🛠️", layout="wide")
st.title("Enterprise Agentic AI — Operations Assistant")


def _get(path: str):
    return client.get(path).raise_for_status().json()


def _post(path: str, **kwargs):
    resp = client.post(path, **kwargs)
    if resp.status_code >= 400:
        st.error(f"{resp.status_code}: {resp.json().get('detail', resp.text)}")
        return None
    return resp.json()


# -- sidebar: health + ingestion ------------------------------------
with st.sidebar:
    st.header("Corpus")
    try:
        health = _get("/health")
        st.caption(
            f"llm: **{health['llm_mode']}** · vectors: {health['chunks']} chunks / "
            f"{health['documents']} docs ({health['vector_backend']})"
        )
        if health["tables"]:
            st.caption("tables: " + ", ".join(health["tables"]))
    except Exception as exc:  # noqa: BLE001
        st.error(f"API unreachable at {API_URL}: {exc}")
        st.stop()

    uploaded = st.file_uploader("Add a document", type=["pdf", "csv", "json", "md", "txt"])
    if uploaded and st.button("Upload & ingest", use_container_width=True):
        up = _post(
            "/documents/upload",
            files={"file": (uploaded.name, uploaded.getvalue())},
        )
        if up:
            doc_id = up["document"]["document_id"]
            res = _post("/documents/ingest", json={"document_id": doc_id})
            if res:
                r = res["results"][0]
                st.success(f"{r['filename']}: {r['chunks_created']} chunks")
                st.rerun()

    if st.button("Re-ingest sample corpus", use_container_width=True):
        res = _post("/documents/ingest", json={"ingest_all": True})
        if res is not None:
            st.success(f"ingested {len(res['results'])} document(s)")
            st.rerun()


ask_tab, investigate_tab, eval_tab = st.tabs(["Ask", "Investigate", "Evaluate"])

with ask_tab:
    st.subheader("Grounded Q&A over the corpus")
    q = st.text_input("Question", "How do we roll back payment-service?")
    if st.button("Ask", type="primary"):
        data = _post("/query", json={"query": q})
        if data:
            ans = data["answer"]
            st.markdown(f"**Answer** ({'confident' if ans['confident'] else 'low confidence'})")
            st.write(ans["answer"])
            for note in ans["notes"]:
                st.caption(f"note: {note}")
            if ans["citations"]:
                st.markdown("**Sources**")
                for c in ans["citations"]:
                    st.markdown(f"- {c['marker']} `{c['filename']}` — {c['locator']}")

with investigate_tab:
    st.subheader("Multi-agent investigation → operational decision")
    req = st.text_area(
        "Describe the situation",
        "payment-service is returning elevated 5xx errors after a deploy 10 minutes ago. "
        "Give me an incident summary, next steps, and a remediation checklist.",
    )
    if st.button("Run agents", type="primary"):
        data = _post("/agent/run", json={"request": req})
        if data:
            result = data["result"]
            v = result["validation"]
            (st.success if v["passed"] else st.warning)(
                f"validation: passed={v['passed']} · grounded={v['grounded']} · "
                f"{v['supported_claims']}/{v['checked_claims']} claims supported"
            )
            with st.expander("Plan"):
                for s in result["plan"]["steps"]:
                    st.markdown(
                        f"{s['step']}. **{s['agent']}** — {s['objective']}"
                        + (f"  ·  `{s['tool']}`" if s["tool"] else "")
                    )
            dec = result["decision"]
            if dec:
                inc = dec["incident"]
                st.markdown(f"### {inc['title']}")
                st.markdown(
                    f"**Severity:** {inc['severity']}  ·  " f"**Confidence:** {dec['confidence']}"
                )
                st.markdown(f"**Summary.** {inc['summary']}")
                st.markdown(f"**Likely cause.** {inc['likely_cause']}")
                st.markdown(f"**Impact.** {inc['impact']}")
                st.markdown("**Recommended next steps**")
                for step in dec["recommended_next_steps"]:
                    st.markdown(f"- {step}")
                st.markdown("**Remediation checklist**")
                for item in dec["remediation_checklist"]:
                    flag = " ⛔" if item["blocking"] else ""
                    st.markdown(f"{item['order']}. {item['action']} — *{item['owner_role']}*{flag}")
                if dec["open_questions"]:
                    st.markdown("**Open questions**")
                    for oq in dec["open_questions"]:
                        st.markdown(f"- {oq}")
                if dec["citations"]:
                    st.markdown("**Citations**")
                    for c in dec["citations"]:
                        st.markdown(f"- {c['marker']} `{c['filename']}` — {c['locator']}")
            for issue in v["issues"]:
                st.caption(f"⚠ {issue['field']}: {issue['detail']}")

with eval_tab:
    st.subheader("Offline evaluation on the sample dataset")
    if st.button("Run evaluation", type="primary"):
        summary = _post("/evaluate", json={})
        if summary:
            cols = st.columns(4)
            metric_items = [
                ("Retrieval", summary["retrieval_relevance"]),
                ("Citations", summary["citation_presence"]),
                ("Tool choice", summary["tool_selection_accuracy"]),
                ("Structured", summary["structured_output_validity"]),
                ("Grounded", summary["groundedness"]),
                ("Missing-info", summary["missing_info_handling"]),
                ("Consistency", summary["response_consistency"]),
                ("Pass rate", summary["pass_rate"]),
            ]
            for i, (name, value) in enumerate(metric_items):
                cols[i % 4].metric(name, f"{value:.2f}")
            st.dataframe(summary["results"], use_container_width=True)
