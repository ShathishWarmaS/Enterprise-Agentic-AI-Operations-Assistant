# Résumé bullets

Truthful, verifiable-by-reading-the-repo bullets for the Enterprise Agentic AI
Operations Assistant (personal portfolio project). Pick the 4–6 that fit the
role.

---

- Built a multi-agent RAG system (FastAPI + Streamlit, Python 3.11) in which
  five specialised agents — Planner, Retrieval, Data Analysis, Action,
  Validation — are coordinated by an orchestrator that owns control flow and
  degrades a run per-step (records the error, sets a `degraded` flag, continues)
  instead of aborting on partial failure.

- Implemented an ingestion pipeline for messy enterprise data — PDF (PyMuPDF),
  CSV/TSV including ragged-row recovery, JSON objects and record arrays,
  Markdown and plain text — with a cleaning stage that normalises column names,
  gates type coercion on parse-success thresholds (≥90% datetime, ≥70%
  numeric), detects nulls, duplicates, outliers (>4σ) and sentinel values, and
  emits a structured `CleaningReport`.

- Wrote a deterministic offline embedding (512-dim signed-hashed word n-grams +
  character trigrams, sublinear TF weighting) behind a small `VectorStore`
  interface backed by FAISS `IndexFlatIP` with a pure-numpy fallback, so
  retrieval and the evaluation harness run with no network access or model
  downloads.

- Designed six MCP-compatible tools (`search_documents`, `query_data`,
  `compute_metrics`, `check_schema`, `draft_incident_summary`,
  `generate_checklist`) behind one `ToolRegistry` that backs both the Action
  agent and an optional stdio MCP server, with each tool defined as plain JSON
  Schema so the same definition serves Anthropic tool-use and MCP; tool calls
  are executed via `Tool.invoke()` which returns timed `ToolCall` trace records
  and never raises.

- Enforced grounding as a hard gate: a Validation agent checks every factual
  claim in the generated decision against retrieved context using a
  conservative content-word / trigram overlap score (tuned to reject
  weak-but-true rather than accept fabricated), requires all output fields, and
  rejects runs where more than a third of claims are unsupported — with a
  structured `OperationalDecision` (incident summary + next steps + remediation
  checklist) carrying inline citations back to source files and locators.

- Made the system run fully offline in a deterministic `mock` mode (rule-based
  agents, no API key) and switch to `claude` mode via one environment variable;
  in `claude` mode Claude refines the plan, answer and decision as validated
  structured JSON while citations, the checklist and the final pass/fail verdict
  stay deterministic (Claude can add validation concerns, not remove them).

- Built a 7-metric evaluation harness (retrieval recall@k, citation presence,
  tool-selection accuracy, structured-output validity, groundedness,
  missing-info handling, response consistency) that computes every score from
  real runs over a labelled case set — no hard-coded numbers — and a
  `run_eval.py --min-pass` CLI that exits non-zero to gate CI on pass rate.

- Hardened the data-query path against injection: the Data Analysis agent and
  `query_data` tool never execute SQL or `df.query` strings — they take a
  Pydantic-validated `QuerySpec` with column names checked against the live
  schema, a fixed operator/aggregation allow-list, and a capped row limit.

- Containerised the stack with a non-root Dockerfile and a Docker Compose setup
  (API, Streamlit UI, Postgres, Redis, seed job) and a GitHub Actions CI
  pipeline running ruff, black, mypy, an import check, pytest with coverage, and
  the seeded evaluation gate at a 70% pass-rate threshold.

- Kept a clean dependency direction (agents depend on schemas and injected
  collaborators, never on services or API) with a single DI container as the
  one construction/override seam, plus a pytest suite covering ingestion,
  retrieval, tools, the LLM layer, agents, API and evaluation.

---

## How to talk about this in an interview

- **"Why five agents instead of one prompt?"** Each stage has a distinct
  failure mode and a distinct correctness check — retrieval relevance,
  numeric accuracy, grounding, schema completeness. Splitting them makes each
  independently testable and makes the run trace show exactly which stage
  produced a bad answer. The orchestrator, not the Planner, owns control flow:
  the plan is a hint, and retrieval-before-action and validation-last are
  invariants the orchestrator enforces no matter what the plan says.

- **"How do you keep it from making things up?"** Grounding is structural, not a
  prompt instruction. The deterministic Validation pass scores every claim
  sentence against the retrieved context and fails the run if too many are
  unsupported or if retrieval was never confident. Because there's no NLI model
  offline, the check is deliberately conservative overlap — I'd rather reject a
  true-but-weakly-supported claim than pass a fabricated one — and the system
  has an explicit "the corpus doesn't cover this" output.

- **"What's the mock mode for?"** Two things: the whole system stays usable and
  honest about its confidence when the API is down, and the evaluation harness
  is fully reproducible — deterministic embeddings plus deterministic agents
  mean `response_consistency` is 1.0 by construction, so any regression in the
  other six metrics is a real signal. Claude mode layers refinement on top but
  never owns the trustworthy parts: citations, the checklist, and the pass/fail
  verdict.

- **"Where does MCP fit?"** The tool contract is plain JSON Schema, so one
  `ToolRegistry` implementation serves the in-process Action agent, Anthropic
  tool-use, and an stdio MCP server with zero duplication. Every tool call in a
  run is captured as a timed trace record in the response, which made debugging
  agent behaviour and scoring tool-selection accuracy straightforward.

- **"What did you deliberately leave out?"** Auth, rate limiting,
  multi-tenancy, async ingestion, neural embeddings, streaming — all edge
  infrastructure that a real deployment adds without touching the agent/RAG
  core. Leaving them out keeps the codebase small enough to read end to end,
  and each omission has a named seam for adding it later.
