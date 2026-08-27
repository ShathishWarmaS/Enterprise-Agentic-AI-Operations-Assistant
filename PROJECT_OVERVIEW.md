# Project Overview

This document explains *why* the Enterprise Agentic AI Operations Assistant is
built the way it is. For the request/response mechanics and diagrams, see
[ARCHITECTURE.md](ARCHITECTURE.md); for setup and API examples, see the
[README](README.md).

---

## Why this architecture was chosen

**A pipeline of narrow agents, not one prompt.** The task — turn a messy
operational question into a grounded, cited decision — decomposes cleanly into
"plan it", "find the context", "check the numbers", "draft the decision",
"verify the decision". Each of those has a different failure mode and a
different correctness check. Splitting them into separate agents keeps every
step individually testable, lets the orchestrator degrade one step without
losing the others, and makes it obvious in the run trace which stage produced a
bad answer.

**An orchestrator that owns control flow.** The Planner *proposes* an ordered
plan, but `Orchestrator.run()` decides what actually executes: retrieval always
runs first (so the Action step has grounding), data analysis runs only if the
plan asked for it *and* tables exist, and validation always runs last regardless
of the plan. A plan is a hint, not a program — this stops a bad plan from
producing an ungoverned run.

**Deterministic core, optional model.** Every agent has a pure-Python path that
runs with no network access. Claude, when enabled, *refines* that path but never
replaces the parts that must be trustworthy: citations and the remediation
checklist come from deterministic tools in both modes, and the deterministic
Validation pass always owns the final pass/fail decision. This keeps the
evaluation harness reproducible and means the system still works — and still
tells the truth about its confidence — when the API is unavailable.

**Grounding is a hard requirement, not a prompt instruction.** The system is
designed to refuse. If retrieval is not confident and content-word overlap
between the drafted claims and the retrieved context is weak, the run is marked
ungrounded and `degraded`, and the answer says the corpus does not cover the
question. "I don't know" is a first-class output.

**Small, swappable seams.** The retrieval layer is pluggable and these seams are
real, not hypothetical: `Embedder` (`app/retrieval/embeddings.py`) has a
`HashingEmbedder` default and a `SentenceTransformerEmbedder` alternative, and
`VectorBackend` (`app/retrieval/backends/`) has `LocalVectorBackend` (in-process
FAISS/numpy) and `PgVectorBackend` (shared Postgres, multi-worker). `EMBEDDING_BACKEND`
and `VECTOR_BACKEND` pick them; the `build_embedder` / `build_vector_store`
factories wire them. `container.py` is the single place dependencies are
constructed, so tests and alternate backends have one seam to override. The
`Tool` contract is plain JSON Schema so the same tool definitions serve the
Action agent and an MCP server.

---

## Agent responsibilities

All agents extend `app/agents/base.py::Agent`, which holds the `LLMClient` and a
`with_retry` helper. `uses_claude` is true only when `LLM_MODE=claude` and the
client built successfully.

### Planner (`app/agents/planner.py`)

- **Input:** the raw request string, plus a `has_tables` boolean.
- **Output:** a `Plan` (`request`, ordered `list[PlanStep]`, `rationale`). Each
  `PlanStep` names exactly one agent (`retrieval`, `data_analysis`, `action`,
  `validation`) and optionally one tool.
- **Mock path:** keyword routing. `_DATA_HINTS` ("how many", "rate", "per
  service", "anomal", …) plus `has_tables` decide whether a `data_analysis` step
  is added; `_DOC_HINTS` ("runbook", "why", "cause", …) or the absence of a data
  intent add a `retrieval` step. An `action` step and a `validation` step are
  always appended.
- **Claude path:** `LLMClient.structured(PLANNER_SYSTEM, …, model=Plan)`. The
  result is passed through `_sanitise`, which drops steps naming unknown agents
  and guarantees the plan still ends with `action` + `validation`. On repeated
  LLM failure the Planner falls through to the deterministic plan.

### Retrieval agent (`app/agents/retrieval_agent.py`)

- **Input:** a query string and optional `top_k`.
- **Output:** `gather()` returns a `RetrievalResult` (chunks + `top_score` +
  `confident`); `answer_query()` returns a `QueryAnswer` (`answer`, inline
  `[n]` `citations`, `supporting_chunks`, `confident`, `notes`).
- **Empty corpus:** returns a non-confident answer stating nothing has been
  ingested.
- **Mock path (`_extractive_answer`):** scores every sentence of every retrieved
  chunk by word overlap with the query, quotes the top three (each tagged with
  its source marker), and computes two guards — query-term coverage against the
  whole retrieved corpus and best single-sentence overlap. If coverage < 0.5, or
  retrieval was not confident and overlap < 0.2, it returns an explicit "the
  documents do not cover this" answer with no citations. Otherwise `confident`
  requires `result.confident and best_overlap >= 0.3 and coverage >= 0.5`.
- **Claude path (`_claude_answer` + `_assemble`):** asks Claude for an
  `_AnswerDraft` under `QUERY_ANSWER_SYSTEM`, then rebuilds citations from the
  chunks the model actually referenced (`used_sources`, or `[n]` markers found
  in the text). If the model cited nothing the answer is forced to
  low-confidence. On repeated LLM failure it falls back to the extractive answer
  and notes that in `notes`.

### Data Analysis agent (`app/agents/data_agent.py`)

- **Input:** the request string. Reads tables from the `TableStore`.
- **Output:** `(DataAnalysisResult, list[ToolCall])` — `row_count`, `findings`
  (metric / value / observation), `anomalies` (strings), `missing_fields`, plus
  the `query_data` tool-call traces it made.
- **No tables:** returns a result whose `missing_fields` is
  `["no tabular data has been ingested"]` and an empty call list.
- **Behaviour (identical in both modes — this agent has no Claude path):** picks
  the table whose column names overlap the request most, always reports
  `row_count`, detects a group-by (`by|per|across|for each <col>`) and an
  aggregation keyword (`\baverages?\b`, `\bmax\b`, … matched on word boundaries
  so "summarise" is not read as "sum"), runs grouped counts for "how many … per
  X" phrasing, and computes the chosen aggregate over named or the first two
  numeric columns. `_anomalies` flags entirely-empty columns, columns >30% null,
  extreme spread relative to the mean, and a single value dwarfing the median
  (likely sentinel / instrumentation error). The `DATA_ANALYSIS_SYSTEM` prompt
  exists but is not currently wired to a call.

### Action agent (`app/agents/action_agent.py`)

- **Input:** `request`, the `RetrievalResult`, and an optional
  `DataAnalysisResult`.
- **Output:** `(OperationalDecision, list[ToolCall])`. The decision carries the
  `IncidentSummary` (title, severity, summary, impact, likely_cause, evidence),
  `recommended_next_steps`, `remediation_checklist`, `citations`,
  `data_findings`, `open_questions`, and a `confidence` string.
- **Deterministic assembly (both modes):** severity from keyword tables
  (`outage`→critical, `5xx`/`timeout`→high, …); observations built only from
  sentences of *confident* retrieval (so weak matches never get authoritative
  `[n]` markers) plus data findings and anomalies; `likely_cause` from the
  best-scoring sentence containing a real causal connective ("because", "due
  to", "caused by", …); `impact` from the top data finding or the request
  alone. It then **invokes two tools** — `draft_incident_summary` and
  `generate_checklist` — via `Tool.invoke()`, and builds `recommended_next_steps`
  from recommendation-verb sentences in the context plus the blocking checklist
  items. `_confidence` returns `high` only when retrieval is confident with ≥2
  chunks and no open questions.
- **Claude path (`_claude_refine`):** asks Claude for an improved
  `OperationalDecision` under `ACTION_SYSTEM`, given the context, the data JSON,
  and the deterministic draft. Afterwards the code **overwrites**
  `draft.citations` and `draft.incident.evidence` with the tool-built citations
  and restores the deterministic checklist if the model dropped it. Claude can
  reword the analysis; it cannot change the evidence or the checklist.

### Validation agent (`app/agents/validation_agent.py`)

- **Input:** the `OperationalDecision`, the retrieved chunks, an
  `expect_grounding` flag, and optional `data_context` (data-finding strings,
  treated as first-class evidence).
- **Output:** a `ValidationReport` (`passed`, `grounded`, `issues`,
  `unsupported_sentences`, `checked_claims`, `supported_claims`).
- **Checks:** required fields present and non-empty (`incident.title/summary/
  impact/likely_cause`, `recommended_next_steps`, `remediation_checklist`),
  severity in the allowed set, and every claim sentence (incident summary +
  likely cause — *not* advice or the standard IR spine) checked against context
  with `grounding.is_supported`. `grounded` requires ≥60% of claims supported
  and some context to exist.
- **Pass rule:** missing/invalid fields always block; individual unsupported
  sentences do not, but a run fails if more than 34% of claims are unsupported
  or if it is largely ungrounded while grounding was expected.
- **Claude path (`_augment_with_claude`):** asks Claude for a second-opinion
  `ValidationReport` under `VALIDATION_SYSTEM` and appends any *new* concerns.
  The deterministic pass still owns `passed` — Claude can add issues, never
  remove them.

---

## RAG flow

```
ingest → clean → chunk → embed → index → retrieve → confidence gate → cited answer
```

1. **Ingest** (`app/ingestion/loaders.py`). Extension → `SourceType`. PDF via
   PyMuPDF (per-page text with `page N` locators; raises on a scanned
   image-only PDF). CSV/TSV via pandas with a ragged-row recovery path that caps
   splits at header width and folds overflow into the last column. JSON: a list
   of flat objects becomes a DataFrame (`json_normalize`), anything else stays
   as pretty-printed text. Markdown / text read as UTF-8 with replacement.
   Every loader is total: it returns a `LoadedSource` or raises `LoaderError`.
2. **Clean** (`app/ingestion/cleaning.py`, tabular sources only) — see the next
   section. Produces a cleaned frame plus a `CleaningReport`.
3. **Chunk** (`app/ingestion/chunking.py`). Text is split on Markdown headings
   then blank lines, packed greedily to `CHUNK_SIZE` (800) chars with
   `CHUNK_OVERLAP` (120) carried between chunks; PDF pages keep their page
   locator. Tables get one `schema` summary chunk plus `rows a–b` window chunks
   so both "what columns exist" and "which rows say X" retrieve something.
4. **Embed** (`app/retrieval/embeddings.py`). Deterministic 512-dim hashed
   bag-of-n-grams: word unigrams, word bigrams, and character trigrams
   (so "rollback" ≈ "roll back", "5xx" ≈ "5 xx"), signed-hashed into buckets
   with sublinear term-frequency damping, L2-normalised. No model download, no
   randomness.
5. **Index** (`app/retrieval/vector_store.py`). FAISS `IndexFlatIP` over
   normalised vectors (= cosine) when `faiss` imports, else a numpy
   brute-force dot product. Vectors in `vectors.npy`, chunk metadata in
   `chunks.json`; positions line up.
6. **Retrieve** (`app/retrieval/retriever.py`). Embed the query, take top-k,
   de-dupe by `chunk_id`, keep chunks at or above `RETRIEVAL_MIN_SCORE` (0.22)
   but always keep at least the best one.
7. **Confidence gate.** `confident = top_score >= min_score`. The Retrieval
   agent layers its own overlap/coverage guards on top (above). A non-confident
   result still returns its best chunk so the caller can show *something* and
   explain why it is unsure.
8. **Cited answer.** `build_citations` numbers the supporting chunks `[1]`,
   `[2]`, … as `Citation(marker, filename, locator, chunk_id)`; the answer text
   carries those markers inline.

---

## MCP / tool-calling flow

**One registry, two surfaces.** `app/tools/registry.py::ToolRegistry` is
constructed once in `container.py` with the `Retriever` and `TableStore`. It is
the *only* place the Action agent and the stdio MCP server
(`scripts/mcp_server.py`) look up tools, so the tool set is identical on both.

**The `Tool` contract** (`app/tools/base.py`): a `name`, a `description`, a
plain-JSON-Schema `input_schema` (works unchanged as an Anthropic
`input_schema` via `to_anthropic()` and as an MCP `inputSchema` via
`to_mcp()`), and `run(arguments) -> dict`. `Tool.invoke(arguments)` runs `run`,
times it, and returns a `ToolCall` trace (`tool`, `arguments`, `ok`, `result`
or `error`, `duration_ms`) — it never raises; expected failures surface as
`ok=False`.

**The six tools:**

| Tool | Backed by | Purpose |
| --- | --- | --- |
| `search_documents` | `Retriever` | vector search; returns chunks with filename, locator, score, `confident` |
| `query_data` | `TableStore` | validated structured query — projection, filters, group-by, aggregation |
| `compute_metrics` | `TableStore` | several named aggregations over one table in one call |
| `check_schema` | `TableStore` | required-column check, mostly-null flagging, coarse type check |
| `draft_incident_summary` | deterministic template | formats given observations + evidence into an `IncidentSummary`; invents nothing |
| `generate_checklist` | deterministic keyword playbook | standard IR spine + scenario-specific steps chosen by keyword |

**How the Action agent uses them.** It calls
`self._tools.get("draft_incident_summary").invoke({...})` and
`self._tools.get("generate_checklist").invoke({...})`, collects the two
`ToolCall` records into the `AgentStep`, and reads `call.result` back into
`IncidentSummary` / `ChecklistItem` models. The Data Analysis agent similarly
records its `query_data` calls as `ToolCall`s. Every tool invocation in a run is
visible in the response trace.

**MCP server parity.** `scripts/mcp_server.py` lists `registry.all()` as MCP
tools and routes `call_tool(name, arguments)` straight through
`registry.get(name).invoke(arguments)`, returning `call.result` (or
`{"error": …}`) as JSON text. There is exactly one implementation of each tool.

---

## Data-cleaning approach

`clean_frame` (`app/ingestion/cleaning.py`) makes tabular data *predictable*,
not "perfect", and records everything it changed or distrusted in a
`CleaningReport` (`rows_in`, `rows_out`, `dropped_rows`, `coerced_cells`,
`issues: list[CleaningIssue]` with `info` / `warning` / `error` severity).

- **Column normalisation.** Non-alphanumerics → `_`, lower-cased, stripped;
  empty names become `column`; duplicates get a numeric suffix and a `warning`.
- **Whitespace + sentinels.** Every string cell trimmed; `""`, `null`, `NA`,
  `N/A` → `NaN`.
- **Empty structure.** Entirely-empty columns dropped (with a `warning`);
  all-NaN rows dropped and counted in `dropped_rows`.
- **Type coercion, threshold-gated.** Datetime first: `to_datetime(format=
  "mixed")` accepted only if **≥90%** of non-null values parse. Then numeric:
  strip `,$%` and whitespace, `to_numeric`; accepted only if **≥70%** of
  non-null values survive. Values lost to an accepted coercion are counted in
  `coerced_cells` and raise a `warning`. A coercion that would destroy too much
  is rejected and the column stays as-is.
- **Duplicates.** Exact duplicate rows dropped (keep first) with a `warning`.
- **Quality flags (`_flag_quality`).** Per-column null rate between 0 and 1 →
  `info`, or `error` above 50%. For numeric columns with ≥8 values and non-zero
  std, points more than **4 standard deviations** from the mean are flagged
  individually as possible bad data.

Downstream, the Data Analysis agent's `_anomalies` adds runtime checks (extreme
spread, sentinel-vs-median) and `check_schema` re-checks required columns and
coarse types on demand.

---

## Evaluation approach

`app/services/evaluation.py` runs each labelled case in
`sample_data/eval_cases.json` (7 cases: 4 `query`, 3 `agent`, one flagged
`unanswerable`) through the **real** retrieval / orchestrator code and computes
seven metrics from the outputs — nothing is hard-coded or sampled.

| Metric | Computed as |
| --- | --- |
| `retrieval_relevance` | recall@k of the labelled `expected_sources` among cited + top-4 retrieved filenames (1.0 for unanswerable cases iff nothing was cited) |
| `citation_presence` | fraction of runs with citations (or, for unanswerable cases, correctly *without* citations) |
| `tool_selection_accuracy` | Jaccard of (planned ∪ called) tools against `expected_tools` |
| `structured_output_validity` | decision parses and has title + next steps + checklist |
| `groundedness` | `validation.grounded and validation.passed` and the answer contains the expected substrings |
| `missing_info_handling` | for unanswerable cases only: the system was not confident / raised open questions |
| `response_consistency` | run each case twice; equal answer / decision signature → 1.0 |

`pass_rate` is the mean of per-case `passed`. In **mock mode** every run is
deterministic, so `response_consistency` is 1.0 *by construction* — that is a
property of mock mode, not a claim about Claude mode.

**CI gate.** `scripts/run_eval.py` exits non-zero when `pass_rate` drops below
`--min-pass` (default `0.6`; CI runs it at `0.7` after `scripts/seed.py`). It
refuses to run against an empty vector store.

---

## Retry & failure-handling strategy

- **Transport errors (`LLMClient._call`).** `tenacity` retries
  `APIConnectionError` / `RateLimitError` / `InternalServerError` up to 3 times
  with exponential backoff. A 4xx `APIStatusError` is *not* retried — it is a
  request problem, raised as `LLMOutputError`.
- **Malformed JSON (`LLMClient.structured`).** One repair retry: the parser
  error is fed back to the model with "return corrected JSON only". A second
  failure raises `LLMOutputError` — no guessing.
- **Agent level (`Agent.with_retry`).** Wraps an LLM call, retrying only
  `LLMOutputError` / `LLMError`; any other exception propagates so real bugs are
  not masked. Returns `(result_or_none, retries_used, error)`. Every agent's
  Claude path falls back to its deterministic path when this returns `None`.
- **Orchestrator level.** Each step (retrieval, data, action) runs in its own
  `try/except`; a failure sets `degraded=True`, records the error on that
  `AgentStep`, and the run continues — retrieval failure substitutes an empty
  `RetrievalResult` so Action and Validation still run. Validation always runs;
  if there is no decision at all it emits a failing report. `degraded` is also
  set whenever validation does not pass.
- **No silencing.** The two broad `except Exception` blocks in the orchestrator
  are annotated `# noqa: BLE001`, log the full traceback, and record the error
  in the response. `main.py` logs unhandled exceptions server-side and returns a
  generic 500.

---

## Security considerations

- **No arbitrary SQL or `eval`.** The Data Analysis agent and `query_data`
  never run SQL or `df.query` strings. They pass a validated `QuerySpec`
  (`app/data/tables.py`) — column names checked against the real schema, a
  fixed operator allow-list (`eq/ne/gt/gte/lt/lte/contains/isnull/notnull`), a
  fixed aggregation allow-list, a `limit` capped at 1000, and a table name
  sanitised to `[a-z0-9_]`. A hallucinated or hostile query does nothing worse
  than return an empty result or a `KeyError`.
- **Secrets only via env.** `app/config.py` is the single reader of the
  environment; `LLM_MODE=claude` fails fast at startup if `ANTHROPIC_API_KEY`
  is missing. `.env` is gitignored; `.env.example` ships with empty secret
  values.
- **Non-root container.** The Dockerfile creates `appuser` (uid 10001), chowns
  `/app`, and runs as that user.
- **Upload cap.** `POST /documents/upload` rejects bodies over 20 MB with a
  413; filenames are sanitised before touching disk.
- **Logs avoid secrets.** Startup logs mode and backend names only; the LLM
  layer logs parser errors and attempt counts, not payloads or keys.

---

## Production-readiness trade-offs

Deliberately **out of scope** for a portfolio project, with the reasoning:

- **No authentication / authorization.** The API is unauthenticated. Adding
  real auth would be boilerplate that demonstrates nothing about the RAG /
  agent design, which is the point of the project.
- **No rate limiting.** Same reasoning; it belongs at a gateway, not in app
  code.
- **No multi-tenancy.** One corpus, one table namespace. Per-source access
  control and PII redaction are noted as future work in the cleaning stage.
- **Synchronous ingestion.** Upload → ingest is a blocking request. Fine for
  the sample corpus; a Redis Streams queue is the noted path for large batches.
- **Lexical embeddings by default.** Hashed n-grams, not a neural encoder — no
  model download, fully deterministic evaluation. `EMBEDDING_BACKEND=sentence_transformers`
  swaps in a real encoder (needs the `embeddings` extra); the `Embedder` seam is
  built for exactly this.
- **In-process vector store by default.** `VECTOR_BACKEND=pgvector` moves the
  index into Postgres for multi-worker deploys; the `VectorBackend` seam is real.
- **No streaming.** `/agent/run` returns the whole result at once. SSE is noted
  as future work; it would not change any correctness property.

The through-line: everything omitted is infrastructure that a real deployment
adds at its edges, none of it changes the agent/RAG/grounding core, and leaving
it out keeps the codebase small enough to read in one sitting.
