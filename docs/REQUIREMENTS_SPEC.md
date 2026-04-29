# Software Requirements Specification — research-agent

**Version**: 1.0
**Date**: 2026-04-29
**Status**: Initial draft
**Repository**: `https://github.com/brandon-behring/research-agent`

This document specifies *what* `research-agent` must do and *how* it will be validated. It complements the architectural rationale in [README.md § Architecture Decisions](../README.md#architecture-decisions) — that section answers *why* we built it this way; this document answers *what it must do* and *how we verify it*.

The format follows IEEE 830 / ISO/IEC/IEEE 29148 conventions (numbered FRs/NFRs, traceability matrix, given/when/then acceptance criteria) but uses plainspoken prose rather than "the system shall" boilerplate.

---

## 1. Introduction

### 1.1 Purpose

`research-agent` is a multi-agent research analysis pipeline that decomposes a natural-language research question, retrieves evidence from a knowledge base via MCP tools, traces conceptual connections, audits methodological assumptions, and produces a structured synthesis report with cited sources.

This SRS specifies the system's required capabilities (FRs), measurable quality bounds (NFRs), external interfaces (MCP tool contracts), data model (Pydantic state schema), use cases, constraints, and a traceability matrix mapping requirements to existing tests in `tests/`.

### 1.2 Scope

In scope:

- The LangGraph-orchestrated pipeline of seven nodes (Query Planner, Literature Search, Concept Explorer, Citation Analyzer, Assumption Auditor, Connection Explorer, Synthesis Writer)
- The CLI entry point (`research-agent` console script)
- The MCP client wrapper for `research-kb` (stdio + HTTP transports)
- The SQLite report cache
- Configuration loading (env vars + `AgentConfig`)

Out of scope:

- The `research-kb` knowledge base itself (separate repository, separate SRS)
- LangSmith tracing instrumentation (orthogonal observability layer)
- Future enhancements listed in `ROADMAP.md` (tool-call planning mode, multi-KB routing)

### 1.3 Definitions and Acronyms

| Term | Definition |
|------|-----------|
| MCP | Model Context Protocol — JSON-RPC standard for LLM tool invocation |
| ADR | Architecture Decision Record — informal rationale capture |
| SRS | Software Requirements Specification — this document |
| FR | Functional Requirement |
| NFR | Non-Functional Requirement |
| UC | Use Case |
| RTM | Requirements Traceability Matrix |
| KB | Knowledge Base — refers specifically to `research-kb` |
| Sub-task | A decomposed unit of research work produced by the Query Planner |
| Source | A document in the KB (paper, book chapter, technical note) |
| Concept | A node in the KB's typed knowledge graph |

### 1.4 References

- IEEE 830-1998 (Recommended Practice for Software Requirements Specifications)
- ISO/IEC/IEEE 29148:2018 (Systems and software engineering — Life cycle processes — Requirements engineering)
- LangGraph documentation: <https://langchain-ai.github.io/langgraph/>
- Model Context Protocol: <https://modelcontextprotocol.io/>
- `research-kb` repository: <https://github.com/brandon-behring/research-kb>
- Internal: `README.md` § Architecture (pipeline diagram, MCP tool mapping)
- Internal: `docs/eval_baselines.md` (NFR-001 calibration source)
- Internal: `CLAUDE.md` (developer-facing operational notes)

### 1.5 Document Conventions

- FRs are numbered FR-001..N. NFRs are numbered NFR-001..N. Use cases are UC-001..N.
- Each FR/NFR includes a description, acceptance criteria (given/when/then where appropriate), and at least one test reference.
- Cross-references to source code use repository-relative paths (e.g., `src/research_agent/graph.py`).

---

## 2. Overall Description

### 2.1 Product Perspective

`research-agent` is a stand-alone Python application packaged as a console script. It depends on:

- `research-kb` — provides the knowledge base via MCP (stdio or HTTP transport)
- An LLM provider — Anthropic by default; provider-agnostic dispatch via LangChain's `init_chat_model` allows swapping in OpenAI, Ollama, etc.

It does not embed `research-kb`. The two services run independently and communicate via MCP.

### 2.2 Product Functions (overview)

The system answers a research question by:

1. Decomposing the question into sub-tasks (each with target concepts, search queries, methods to audit)
2. Retrieving relevant sources via 4-signal hybrid search across the KB
3. Traversing the concept graph for context
4. Mapping citation networks among retrieved sources
5. Auditing the assumptions of any statistical methods identified
6. Tracing graph paths between concept pairs flagged by the planner
7. Synthesizing a structured report with citation validation

Steps 3 and 4 run in parallel (LangGraph fan-out). Step 5 runs conditionally (only when methods are present in the planner output or auto-discovered from the concept graph). The full pipeline reaches a structured Markdown report in 191-210s typical end-to-end latency for the three golden cases in `docs/eval_baselines.md`.

### 2.3 User Characteristics

The intended user is a research-oriented developer or analyst comfortable with the command line and structured Markdown output. The CLI is the primary surface; downstream programmatic consumption uses the `--json` flag for structured JSON output.

### 2.4 Constraints

- Python 3.11+ (matches CI matrix: 3.11, 3.12, 3.13)
- Anthropic API key required for default models; alternative providers require corresponding LangChain integration packages
- For stdio transport: research-kb repository must be cloned locally and have a `.venv` with research-kb installed
- For HTTP transport: research-kb must be reachable at the configured `RESEARCH_KB_URL` + `MCP_PATH`

### 2.5 Assumptions and Dependencies

- Anthropic API is available with sufficient rate limits for the 60-180s synthesis call window
- research-kb has at least one populated domain
- The MCP protocol version is compatible (the `mcp` Python package handles version negotiation)
- Network is sufficiently stable for the duration of a single query — node-level timeouts handle transient failures, but inter-node state transitions assume no fatal connection loss

---

## 3. Functional Requirements

### FR-001 — Decompose research query into sub-tasks

**Description**: Given a natural-language research question, the system decomposes it into 1-5 sub-tasks. Each sub-task carries a description, search queries, target concepts, methods to audit (optional), domain hint (optional), search-context weighting, and concept pairs to explain (optional).

**Code reference**: `src/research_agent/nodes/query_planner.py`

**Acceptance criteria**:

- *Given* a question of any reasonable complexity, *when* the planner runs with the default `PLANNING_MODEL` (Haiku), *then* it returns a `list[SubTask]` with at least one entry and at most 5 entries.
- *Given* a question that mentions a statistical method by name (e.g., "DML", "instrumental variables"), *when* the planner runs, *then* the method appears in `methods_to_audit` for at least one sub-task.
- *Given* a question that compares two concepts (e.g., "DML vs IV"), *when* the planner runs, *then* the concept pair appears in `connections_to_explain` for at least one sub-task.
- The planner must return within 60s; if not, `NodeTimeoutError` is raised (NFR-003).

**Tests**: `tests/test_nodes.py` (query planner cases), `tests/test_graph.py` (entry-point wiring).

### FR-002 — Hybrid literature search across the knowledge base

**Description**: For each sub-task, the system runs `research_kb_search` (4-signal hybrid: BM25 + vector + graph + PageRank citation authority) for each search query. If `research_kb_search` returns insufficient results or errors, the system falls back to `research_kb_fast_search` (vector-only, ~200ms).

**Code reference**: `src/research_agent/nodes/literature_search.py`, `src/research_agent/mcp_client.py` (search + fast_search)

**Acceptance criteria**:

- *Given* a sub-task with non-empty `search_queries`, *when* literature_search runs, *then* `state.search_results` is populated with at least the union of all queries' results, deduplicated by `source_id`.
- *Given* a transient `MCPToolError` from `research_kb_search`, *when* literature_search retries (up to 3 attempts with exponential backoff per `mcp_client.py`), *then* the system either succeeds or falls back to `fast_search`.
- *Given* parallel sub-tasks, *when* literature_search invokes the KB concurrently, *then* concurrency is bounded by `asyncio.Semaphore(3)` to prevent connection-pool exhaustion (per `docs/eval_baselines.md`).

**Tests**: `tests/test_nodes.py` (literature_search cases), `tests/test_mcp_client.py` (search + fast_search + retry behavior).

### FR-003 — Concept graph traversal

**Description**: For each `concepts_to_explore` item across sub-tasks, the system retrieves concept details (`research_kb_get_concept`) and traverses 2-hop neighborhoods (`research_kb_graph_neighborhood`). If `ENABLE_CROSS_DOMAIN=true`, the explorer additionally bridges concepts across domains via `research_kb_cross_domain_concepts`. The explorer also auto-discovers methods from the concept neighborhoods that the planner did not explicitly list, surfacing them in `state.discovered_methods` for downstream conditional auditing (FR-005).

**Code reference**: `src/research_agent/nodes/concept_explorer.py`

**Acceptance criteria**:

- *Given* a sub-task with non-empty `concepts_to_explore`, *when* concept_explorer runs, *then* `state.concepts` is populated with each concept's details and neighborhood summary.
- *Given* `ENABLE_CROSS_DOMAIN=true`, *when* concept_explorer runs, *then* `state.cross_domain_matches` is populated for any concept that has cross-domain bridges in the KB.
- *Given* a concept neighborhood references a statistical method not in the planner output, *when* concept_explorer surfaces it, *then* it appears in `state.discovered_methods` and is eligible for FR-005 audit.

**Tests**: `tests/test_nodes.py` (concept explorer cases).

### FR-004 — Citation network and bibliographic coupling analysis

**Description**: For the top-N search results (controlled by `TOP_RESULTS_FOR_CITATIONS`, default 5), the system retrieves citation networks (`research_kb_citation_network`) and bibliographic coupling neighbors (`research_kb_biblio_coupling`).

**Code reference**: `src/research_agent/nodes/citation_analyzer.py`

**Acceptance criteria**:

- *Given* `state.search_results` is non-empty, *when* citation_analyzer runs, *then* `state.citations` is populated with citing/cited-by chains and biblio_coupling neighbors for the top `TOP_RESULTS_FOR_CITATIONS` sources.
- *Given* a source with no available citation data, *when* citation_analyzer runs, *then* it returns gracefully with an empty `CitationInfo` rather than raising.

**Tests**: `tests/test_nodes.py` (citation analyzer cases).

### FR-005 — Conditional method assumption audit

**Description**: After `analysis_join` (the fan-in barrier following parallel concept_explorer + citation_analyzer), the graph routes conditionally:

- If any sub-task has non-empty `methods_to_audit` OR `state.discovered_methods` is non-empty → run `assumption_auditor`
- Else → skip directly to `connection_explorer`

The auditor calls `research_kb_audit_assumptions` for each method, retrieving structured assumption details, violation consequences, and verification approaches.

**Code reference**: `src/research_agent/nodes/assumption_auditor.py`, `src/research_agent/graph.py` (`_should_audit_assumptions` router)

**Acceptance criteria**:

- *Given* sub-tasks contain at least one `methods_to_audit` entry, *when* the conditional router fires, *then* `assumption_auditor` is the next node.
- *Given* no methods are in `methods_to_audit` and `discovered_methods` is empty, *when* the conditional router fires, *then* `connection_explorer` is the next node (assumption_auditor is skipped).
- *Given* `assumption_auditor` runs, *when* it completes, *then* `state.assumption_audits` is populated with structured assumption details per audited method.

**Tests**: `tests/test_nodes.py` (assumption auditor), `tests/test_graph.py` (`_should_audit_assumptions` routing — both branches).

### FR-006 — Concept-path explanation via knowledge graph

**Description**: For each `connections_to_explain` pair flagged by the planner, the system invokes `research_kb_explain_connection` in graph-only mode (`use_llm=False`) to retrieve a deterministic graph path with evidence chunks. Synthesis later weaves these paths into the final report's narrative.

**Code reference**: `src/research_agent/nodes/connection_explorer.py`

**Acceptance criteria**:

- *Given* sub-tasks contain non-empty `connections_to_explain`, *when* connection_explorer runs, *then* `state.connection_explanations` is populated with one entry per concept pair, each containing path steps and evidence chunks.
- *Given* a connection pair where no graph path exists, *when* `explain_connection` returns an empty path, *then* connection_explorer logs a warning and continues to the next pair.
- The explorer must complete within 30s combined for all pairs (`_CONNECTION_TIMEOUT_SECONDS` in `connection_explorer.py`); otherwise the node times out and downstream synthesis proceeds without these explanations.

**Tests**: `tests/test_connection_explorer.py` (covers happy path, parse failures, timeout).

### FR-007 — Synthesize structured report with citation validation

**Description**: The synthesis writer (default model: Sonnet) consumes the entire state — search results, concepts, citations, assumption audits, connection explanations — and produces a structured Markdown report with sections: Executive Summary, Key Findings, Concept Map, Citation Landscape, Methodological Considerations, Gaps & Limitations, Confidence Assessment.

After synthesis, the system validates that each in-text citation `[Source: <id>]` corresponds to a real source in `state.search_results`; any hallucinated citations are flagged in `state.evidence_metadata`.

**Code reference**: `src/research_agent/nodes/synthesis.py`

**Acceptance criteria**:

- *Given* a populated `ResearchState`, *when* synthesis runs, *then* `state.report` is non-empty and contains all 7 expected sections.
- *Given* a generated report, *when* citation validation runs, *then* `state.evidence_metadata` records the count of valid vs hallucinated citations.
- *Given* a report with hallucinated citations, *when* validation completes, *then* the validation result is included in `state.evidence_metadata` (the report is not silently rejected — the user sees both the report and the validation findings).

**Tests**: `tests/test_synthesis.py` (citation validation, structured-output parsing, multi-section coverage).

---

## 4. Non-Functional Requirements

| ID | Requirement | Bound | Verification |
|----|-------------|-------|--------------|
| NFR-001 | End-to-end pipeline latency p95 | ≤ 240s | Calibrated against `docs/eval_baselines.md` (191-210s observed across 3 golden cases). Latency budget assumes stdio transport; HTTP transport is permitted to be faster (eval baselines note 2-3× speedup expected with HTTP + remote KB). The README's prior "~3 minutes" claim corresponds to this NFR. |
| NFR-002 | Cache-hit latency for repeated queries | ≤ 50ms | SQLite query by `(query_hash, model_config)` key; tested in `tests/test_cache.py`. |
| NFR-003 | Per-node timeout enforcement | Each node respects `NODE_TIMEOUTS` config; on timeout, `NodeTimeoutError` is raised and the graph proceeds with the partial state already accumulated | Default timeouts: planner 60s, search 120s, explorer 90s, citations 90s, auditor 60s, connection 60s, synthesis 180s. Tested in `tests/test_graph.py`. |
| NFR-004 | Structured JSON output for programmatic use | `--json` flag emits a single JSON object on stdout; logs go to stderr | Tested in `tests/test_cli.py`. |
| NFR-005 | Test coverage gating CI | ≥ 80% line coverage | Enforced via `pytest --cov-fail-under=80` in `.github/workflows/ci.yml`. Current observed: ~96% (per recent commit `dfac6fd`). |
| NFR-006 | MCP transport flexibility | Both stdio (subprocess) and HTTP transports supported behind a single client interface | `MCPConfig.transport` selects; tested in `tests/test_mcp_client.py`. |
| NFR-007 | Provider-agnostic LLM dispatch | Any LangChain-supported provider usable via `PLANNING_MODEL` / `SYNTHESIS_MODEL` env vars (e.g., `ollama/llama3`, `openai/gpt-4o`) | Tested in `tests/test_llm.py`. |
| NFR-008 | Graceful degradation on KB tool errors | Transient errors retry up to 3× with exponential backoff; permanent errors surface as `MCPToolError` and the node returns a degraded result rather than crashing the graph | Implemented via `tenacity` retry decorator in `mcp_client.py`. |
| NFR-009 | No secrets in source | gitleaks pre-commit hook + GitHub Actions secret scanning | `.pre-commit-config.yaml` runs gitleaks. |

---

## 5. External Interface Specifications

### 5.1 MCP Tool Contracts

The agent calls 13 tools on `research-kb`. Each contract is `Operation: input → output`. All tools are invoked through `ResearchKBClient` (`src/research_agent/mcp_client.py`), which handles transport selection (stdio vs HTTP), retries, and tracing.

| # | Tool | Input | Output | Used by |
|---|------|-------|--------|---------|
| 1 | `research_kb_search` | `{query, limit, domain?, context_type, use_graph, use_rerank, use_citations, citation_weight, use_expand}` | JSON: ranked results with title, content, source_id, score, authors, year, chunk_id | FR-002 |
| 2 | `research_kb_fast_search` | `{query, limit, domain?}` | JSON: vector-only ranked results | FR-002 (fallback) |
| 3 | `research_kb_get_concept` | `{concept_id, include_relationships}` | JSON: concept details + edges | FR-003 |
| 4 | `research_kb_graph_neighborhood` | `{concept_name, hops, limit}` | JSON: connected concepts + relationship summary | FR-003 |
| 5 | `research_kb_citation_network` | `{source_id, limit}` | JSON: citing + cited-by chains | FR-004 |
| 6 | `research_kb_biblio_coupling` | `{source_id, limit, min_coupling}` | JSON: bibliographically similar sources | FR-004 |
| 7 | `research_kb_audit_assumptions` | `{method_name, include_docstring, domain?, scope}` | JSON: assumptions, violation consequences, verification approaches | FR-005 |
| 8 | `research_kb_explain_connection` | `{concept_a, concept_b, style, max_evidence_per_step, use_llm}` | JSON: graph path steps + evidence chunks | FR-006 |
| 9 | `research_kb_get_source` | `{source_id, include_chunks?, chunk_limit?}` | Markdown: full source metadata | FR-007 (synthesis enrichment) |
| 10 | `research_kb_find_similar_concepts` | `{concept_id, limit, threshold}` | Markdown: embedding-similar concepts | FR-003 (enrichment) |
| 11 | `research_kb_cross_domain_concepts` | `{source_domain, target_domain, concept_name?, concept_id?, similarity_threshold, limit}` | Markdown: cross-domain matches | FR-003 (cross-domain mode) |
| 12 | `research_kb_list_domains` | `{}` | Markdown: domain list | KB context (pre-pipeline) |
| 13 | `research_kb_stats` | `{}` | Markdown: corpus stats | KB context (pre-pipeline) |

### 5.2 CLI

```
research-agent [OPTIONS] QUERY

Options:
  -v, --verbose         Verbose logging
  --stream              Stream node-completion events to stderr
  -o, --output PATH     Write report to PATH instead of stdout
  --json                Emit structured JSON instead of Markdown
  --no-cache            Bypass cache for this query
  --clear-cache         Clear all cached reports and exit
  --help                Show help message
```

### 5.3 Environment Variables

See [`README.md` § Configuration → Environment variables](../README.md#configuration). Loaded via `pydantic-settings` from `AgentConfig` in `src/research_agent/config.py`.

---

## 6. Data Model

### 6.1 ResearchState (top-level mutable state)

`src/research_agent/state.py:ResearchState` — Pydantic v2 BaseModel. LangGraph reconstructs via `schema(**dict)`; nodes return partial dicts that LangGraph merges.

| Field | Type | Source | Description |
|-------|------|--------|-------------|
| `query` | `str` | input | Original research question |
| `sub_tasks` | `list[SubTask]` | FR-001 | Decomposed sub-tasks |
| `planning_rationale` | `str` | FR-001 | Planner's natural-language rationale |
| `search_results` | `list[SearchResult]` | FR-002 | Retrieved sources |
| `search_summary` | `str` | FR-002 | Search-level Markdown summary |
| `concepts` | `list[ConceptInfo]` | FR-003 | Explored concept details |
| `concept_map_summary` | `str` | FR-003 | Concept-map narrative |
| `citations` | `list[CitationInfo]` | FR-004 | Citation networks |
| `citation_summary` | `str` | FR-004 | Citation-landscape narrative |
| `assumption_audits` | `list[AssumptionAudit]` | FR-005 | Per-method assumption details |
| `assumption_summary` | `str` | FR-005 | Assumption-audit narrative |
| `discovered_methods` | `list[str]` | FR-003 → FR-005 | Methods auto-found from concept graph (drives conditional routing) |
| `connection_explanations` | `list[dict]` | FR-006 | Concept-pair graph paths with evidence |
| `kb_domains` | `list[str]` | KB context | Available domains (from `list_domains`) |
| `kb_stats_summary` | `str` | KB context | Corpus stats (from `stats`) |
| `similar_concepts` | `list[dict]` | FR-003 enrichment | Embedding-similar concepts |
| `cross_domain_matches` | `list[dict]` | FR-003 cross-domain mode | Cross-domain bridges |
| `source_details` | `list[dict]` | FR-007 enrichment | Full source metadata |
| `report` | `str` | FR-007 | Final structured Markdown report |
| `confidence_assessment` | `str` | FR-007 | Confidence level + rationale |
| `evidence_metadata` | `dict` | FR-007 | Citation-validation results |
| `current_node` | `str` (with reducer) | runtime | Last-writer-wins node-completion marker |

### 6.2 Immutable sub-models

All sub-models are frozen Pydantic models (`ConfigDict(frozen=True)`):

- `SubTask` — description, search_queries, concepts_to_explore, methods_to_audit, search_domain, search_context, connections_to_explain
- `SearchResult` — title, content, source_id, score (0.0-1.0), authors, year, chunk_id
- `ConceptInfo` — concept_id, name, concept_type, description, relationships, neighborhood_summary
- `CitationInfo` — source_id, source_title, citing, cited_by, similar_papers
- `AssumptionAudit` — method_name, assumptions, raw_output

### 6.3 Node return contract

Each node returns a `NodeUpdate` (a `TypedDict` with all fields optional). LangGraph merges these into the running `ResearchState`. Concurrent nodes (concept_explorer + citation_analyzer in parallel) write disjoint state slices, so no merge conflicts arise. The `current_node` field uses a last-writer-wins reducer (`_last_value`) to handle concurrent writes from parallel nodes.

---

## 7. Use Cases

### UC-001 — Single-question synthesis (happy path)

**Actor**: developer/analyst on the command line
**Pre-condition**: research-kb running locally (stdio) or reachable (HTTP); `ANTHROPIC_API_KEY` set
**Trigger**: `research-agent "What are the assumptions of double machine learning?"`
**Main flow**:

1. CLI parses arguments and validates env config
2. Cache lookup runs first; if hit and TTL valid → return cached report (UC-002)
3. Otherwise, planner decomposes the query into 4-5 sub-tasks (FR-001)
4. Literature search runs hybrid retrieval per query (FR-002)
5. Concept explorer + citation analyzer run in parallel (FR-003 + FR-004)
6. `analysis_join` barrier fans them in
7. Conditional router checks for methods → `assumption_auditor` runs (FR-005)
8. `connection_explorer` runs (FR-006)
9. `synthesis_writer` produces the final report (FR-007)
10. Citation validation runs; result attached to `evidence_metadata`
11. Cache stores the result keyed on (query_hash, model_config)
12. Markdown report emitted to stdout (or to `-o PATH` / as JSON via `--json`)

**Post-condition**: cached report exists; report on stdout/file matches NFR-001 latency bound.

### UC-002 — Cached repeat query

**Actor**: same
**Pre-condition**: a prior identical query was cached within `CACHE_TTL_HOURS`
**Trigger**: same query string, same `PLANNING_MODEL` + `SYNTHESIS_MODEL`
**Main flow**:

1. CLI computes query hash + config hash
2. Cache lookup hits
3. Cached report returned immediately (NFR-002)

**Post-condition**: result returned in ≤ 50ms; no MCP calls or LLM calls made.

### UC-003 — Long-running query with per-node timeout

**Actor**: same
**Pre-condition**: research-kb is partially degraded; one node exceeds its timeout budget
**Trigger**: a query whose search phase happens to exceed 120s
**Main flow**:

1. literature_search invokes `research_kb_search`
2. The call exceeds the per-node timeout (NFR-003)
3. `NodeTimeoutError` is raised inside `_make_resilient_node`
4. LangGraph marks the node as failed but the graph proceeds with the partial state already accumulated
5. Synthesis sees an empty/sparse `state.search_results` and produces a degraded report explicitly noting the gap (per FR-007's gap-honesty requirement)

**Post-condition**: user receives a report with explicit gap acknowledgment rather than a hard error.

### UC-004 — Structured-JSON consumption (programmatic)

**Actor**: a downstream automation script
**Pre-condition**: same as UC-001
**Trigger**: `research-agent --json "What are the assumptions of DML?"`
**Main flow**:

1. Same pipeline as UC-001
2. Final emission is a single JSON object on stdout containing: `query`, `report`, `confidence_assessment`, `evidence_metadata`, `sub_tasks`, `search_results` (full list), `concepts`, `citations`, `assumption_audits`, `connection_explanations`
3. Logs go to stderr (does not pollute stdout)

**Post-condition**: stdout is parseable as JSON; pipe-friendly for downstream tools.

---

## 8. Constraints, Assumptions, Dependencies

### 8.1 Constraints

- Python 3.11+ (matches CI matrix)
- Anthropic API rate limits cap synthesis throughput at roughly 30 reports/minute under default Sonnet usage
- SQLite cache uses a single file; concurrent writes from multiple processes are serialized
- Stdio transport spawns research-kb as a subprocess; this fails if research-kb's `.venv` is not built or `RESEARCH_KB_PYTHON` is misconfigured

### 8.2 Assumptions

- research-kb returns well-formed JSON for `output_format=json` tool variants
- The Anthropic models referenced (`claude-haiku-4-5-20251001` for planning, `claude-sonnet-4-6` for synthesis) remain available; if deprecated, the `PLANNING_MODEL` / `SYNTHESIS_MODEL` env vars allow swap without code change
- The MCP protocol versioning negotiated by the `mcp` Python package is forward-compatible

### 8.3 Dependencies

| Dependency | Purpose | Failure mode |
|------------|---------|--------------|
| `langgraph` | StateGraph orchestration | Pipeline cannot run |
| `langchain` (chat-model dispatch) | Provider-agnostic LLM calls | Pipeline cannot run |
| `anthropic` SDK (via langchain) | Default LLM provider | Synthesis fails; alternative provider via env |
| `mcp` (Python SDK) | MCP transport | KB calls fail |
| `httpx` | HTTP MCP transport | HTTP mode fails; stdio still works |
| `tenacity` | Retry-with-backoff | Transient errors no longer auto-retry |
| `pydantic` + `pydantic-settings` | State validation + config loading | Graph cannot construct state |
| `langsmith` | Optional tracing | No-op if not configured |
| `gitleaks` (pre-commit) | Secret scanning | Pre-commit blocks commit |
| `ruff`, `mypy` (pre-commit + CI) | Lint + type-check | CI blocks merge |

---

## 9. Acceptance Criteria

The system is considered to meet this SRS when:

1. All 7 FRs have at least one passing test in `tests/` (see § 10 traceability matrix)
2. All 9 NFRs are either:
   - Quantitatively validated (e.g., NFR-001 via `docs/eval_baselines.md`, NFR-005 via CI gate)
   - Architecturally enforced (e.g., NFR-003 via `NODE_TIMEOUTS` config, NFR-009 via gitleaks)
3. CI is green on the `main` branch:
   - `lint` passes (ruff)
   - `typecheck` passes (mypy)
   - `test` matrix passes (Python 3.11/3.12/3.13) with `--cov-fail-under=80`
4. The `research-agent --help` output matches § 5.2 exactly
5. The `README.md` § Architecture diagrams + tool table match the wired graph in `src/research_agent/graph.py`

---

## 10. Traceability Matrix

| Requirement | Source files | Test files | Use case |
|------------|-------------|------------|----------|
| FR-001 | `nodes/query_planner.py` | `tests/test_nodes.py`, `tests/test_graph.py` | UC-001 step 3 |
| FR-002 | `nodes/literature_search.py`, `mcp_client.py` (search + fast_search) | `tests/test_nodes.py`, `tests/test_mcp_client.py` | UC-001 step 4 |
| FR-003 | `nodes/concept_explorer.py`, `mcp_client.py` (get_concept, graph_neighborhood, find_similar_concepts, cross_domain_concepts) | `tests/test_nodes.py` | UC-001 step 5 (parallel) |
| FR-004 | `nodes/citation_analyzer.py`, `mcp_client.py` (citation_network, biblio_coupling) | `tests/test_nodes.py` | UC-001 step 5 (parallel) |
| FR-005 | `nodes/assumption_auditor.py`, `graph.py:_should_audit_assumptions`, `mcp_client.py` (audit_assumptions) | `tests/test_nodes.py`, `tests/test_graph.py` | UC-001 step 7 |
| FR-006 | `nodes/connection_explorer.py`, `mcp_client.py` (explain_connection) | `tests/test_connection_explorer.py` | UC-001 step 8 |
| FR-007 | `nodes/synthesis.py`, `mcp_client.py` (get_source enrichment) | `tests/test_synthesis.py` | UC-001 step 9 |
| NFR-001 | `graph.py` (parallel fan-out + analysis_join), `docs/eval_baselines.md` (calibration) | `evals/test_synthesis_eval.py` | UC-001 latency budget |
| NFR-002 | `cache.py` | `tests/test_cache.py` | UC-002 |
| NFR-003 | `config.py:NODE_TIMEOUTS`, `graph.py:_make_resilient_node` | `tests/test_graph.py` | UC-003 |
| NFR-004 | `cli.py` (`--json` flag) | `tests/test_cli.py` | UC-004 |
| NFR-005 | `.github/workflows/ci.yml` (`--cov-fail-under=80`) | CI gate | All UCs |
| NFR-006 | `mcp_client.py` (`_connect_stdio`, `_connect_http`) | `tests/test_mcp_client.py` | All UCs |
| NFR-007 | `llm.py:create_llm`, `config.py:ModelConfig` | `tests/test_llm.py` | All UCs |
| NFR-008 | `mcp_client.py` (`@retry` with exponential backoff) | `tests/test_mcp_client.py` | UC-001 fault tolerance |
| NFR-009 | `.pre-commit-config.yaml` (gitleaks) | Pre-commit gate | Repo hygiene |

Total test inventory: ~451 tests across 14 test modules; CI enforces ≥ 80% line coverage with current observed at ~96%.

---

## Appendix A — Document Lifecycle

This SRS is a living document. Changes to the system that affect observable behavior should be reflected here in the same commit (or a follow-up commit). The `Status:` field on the front-matter header indicates document maturity:

- **Initial draft** — written from existing code; some sections may need refinement after first stakeholder review
- **Reviewed** — at least one round of review applied
- **Stable** — used as the reference for change-control

A future commit may add a CHANGELOG section tracking material requirement changes.
