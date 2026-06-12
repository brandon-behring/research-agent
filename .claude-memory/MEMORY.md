# Research-Agent Memory Index

> Last updated: 2026-05-30 | Status: **synthesis-kb M1 + EVAL COMPLETE *and TIGHTENED* (agents domain), all committed.** `~/Claude/synthesis-kb` HEAD `bced8b4`, **9 commits**, tree clean, 49 tests green. KG (unchanged): 120 concepts / 181 anchors (100% grounded) / 120 embedded / bridges 119 shared_source (11 cross) + 52 citation_edge + 6 hub. **Eval now PANEL-adjudicated (3 independent blind source-grounded judges, replacing single-Opus-inline): L1 recall 72.7% → 75.0% PASS (κ=1.00, one defensible flip agent_evaluation); L3 citation-edge cross bridge precision 92% (unchanged); shared_source cross 45% → 36% stricter (κ=0.93). The handoff's open single-judge items are RESOLVED; "6 genuine gaps" → ~2 clean true-misses.** Details: [[synthesis-kb-eval]]. **HANDOFF: `~/Claude/synthesis-kb/docs/SESSION_HANDOFF_2026-05-30.md`** + review checklist `docs/MANUAL_REVIEW_2026-05-30.md` (the items Brandon should confirm; 2026-05-29 handoff kept as history). Next=**domain #2 ml_security** (verified viable: research-kb 148 ml_security sources + pi_portfolio dossiers in inbox). Plans: `~/.claude/plans/use-the-following-handof-mutable-sundae.md` (this session) + `...keen-matsumoto.md`. ⚠️ tool-result LAG persists — [[feedback-verify-against-source-of-truth]] (followed throughout this session).

## Current Direction (2026-05-27)

- [synthesis-kb planned](synthesis_kb_planned.md) — unified **all-domain** sibling KB to research-kb; **design-&-de-risk-first** (2026-05-27 reframe). Approved plan: `~/.claude/plans/dig-deep-and-understand-sparkling-cloud.md`; design foundation: `research-agent/docs/plans/active/synthesis_kb_migration_2026-05-21.md`
- [research-kb on desktop](research_kb_desktop.md) — **LIVE & mature on desktop** (2026-05-29: 3,446 src / 1.74M chunks / 36 domains); **cache→KB ingest DONE** (agents+ml_security live; ingester `research-kb/scripts/ingest_cache.py`); next = KG/concept layer (disk-gated)
- [Session handoff 2026-05-29](../../../../Claude/research-agent/docs/plans/active/2026-05-29_session-handoff.md) — deferred-items playbook (KG layer, dependabot, gotchas) for the next session; full state + commits
- [KB north-star](kb_north_star.md) — **the validated purpose**: cross-domain connections first + rigor backbone + "actually used"; DB is an eventual uncommitted scale path. Full doc: `~/.claude/plans/dig-deep-and-understand-sparkling-cloud.md`
- [User: ML generalist](user_ml_generalist.md) — mathematically-rigorous ML generalist; #1 goal = senior/staff ML role; broad interests, connections excite him most
- [Feedback: understand before executing](feedback_understand_before_executing.md) — deep goal-grounding + inconsistency-checking BEFORE execution; planning is his thinking space
- [Feedback: verify against source of truth](feedback_verify_against_source_of_truth.md) — tool results lag a round in this env; never narrate unseen results, verify against DB/file before reporting
- [Feedback: tests vs evals](feedback_tests_vs_evals.md) — unit tests (code correctness) FIRST, evals (deployed-KG usefulness) AFTER; mirror research-kb; don't enable MCP until tested
- [synthesis-kb eval design](synthesis_kb_eval.md) — M1 eval: 5-layer framework, full gold-set + tiers 0-2 + EVAL.md portfolio piece; verified reusable assets (ir-eval pkg etc.); audit truth (KG sound, 11 cross-dossier bridges, 9.3% orphan claims)
- [research-agent archived at M6](research_agent_archived.md) — capabilities re-home as synthesis-kb MCP tools
- [Epistemic separation required](feedback_epistemic_separation.md) — never conflate primary literature with synthesized claims
- [Regression-check on regather](feedback_regression_check_on_regather.md) — when re-gathering/migrating dossiers, compare new vs. old; v3 not authoritative until checked; never silently lose curatorial work
- [Pacing: one question per round](feedback_pacing_one_question.md) — for `/exploring-options` sessions, depth not breadth
- [User owns runpod-deploy](user_runpod_deploy.md) — Tier-2 GPU escalation path
- [brandon-behring.dev reference](reference_brandon_behring_dev.md) — Astro + Cloudflare Workers; tracked issue #1 on that repo for synthesis-kb integration
- [Memory synced via git](reference_memory_synced_via_git.md) — .claude-memory/ symlinked into repo; memory writes dirty the working tree (expected)

## Legacy reference (research-agent as-it-stands until M6 archival)

- **Project**: Multi-agent research analysis (LangGraph + MCP)
- **Source**: `src/research_agent/` — 13 modules (12 + connection_explorer)
- **Tests**: `tests/` — 417 tests, 9 test files + conftest
- **Python**: >=3.11 | Build: hatchling | Lint: ruff (100-char) | Types: mypy strict
- **Entry point**: `research-agent` CLI → `cli.py:main()`

## Architecture (→ [architecture.md](architecture.md))

Pipeline (Phase 7 — complete):
```
[kb_context]  # Pre-pipeline: list_domains + stats
  → query_planner → literature_search
  → {concept_explorer, citation_analyzer}  # parallel fan-out
  → analysis_join → [assumption_auditor] → connection_explorer → synthesis
```

- **Pre-pipeline KB context**: `_fetch_kb_context()` — `graph.py:126-153`
- **Parallel fan-out**: LangGraph native — `graph.py:131-136`
- **Resilient nodes**: `_make_resilient_node()` wraps each node — `graph.py:162-220`
- **Critical nodes**: `{query_planner, literature_search, synthesis}` propagate errors
- **Timing**: wrapper injects `node_duration_ms` into every NodeUpdate
- Pydantic BaseModel state — `state.py:113`
- Closure-injected deps — `graph.py:91-111`
- Provider-agnostic LLM via `init_chat_model` — `llm.py:14`
- MCP transport: stdio (local) or HTTP (Docker)

## Key Modules

| Module | Role |
|--------|------|
| `state.py` | Pydantic state schema (12 models + NodeUpdate + reducer) |
| `graph.py` | LangGraph StateGraph + streaming + resilient wrapper |
| `config.py` | pydantic-settings (env vars, frozen, node_timeouts) |
| `mcp_client.py` | 13-tool MCP wrapper + retry/tracing |
| `cache.py` | SQLite report cache (TTL, graceful degradation) |
| `cli.py` | argparse + asyncio.run + cache + health check + JSON output |
| `parsing.py` | JSON-first with markdown fallback |
| `llm.py` | `create_llm()` factory |
| `exceptions.py` | 7-class hierarchy (NodeTimeoutError) |
| `nodes/synthesis.py` | Finding model, Mermaid maps, next questions |

## MCP Tools (13 of research-kb's 20+)

| Method | Signal |
|--------|--------|
| `search` | BM25 + vector + graph + PageRank |
| `fast_search` | Vector-only (~200ms) |
| `get_concept` | Knowledge graph concept details |
| `graph_neighborhood` | N-hop concept traversal |
| `citation_network` | Citing/cited-by chains |
| `biblio_coupling` | Jaccard similarity on shared refs |
| `audit_assumptions` | Method assumption documentation |
| `explain_connection` | Concept path tracing with evidence |
| `get_source` | Full source metadata + chunks |
| `find_similar_concepts` | Embedding-based concept discovery |
| `cross_domain_concepts` | Cross-domain concept bridging |
| `list_domains` | Available KB domains |
| `stats` | Corpus size and composition |

## Development State — ALL PHASES COMPLETE

| Phase | Focus | Tests | Key Commits |
|-------|-------|-------|-------------|
| 1 | Core pipeline, 6 nodes, MCP client | ~100 | — |
| 2 | Streaming, caching, CLI, Docker | ~150 | — |
| 3 | Robust parsing, error hierarchy | 258 | — |
| 4 | Parallel fan-out, connection explorer, domain-aware | 298 | `5378f66` |
| 5 | 5 new MCP tools, KB context, enrichment | 362 | `ca1ab69` |
| 6 | Per-node timeouts, timing, health check, 96% cov | 400 | `dfac6fd` |
| 7 | JSON output, Finding model, Mermaid, error hints | 417 | `0db16e6` |

### Phase 7 Features
- `--json` flag: structured output (report + metadata + config)
- `Finding` model: per-finding confidence (high/medium/low) + source_count
- `concept_map_mermaid`: Mermaid graph syntax in reports
- `next_questions`: Follow-up research questions
- `_format_error()`: Actionable hints for MCPConnectionError, SearchError, NodeTimeoutError
- `--verbose`: Full traceback on errors

## Patterns & Conventions

- **Error handling**: Never silent. `ResearchAgentError` hierarchy
- **Parsing**: JSON-first via `parse_json_first()`, markdown fallback
- **Retry**: Exponential backoff on `MCPToolError` (3 attempts) via tenacity
- **Caching**: SHA-256 key. Graceful degradation on sqlite3 errors
- **Tracing**: LangSmith `@traceable` (no-op without API key)
- **Models**: Haiku for planning, Sonnet for synthesis
- **Streaming**: `StreamEvent` with `duration_ms` and `timestamp`
- **Resilience**: Non-critical nodes degrade; critical nodes propagate

## Testing Notes

- All tests mock MCP and LLM — no live services for `pytest`
- Integration tests (`-m integration`) require live MCP + API key
- Coverage: 96% (417 tests)

## Gotchas

- **Parallel state writes**: Use `_last_value` reducer on `current_node`
- **Auto-discovery in tests**: Default mock `graph_neighborhood` triggers auditor
- **NodeTimeoutError re-raise**: Explicit `except NodeTimeoutError: raise` in wrapper
- **KB tools return markdown**: Phase 5 tools parse markdown, not JSON
- **Finding schema change**: `key_findings` is now `list[Finding]` not `list[str]`
