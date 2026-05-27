# Research-Agent Memory Index

> Last updated: 2026-05-21 | Status: **research-agent slated for archival; synthesis-kb is the successor**

## Current Direction (2026-05-21)

- [synthesis-kb planned](synthesis_kb_planned.md) — new sibling RAG to research-kb; plan at `~/.claude/plans/i-want-to-think-foamy-summit.md`
- [research-agent archived at M6](research_agent_archived.md) — capabilities re-home as synthesis-kb MCP tools
- [Epistemic separation required](feedback_epistemic_separation.md) — never conflate primary literature with synthesized claims
- [Pacing: one question per round](feedback_pacing_one_question.md) — for `/exploring-options` sessions, depth not breadth
- [User owns runpod-deploy](user_runpod_deploy.md) — Tier-2 GPU escalation path
- [brandon-behring.dev reference](reference_brandon_behring_dev.md) — Astro + Cloudflare Workers; tracked issue #1 on that repo for synthesis-kb integration

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
