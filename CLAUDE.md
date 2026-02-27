# Research Agent

Multi-agent research analysis system powered by LangGraph + MCP. Decomposes research
questions into sub-tasks, searches a causal inference knowledge base, explores concept
graphs, analyzes citation networks, audits statistical assumptions, and synthesizes
a structured report.

## Commands

```bash
# Testing
pytest                                    # Run all tests (unit only)
pytest tests/test_mcp_client.py           # MCP client tests only
pytest --cov=research_agent               # With coverage
pytest -m integration -v                  # Integration tests (requires live MCP + API key)

# Code quality
ruff check src/ tests/                    # Lint
ruff format src/ tests/                   # Format

# Running
research-agent "What are the assumptions of double machine learning?"
research-agent --verbose -o report.md "Compare DML and IV"
research-agent --stream "What is DML?"    # Stream progress to stderr
research-agent --no-cache "Query"         # Bypass cache for this query
research-agent --clear-cache              # Clear all cached reports and exit
research-agent --health-check             # Check MCP connection health and exit
research-agent --json "Query"             # Structured JSON output (report + metadata)

# Docker (multi-service with research-kb)
docker-compose up                         # Start agent + research-kb
docker-compose run agent "Query here"     # One-shot query

# Installation
pip install -e ".[dev]"                   # Editable install with dev deps
```

## Architecture

### Pipeline

```
query_planner → literature_search
              → {concept_explorer, citation_analyzer}  (parallel fan-out)
              → analysis_join  (sync barrier)
              → [assumption_auditor]  (conditional)
              → connection_explorer
              → synthesis
```

Parallel fan-out: concept_explorer and citation_analyzer run concurrently via
LangGraph native fan-out, joined by analysis_join barrier.

Conditional edge: analysis_join skips assumption_auditor → connection_explorer if no
methods were identified (planner-specified or auto-discovered from graph neighbors).

### Design Decisions

- **Pydantic BaseModel state** (not TypedDict) for richer type support and validation (`state.py:113`)
- **Closure-injected dependencies** — config and MCP client injected via closure,
  not in graph state (`graph.py:72-107`)
- **Provider-agnostic LLM dispatch** — `create_llm()` in `llm.py` wraps `init_chat_model`; model names auto-resolve providers (`config.py:20-36`)
- **MCP transport abstraction** — stdio (local dev) or HTTP via streamable_http_client (Docker) (`mcp_client.py:52-57`)
- **Parallel fan-out** — LangGraph native (not asyncio.gather in a node) with `_last_value` reducer on `current_node` (`state.py:15-21`)
- **Auto-discovered methods** — concept_explorer extracts ASSUMPTION/THEOREM neighbors → `discovered_methods` (capped at 3) (`concept_explorer.py:33`)
- **Per-node resilience** — `_make_resilient_node()` wraps each node with `asyncio.timeout` and error handling; critical nodes (planner, search, synthesis) propagate errors, others degrade gracefully (`graph.py:162-220`)
- **Timing instrumentation** — wrapper injects `node_duration_ms` into every NodeUpdate; `StreamEvent` carries `duration_ms` and `timestamp` for per-node and total timing (`graph.py:390-406`)

### Source Layout

```
src/research_agent/
├── cache.py            # SQLite report cache (query hash + TTL, context manager)
├── cli.py              # CLI entry point (argparse + asyncio.run)
├── config.py           # ModelConfig, MCPConfig, AgentConfig (frozen dataclasses)
├── exceptions.py       # ResearchAgentError hierarchy
├── graph.py            # LangGraph StateGraph — build_graph() + run_research()
├── llm.py              # Provider-agnostic LLM factory (init_chat_model wrapper)
├── mcp_client.py       # ResearchKBClient — 13-tool MCP wrapper
├── state.py            # ResearchState + typed sub-dataclasses
└── nodes/
    ├── query_planner.py        # Decompose query into sub-tasks
    ├── literature_search.py    # Hybrid search via MCP (domain-aware)
    ├── concept_explorer.py     # Knowledge graph exploration + method discovery
    ├── citation_analyzer.py    # Citation network analysis
    ├── assumption_auditor.py   # Method assumption auditing (domain-scoped)
    ├── connection_explorer.py  # Concept path tracing via explain_connection
    └── synthesis.py            # Final report generation
```

## Research-KB Integration

Thirteen MCP tools exposed via `ResearchKBClient` (`mcp_client.py`):

| Method               | MCP Tool                           | Signal                              |
|----------------------|------------------------------------|-------------------------------------|
| `search`             | `research_kb_search`               | BM25 + vector + graph + PageRank    |
| `fast_search`        | `research_kb_fast_search`          | Vector-only (~200ms)                |
| `get_concept`        | `research_kb_get_concept`          | Knowledge graph concept details     |
| `graph_neighborhood` | `research_kb_graph_neighborhood`   | N-hop concept traversal             |
| `citation_network`   | `research_kb_citation_network`     | Citing/cited-by chains              |
| `biblio_coupling`    | `research_kb_biblio_coupling`      | Jaccard similarity on shared refs   |
| `audit_assumptions`  | `research_kb_audit_assumptions`    | Method assumption documentation     |
| `explain_connection` | `research_kb_explain_connection`   | Concept path tracing with evidence  |
| `get_source`         | `research_kb_get_source`           | Full source metadata + chunks       |
| `find_similar_concepts` | `research_kb_find_similar_concepts` | Embedding-based concept discovery |
| `cross_domain_concepts` | `research_kb_cross_domain_concepts` | Cross-domain concept bridging     |
| `list_domains`       | `research_kb_list_domains`         | Available KB domains                |
| `stats`              | `research_kb_stats`                | Corpus size and composition         |

Each method requests JSON output (`output_format='json'`) and returns a JSON string.
Agent nodes parse with `json.loads()` and fall back to markdown parsing on failure.

## Configuration

Environment variables (see `.env.example`):
- `ANTHROPIC_API_KEY` — Required
- `MCP_TRANSPORT` — `stdio` (default) or `http`
- `RESEARCH_KB_PATH` — Path to research-kb repo (stdio mode)
- `RESEARCH_KB_URL` — HTTP endpoint (Docker mode, default `http://research-kb:8000`)
- `MCP_PATH` — MCP endpoint path appended to HTTP URL (default `/mcp`)
- `PLANNING_MODEL`, `SYNTHESIS_MODEL` — Override model selection
- `MAX_SEARCH_RESULTS`, `MAX_CONCEPTS`, `MAX_CITATIONS` — Limit tuning
- `TOP_RESULTS_FOR_CITATIONS` — Number of top results for citation analysis (default: `5`)
- `CACHE_ENABLED` — Enable/disable report cache (default: `true`)
- `CACHE_DB_PATH` — SQLite cache file path (default: `~/.cache/research-agent/cache.db`)
- `CACHE_TTL_HOURS` — Cache expiry in hours (default: `24`)
- `MAX_SIMILAR_CONCEPTS` — Cap on embedding-similar concepts per query (default: `5`)
- `ENABLE_CROSS_DOMAIN` — Enable cross-domain concept bridging (default: `true`)
- `NODE_TIMEOUTS` — Per-node timeout JSON (default: planner 60s, search 120s, synthesis 180s)
