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

# Docker (multi-service with research-kb)
docker-compose up                         # Start agent + research-kb
docker-compose run agent "Query here"     # One-shot query

# Installation
pip install -e ".[dev]"                   # Editable install with dev deps
```

## Architecture

### Pipeline

```
Query Planner → Literature Search → Concept Explorer → Citation Analyzer
                                                      ↘ Assumption Auditor (conditional)
             → Synthesis Writer
```

Conditional edge: Citation Analyzer skips Assumption Auditor if no methods were
identified in the sub-tasks (`state.py:31` — `SubTask.methods_to_audit`).

### Design Decisions

- **Pydantic BaseModel state** (not TypedDict) for richer type support and validation (`state.py:113`)
- **Closure-injected dependencies** — config and MCP client injected via closure,
  not in graph state (`graph.py:60-92`)
- **Provider-agnostic LLM dispatch** — `create_llm()` in `llm.py` wraps `init_chat_model`; model names auto-resolve providers (`config.py:20-36`)
- **MCP transport abstraction** — stdio (local dev) or HTTP via streamable_http_client (Docker) (`mcp_client.py:52-57`)

### Source Layout

```
src/research_agent/
├── cache.py            # SQLite report cache (query hash + TTL, context manager)
├── cli.py              # CLI entry point (argparse + asyncio.run)
├── config.py           # ModelConfig, MCPConfig, AgentConfig (frozen dataclasses)
├── exceptions.py       # ResearchAgentError hierarchy
├── graph.py            # LangGraph StateGraph — build_graph() + run_research()
├── llm.py              # Provider-agnostic LLM factory (init_chat_model wrapper)
├── mcp_client.py       # ResearchKBClient — 7-tool MCP wrapper
├── state.py            # ResearchState + typed sub-dataclasses
└── nodes/
    ├── query_planner.py      # Decompose query into sub-tasks
    ├── literature_search.py  # Hybrid search via MCP
    ├── concept_explorer.py   # Knowledge graph exploration
    ├── citation_analyzer.py  # Citation network analysis
    ├── assumption_auditor.py # Method assumption auditing
    └── synthesis.py          # Final report generation
```

## Research-KB Integration

Seven MCP tools exposed via `ResearchKBClient` (`mcp_client.py:7-14`):

| Method             | MCP Tool                        | Signal                              |
|--------------------|---------------------------------|-------------------------------------|
| `search`           | `research_kb_search`            | BM25 + vector + graph + PageRank    |
| `fast_search`      | `research_kb_fast_search`       | Vector-only (~200ms)                |
| `get_concept`      | `research_kb_get_concept`       | Knowledge graph concept details     |
| `graph_neighborhood` | `research_kb_graph_neighborhood` | N-hop concept traversal          |
| `citation_network` | `research_kb_citation_network`  | Citing/cited-by chains              |
| `biblio_coupling`  | `research_kb_biblio_coupling`   | Jaccard similarity on shared refs   |
| `audit_assumptions`| `research_kb_audit_assumptions` | Method assumption documentation     |

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
- `CACHE_ENABLED` — Enable/disable report cache (default: `true`)
- `CACHE_DB_PATH` — SQLite cache file path (default: `~/.cache/research-agent/cache.db`)
- `CACHE_TTL_HOURS` — Cache expiry in hours (default: `24`)
