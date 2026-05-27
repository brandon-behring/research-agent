# Research-Agent Development Log

## Phase 1: Core Pipeline (Commits: 05ff689 → af01c8e)

**Objective**: Establish working LangGraph pipeline with MCP integration.

### Work Completed
- Designed 6-node pipeline: query_planner → literature_search → concept_explorer → citation_analyzer → assumption_auditor → synthesis
- Implemented Pydantic BaseModel state schema with frozen sub-models
- Built MCP client with stdio transport for local development
- Created conditional edge: skip assumption_auditor when no methods identified
- Initial test suite with model validation tests
- Mermaid architecture diagrams in README

### Key Decisions
- Pydantic BaseModel over TypedDict for richer validation
- Closure injection for config/MCP (not in graph state)
- Sequential pipeline (not parallel) for simplicity in v1

## Phase 2: Production Features (Commits: a77602c → 48ceb6f)

**Objective**: Add caching, streaming, CLI, Docker, provider-agnostic LLM.

### Work Completed
- SQLite report cache with SHA-256 keys, TTL expiration, graceful degradation
- Streaming support via LangGraph `astream(stream_mode="updates")` + StreamEvent
- CLI with argparse: `--verbose`, `--output`, `--stream`, `--no-cache`, `--clear-cache`
- Docker Compose with agent + research-kb services
- Provider-agnostic LLM via `init_chat_model` (supports Anthropic, OpenAI, Ollama)
- HTTP transport for MCP (Docker mode)
- pydantic-settings for environment variable configuration

### Key Decisions
- Cache key excludes transport/timeout/planning model (only output-affecting params)
- Null-object pattern for disabled cache (no code branches needed)
- WAL journal mode for SQLite (concurrent reads)
- Haiku for planning, Sonnet for synthesis (cost/quality tradeoff)

## Phase 3: Production Safety (Current — Uncommitted)

**Objective**: Robust error handling, comprehensive parsing, production-grade test suite.

### Work Completed
- Created `parsing.py` with generic `parse_json_first()` utility
- Refactored all nodes to use JSON-first parsing with markdown fallback
- Built comprehensive markdown parsers for every node type
- Expanded exception hierarchy: 6 specific exception classes
- Added retry with exponential backoff on MCPToolError (tenacity)
- Added LangSmith `@traceable` on MCP calls
- Evidence quality metadata in synthesis context (score distributions, year ranges)
- **Test expansion**: 258 tests total (up from ~120)
  - `test_parsers.py`: 708 lines — exhaustive JSON + markdown parsing tests
  - `test_mcp_client.py`: +145 lines — retry, error handling, connection tests
  - `test_synthesis.py`: 196 lines — evidence context building, metadata
  - `test_exceptions.py`: 68 lines — hierarchy and attribute tests
  - `test_llm.py`: 36 lines — factory dispatch tests
  - `test_parsing_util.py`: 68 lines — generic fallback behavior
- **Coverage**: 91% (target was 80%+)
- **Lint**: All checks passed (ruff check + ruff format)

### Files Changed (17 total)
**Modified (13):**
- `.gitignore` — +6 lines
- `CLAUDE.md` — +3 lines
- `mcp_client.py` — +52/-changes (retry, tracing, error handling)
- All 6 node files — parsing refactors, error handling hardening
- `conftest.py` — expanded fixtures
- `test_mcp_client.py`, `test_nodes.py`, `test_parsers.py` — expanded tests

**New (4):**
- `src/research_agent/parsing.py` — generic parse utility
- `tests/test_exceptions.py` — exception hierarchy tests
- `tests/test_llm.py` — LLM factory tests
- `tests/test_parsing_util.py` — parse_json_first tests
- `tests/test_synthesis.py` — synthesis-specific tests

### Delta: +1,481 / -337 lines

## Current State (2026-02-26)

### Ready
- 258 tests passing
- 91% coverage
- Lint clean
- All Phase 3 work complete but uncommitted

### Uncommitted Work
17 files (13 modified + 4 new) totaling +1,481/-337 lines.
This is the Phase 3 production safety work — ready to commit.

## Potential Next Steps

### Near-term
1. **Commit Phase 3** — tests pass, lint clean, ready to ship
2. **Integration test** with live research-kb MCP server
3. **JSON output format** for search/concept/citation tools (flagged HIGH priority)

### Medium-term
4. **Parallel node execution** — literature_search and concept_explorer could run concurrently
5. **LangSmith evaluation harness** — automated quality scoring
6. **Rate limiting** — token/request budgets per pipeline run
7. **Retry at node level** — currently only MCP calls retry; node-level retry for LLM failures

### Architecture Considerations
- Pipeline is sequential; parallelizing lit_search + concept_explorer would reduce latency
- No prompt versioning system yet — prompts are inline in node modules
- No observability beyond logging + optional LangSmith
- Cache eviction is opportunistic (after writes); no background cleanup

## Codebase Metrics

| Metric | Value |
|--------|-------|
| Source lines | 3,131 |
| Test lines | 3,745 |
| Test count | 258 |
| Coverage | 91% |
| Modules | 12 (9 src + parsing.py + exceptions.py + llm.py) |
| Node modules | 6 |
| MCP tools | 7 |
| Pydantic models | 8 (7 state + 1 NodeUpdate TypedDict) |
| Exception classes | 6 |
| Commits | 10 on main |
