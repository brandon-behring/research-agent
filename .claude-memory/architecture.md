# Research-Agent Architecture Reference

## 1. Pipeline Topology

```
query_planner → literature_search → concept_explorer → citation_analyzer
                                                      ↘ assumption_auditor (conditional)
             → synthesis_writer → END
```

**Conditional edge** (`graph.py:43-58`): `_should_audit_assumptions()` checks if any
`SubTask.methods_to_audit` exist. If empty → skip directly to synthesis.

**Entry point**: `graph.py:107` — `graph.set_entry_point("query_planner")`

## 2. State Model

### 2.1 Immutable sub-models (`state.py:17-107`)

All use `ConfigDict(frozen=True)`:

| Model | Fields | Used By |
|-------|--------|---------|
| `SubTask` | description, search_queries, concepts_to_explore, methods_to_audit | query_planner |
| `SearchResult` | title, content, source_id, score, authors, year, chunk_id | literature_search |
| `ConceptInfo` | concept_id, name, concept_type, description, relationships, neighborhood_summary | concept_explorer |
| `CitationInfo` | source_id, source_title, citing, cited_by, similar_papers | citation_analyzer |
| `AssumptionAudit` | method_name, assumptions, raw_output | assumption_auditor |

### 2.2 Mutable graph state (`state.py:113-151`)

`ResearchState(BaseModel)` — NOT frozen (LangGraph reconstructs via `schema(**dict)`).
All fields have defaults → graph can start at any node for testing.

### 2.3 Typed node return (`state.py:156-172`)

`NodeUpdate(TypedDict, total=False)` — all fields optional for partial updates.
Nodes return `NodeUpdate` dicts; LangGraph merges into `ResearchState`.

## 3. Dependency Injection Pattern

**Closure injection** (`graph.py:77-93`): Config and MCP client are NOT in graph state.
Instead, `build_graph(config, mcp)` creates inner async functions that close over these:

```python
async def _literature_search(state: ResearchState) -> NodeUpdate:
    return await literature_search(state, config, mcp)
```

**Why**: Keeps state schema clean — only research data flows through the graph.

## 4. LLM Dispatch

### 4.1 Factory (`llm.py:14-35`)

`create_llm(model, max_tokens, temperature)` → `BaseChatModel`

Uses `init_chat_model()` from langchain — auto-resolves provider from model name:
- `claude-*` → Anthropic
- `gpt-*` → OpenAI
- `ollama/*` → Ollama
- Explicit prefix: `openai/gpt-4o`

### 4.2 Structured output

Both planner and synthesis use `.with_structured_output(PydanticModel)` for guaranteed
schema compliance. No fragile regex/JSON parsing for LLM responses.

### 4.3 Model selection (`config.py:20-38`)

- **Planning** (Haiku): Fast, cheap routing — `claude-haiku-4-5-20251001`
- **Synthesis** (Sonnet): Strong reasoning for final report — `claude-sonnet-4-6`

## 5. MCP Client Architecture

### 5.1 Transport abstraction (`mcp_client.py:72-76`)

Two transports:
- **stdio**: Spawn research-kb as subprocess (local dev). Uses `StdioServerParameters`
- **http**: Connect to running service (Docker). Uses `streamable_http_client`

### 5.2 Connection lifecycle (`mcp_client.py:63-102`)

`AsyncExitStack` for proper LIFO cleanup ordering: session exits before transport.
Known MCP SDK issue: stdio transport can raise `RuntimeError` during cleanup after
the `ClientSession`'s task group has exited — logged and suppressed (`mcp_client.py:101`).

### 5.3 Resilience (`mcp_client.py:171-176`)

```python
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10),
       retry=retry_if_exception_type(MCPToolError), reraise=True)
@traceable(name="mcp_call", run_type="tool")
async def _call_tool(self, name, arguments) -> str:
```

### 5.4 Tool methods (`mcp_client.py:211-382`)

All 7 methods follow the same pattern:
1. Build args dict with `output_format='json'`
2. Call `_call_tool(tool_name, args)`
3. Return raw JSON string (parsing is the node's responsibility)

## 6. Parsing Strategy

### 6.1 Generic utility (`parsing.py:19-46`)

`parse_json_first(raw, json_parser, markdown_parser, context="")` → `T`

Tries JSON parser first; on `JSONDecodeError | KeyError | TypeError`, falls back to
markdown parser with a warning log.

### 6.2 Node-specific parsers

Each node module contains its own `_parse_*_json()` and `_parse_*_markdown()` functions.
These are passed to `parse_json_first()`. The JSON parsers build Pydantic models from
dicts; markdown parsers use regex extraction from formatted text.

## 7. Caching Architecture

### 7.1 Cache key (`cache.py:45-78`)

SHA-256 of: `{normalized_query, max_search_results, max_concepts, max_citations, synthesis_model}`

Excludes: transport, timeout, planning model (don't affect output).

### 7.2 ReportCache (`cache.py:110-277`)

- SQLite-backed, WAL journal mode
- Context manager protocol + null-object pattern (`enabled=False` → all no-ops)
- **Graceful degradation**: every public method catches `sqlite3.Error` and `OSError`,
  logs warning, returns None/no-op. Callers never handle cache exceptions.
- Corrupt entry detection: on `JSONDecodeError` during `get()`, deletes the entry

### 7.3 CLI integration (`cli.py:157-208`)

Single `with ReportCache(...)` block wraps both cache check and pipeline execution.
Cache writes happen inside the same context manager — no resource leak risk.

## 8. Streaming Architecture

### 8.1 StreamEvent (`graph.py:164-176`)

```python
@dataclass
class StreamEvent:
    event_type: str  # "node_end" | "report_chunk" | "complete"
    node_name: str
    data: str
```

### 8.2 stream_research() (`graph.py:200-251`)

Uses LangGraph's `astream(stream_mode="updates")` which yields `{node_name: update_dict}`
after each node finishes. Wrapped in an `AsyncGenerator[StreamEvent, None]`.

## 9. Error Hierarchy

```
ResearchAgentError (base)
├── MCPConnectionError    — cannot connect to research-kb
├── MCPToolError          — MCP tool call failed (includes tool_name + detail)
├── LLMParsingError       — structured output parsing failed
├── PlannerError          — query decomposition failed
├── SearchError           — literature search failed
└── SynthesisError        — report generation failed
```

All exceptions are catchable as `ResearchAgentError` for top-level handling (`cli.py:181`).

## 10. Configuration Layers

### 10.1 pydantic-settings (`config.py`)

Three nested frozen settings classes:
- `ModelConfig` — planning/synthesis model names
- `MCPConfig` — transport, paths, URLs
- `AgentConfig` — top-level with limits, cache settings, nested Model+MCP configs

All fields load from environment variables via `alias`. Constructor overrides work for tests.

### 10.2 Validation

- `max_search_results`: 1-50 (default 10)
- `max_concepts`: 1-50 (default 15)
- `max_citations`: 1-50 (default 20)
- `synthesis_timeout`: 30-300s (default 120)
- `cache_ttl_hours`: >0, <=720 (default 24)

## 11. Testing Architecture

### 11.1 Fixture strategy (`conftest.py`, 351 lines)

- Shared fixtures for MCP client mocking, config creation, state building
- `mock_mcp_client` fixture returns an `AsyncMock` with pre-configured return values
- Node-level test isolation: each node gets state + config + mocked MCP

### 11.2 Test files

| File | Tests | Focus |
|------|-------|-------|
| `test_nodes.py` | Node logic (all 6 nodes) | JSON/markdown parsing, state updates |
| `test_parsers.py` | Parse functions | JSON-first + markdown fallback per node |
| `test_mcp_client.py` | MCP client | Connection, retry, error handling |
| `test_models.py` | Pydantic models | Validation, immutability, defaults |
| `test_synthesis.py` | Synthesis specifics | Evidence context building, metadata |
| `test_exceptions.py` | Error hierarchy | Inheritance, attributes, catchability |
| `test_llm.py` | LLM factory | Provider dispatch, parameter passing |
| `test_parsing_util.py` | parse_json_first | Generic fallback behavior |
| `test_integration.py` | Integration markers | Full pipeline (requires live MCP) |

### 11.3 Coverage: 91% (258 tests, all passing)
