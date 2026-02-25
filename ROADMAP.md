# Roadmap

Near-term and medium-term improvements for the research-agent pipeline.

---

## Near-Term

### ~~Architecture Diagram~~ *(completed)*

Added two Mermaid diagrams to README.md: pipeline flow with model-tier coloring and conditional routing, plus per-node MCP tool mapping.

### Persistent Session Memory

SQLite-backed cache keyed on query hash. Repeated or similar queries skip the full pipeline and return cached reports. Includes TTL-based invalidation when the knowledge base is updated.

**Design**: `(query_hash, timestamp, report_json)` table. Hash includes config parameters (model, max_results) to avoid stale cache hits across config changes.

---

## Medium-Term

### Tool-Call Planning Mode

Let the planner LLM emit MCP tool calls directly instead of using hardcoded node sequences. The agent interprets tool calls, executes them against research-kb, and feeds results back. More flexible for novel query types that don't fit the current 6-node pipeline.

**Trade-off**: Greater flexibility vs harder deterministic testing. Likely requires a hybrid approach -- structured nodes for common patterns, tool-call mode for exploratory queries.

### Multi-KB Routing

Domain classifier routes queries to different MCP knowledge bases. Example: causal inference queries go to research-kb, software engineering queries go to a separate docs-kb. The synthesis node merges evidence across sources.

**Prerequisite**: A second knowledge base with MCP server support.

---

## Future

### Web Interface

FastAPI backend with HTMX streaming frontend. Stream node progress in real-time as the pipeline executes. Display the final report with collapsible evidence sections.

**Stack**: FastAPI + SSE for streaming, HTMX for progressive enhancement, minimal JS.
