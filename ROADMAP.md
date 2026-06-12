# Roadmap

> **Archived (2026-06-12)** — roadmap not pursued; see the README archive banner.

Near-term and medium-term improvements for the research-agent pipeline.

---

## Near-Term

### ~~Architecture Diagram~~ *(completed)*

Added two Mermaid diagrams to README.md: pipeline flow with model-tier coloring and conditional routing, plus per-node MCP tool mapping.

### ~~Persistent Session Memory~~ *(completed)*

SQLite-backed cache keyed on query hash + config parameters (model, limits). Repeated queries skip the full pipeline and return cached reports in milliseconds. TTL-based invalidation (default 24h) with `--clear-cache` for manual refresh. `--no-cache` flag bypasses cache for a single query.

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
