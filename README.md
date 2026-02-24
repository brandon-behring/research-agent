# research-agent

![CI](https://github.com/brandonmbehring-dev/research-agent/actions/workflows/ci.yml/badge.svg)

Multi-agent research analysis system powered by [LangGraph](https://github.com/langchain-ai/langgraph) and the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/).

Given a research question, the agent decomposes it into sub-tasks, searches a knowledge base, explores concept graphs, analyzes citation networks, audits method assumptions, and produces a structured synthesis report.

## Architecture

```
User Question
  → [Query Planner]         — decomposes into sub-tasks (Haiku: fast routing)
  → [Literature Search]     — hybrid search: BM25 + vector + graph + PageRank
  → [Concept Explorer]      — knowledge graph traversal (2-hop neighborhoods)
  → [Citation Analyzer]     — citation networks + bibliographic coupling
  → [Assumption Auditor]    — method assumption documentation (conditional)
  → [Synthesis Writer]      — structured report with citations (Sonnet: deep reasoning)
```

### Design Decisions

1. **LangGraph over CrewAI/Agent SDK** — Vendor-agnostic, fine-grained state control, conditional routing, proven in production across multiple personal projects.

2. **MCP integration over direct DB calls** — Standard protocol decouples the agent from the knowledge backend. research-kb can be swapped for any MCP-compatible source.

3. **Separate agent from knowledge base** — Single responsibility: agent orchestrates, KB serves knowledge. Services scale independently.

4. **Haiku for planning, Sonnet for synthesis** — Cost/latency optimization. Fast model for routing decisions, powerful model for final output quality.

5. **TypedDict-free state (dataclass)** — Richer type support with defaults on every field. Each node returns a partial dict of updates — LangGraph merges automatically.

## Built on research-kb

This agent consumes [research-kb](https://github.com/brandonmbehring-dev/research-kb), a production knowledge base system I built with:

- **478 sources** in causal inference, time series, and RAG/LLM literature
- **307K+ concepts** in a knowledge graph with typed relationships
- **4-signal hybrid search**: BM25 full-text + BGE-large vectors + graph signals + PageRank citation authority
- **20 MCP tools** for search, concept exploration, citation analysis, and assumption auditing
- **2,040+ tests** with comprehensive CI/CD

The agent uses 7 of these tools:

| Tool | Purpose |
|------|---------|
| `research_kb_search` | Hybrid search (BM25 + vector + graph + PageRank) |
| `research_kb_fast_search` | Lightweight vector-only fallback (~200ms) |
| `research_kb_get_concept` | Retrieve concept details from knowledge graph |
| `research_kb_graph_neighborhood` | Explore related concepts within N hops |
| `research_kb_citation_network` | Find citing/cited-by chains |
| `research_kb_biblio_coupling` | Related papers via shared reference overlap |
| `research_kb_audit_assumptions` | Method assumption documentation |

## Quickstart

### Prerequisites

- Python 3.11+
- [research-kb](https://github.com/brandonmbehring-dev/research-kb) cloned and set up locally
- Anthropic API key

### Installation

```bash
git clone https://github.com/brandonmbehring-dev/research-agent.git
cd research-agent
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

### Configuration

```bash
cp .env.example .env
# Edit .env with your API key and research-kb path
```

**Environment variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | *(required)* | Anthropic API key |
| `MCP_TRANSPORT` | `stdio` | `stdio` (local) or `http` (Docker) |
| `RESEARCH_KB_PATH` | | Path to research-kb repo (stdio mode) |
| `RESEARCH_KB_URL` | `http://research-kb:8000` | HTTP endpoint (Docker mode) |
| `MCP_PATH` | `/mcp` | MCP endpoint path appended to HTTP URL |

### Run

```bash
# CLI (stdio transport — spawns research-kb subprocess)
research-agent "What are the assumptions of double machine learning?"

# With verbose logging
research-agent -v "Compare DML and instrumental variables"

# Stream progress to stderr as nodes complete
research-agent --stream "What are the assumptions of DML?"

# Save report to file
research-agent -o report.md "How does cross-fitting reduce bias?"
```

### Docker (HTTP transport)

```bash
docker-compose up
# Agent connects to research-kb via HTTP at http://research-kb:8000/mcp
docker-compose run agent "Query here"
```

## Example Queries

The agent handles any research topic in the knowledge base. Examples:

```
"What are the assumptions of double machine learning?"
"Compare instrumental variables and regression discontinuity designs"
"How does cross-fitting reduce regularization bias in semiparametric estimation?"
"What methods exist for heterogeneous treatment effect estimation?"
"Explain the relationship between propensity scores and inverse probability weighting"
```

If results are sparse for a topic, the synthesis report honestly identifies gaps:

> *"2 sources found on [topic]. The knowledge base has deeper coverage on causal inference methods — consider refining the query to focus on [related area]."*

## Testing

```bash
# Unit tests (default, no env vars needed — all MCP calls mocked)
pytest tests/ -v --cov=research_agent --cov-fail-under=80

# Integration tests (requires live research-kb + API key)
RESEARCH_KB_PATH=~/Claude/research-kb ANTHROPIC_API_KEY=sk-... \
    pytest tests/ -m integration -v

# Evals (separate, existing)
pytest evals/ -m eval --timeout=120 -v

# Run specific test module
pytest tests/test_nodes.py -v
```

## Project Structure

```
src/research_agent/
├── __init__.py
├── graph.py              # LangGraph StateGraph + conditional edges
├── state.py              # Dataclass state schema
├── config.py             # Model selection, MCP endpoint config
├── mcp_client.py         # Thin wrapper calling research-kb MCP tools
├── cli.py                # CLI entry point
└── nodes/
    ├── query_planner.py      # Decomposes question into sub-tasks
    ├── literature_search.py  # Hybrid search with fallback
    ├── concept_explorer.py   # Knowledge graph traversal
    ├── citation_analyzer.py  # Citation networks + biblio coupling
    ├── assumption_auditor.py # Method assumption documentation
    └── synthesis.py          # Final structured report
```

## License

MIT
