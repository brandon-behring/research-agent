"""LangGraph StateGraph -- orchestrates the multi-agent research pipeline.

Architecture:
    query_planner → literature_search
                  → {concept_explorer, citation_analyzer}  (parallel fan-out)
                  → analysis_join  (sync barrier)
                  → [assumption_auditor]  (conditional)
                  → synthesis

    The graph uses:
    1. LangGraph native fan-out for concept_explorer + citation_analyzer
    2. A no-op join barrier (analysis_join) for state merging
    3. Conditional routing to skip assumption auditing if no methods exist

Design decisions:
    - Pydantic BaseModel state for rich validation and type support
    - Parallel analysis via LangGraph fan-out (not asyncio.gather in a node)
    - Conditional routing avoids unnecessary MCP calls
    - Config and MCP client injected via closure (not in state)
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from research_agent.config import AgentConfig
from research_agent.exceptions import MCPToolError, NodeTimeoutError, PlannerError
from research_agent.mcp_client import ResearchKBClient
from research_agent.nodes.assumption_auditor import assumption_auditor
from research_agent.nodes.citation_analyzer import citation_analyzer
from research_agent.nodes.concept_explorer import concept_explorer
from research_agent.nodes.connection_explorer import connection_explorer
from research_agent.nodes.literature_search import literature_search
from research_agent.nodes.query_planner import query_planner
from research_agent.nodes.synthesis import synthesis_writer
from research_agent.state import NodeUpdate, ResearchState

logger = logging.getLogger(__name__)


def _should_audit_assumptions(state: ResearchState) -> str:
    """Route to assumption auditor only if methods were identified.

    Checks both planner-specified methods and auto-discovered methods
    from concept_explorer's graph traversal.

    Args:
        state: Current state after analysis join.

    Returns:
        'assumption_auditor' if methods exist, 'connection_explorer' otherwise.
    """
    methods = [m for task in state.sub_tasks for m in task.methods_to_audit]
    methods.extend(state.discovered_methods)

    if methods:
        return "assumption_auditor"
    return "connection_explorer"


def _parse_domain_list(raw: str) -> list[str]:
    """Extract domain IDs from list_domains markdown response.

    Parses markdown tables with domain names in the first column.
    Expected format::

        | Domain | Sources | Concepts |
        |--------|---------|----------|
        | causal_inference | 312 | 145 |

    Args:
        raw: Markdown string from list_domains.

    Returns:
        List of domain ID strings (e.g., ['causal_inference', 'time_series']).
    """
    domains: list[str] = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line.startswith("|") or "---" in line:
            continue
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if cells and cells[0] and cells[0].lower() not in ("domain", ""):
            domains.append(cells[0])
    return domains


def _parse_stats_summary(raw: str) -> str:
    """Extract concise corpus summary from stats markdown response.

    Looks for **Sources:** and **Chunks:** bullet lines and combines
    them into a compact summary string.

    Args:
        raw: Markdown string from stats.

    Returns:
        Summary like '495 sources, 226,432 chunks' or empty string on failure.
    """
    sources = ""
    chunks = ""
    for line in raw.split("\n"):
        m = re.search(r"\*\*Sources:\*\*\s*([\d,]+)", line)
        if m:
            sources = m.group(1)
        m = re.search(r"\*\*Chunks:\*\*\s*([\d,]+)", line)
        if m:
            chunks = m.group(1)
    parts = []
    if sources:
        parts.append(f"{sources} sources")
    if chunks:
        parts.append(f"{chunks} chunks")
    return ", ".join(parts)


async def _fetch_kb_context(mcp: ResearchKBClient) -> tuple[list[str], str]:
    """Fetch domain list and corpus stats from KB (pre-pipeline).

    Both calls are lightweight (~100ms total, no LLM). Failures are
    gracefully handled — the pipeline runs fine without this context.

    Args:
        mcp: Connected MCP client.

    Returns:
        Tuple of (domain list, stats summary string).
    """
    kb_domains: list[str] = []
    kb_stats_summary: str = ""

    try:
        domains_raw = await mcp.list_domains()
        kb_domains = _parse_domain_list(domains_raw)
    except (MCPToolError, RuntimeError) as e:
        logger.warning("list_domains failed (non-fatal): %s", e)

    try:
        stats_raw = await mcp.stats()
        kb_stats_summary = _parse_stats_summary(stats_raw)
    except (MCPToolError, RuntimeError) as e:
        logger.warning("stats failed (non-fatal): %s", e)

    return kb_domains, kb_stats_summary


# Nodes where errors must propagate — pipeline can't produce useful output without them.
_CRITICAL_NODES = frozenset({"query_planner", "literature_search", "synthesis"})

# Default timeout for nodes not listed in config.node_timeouts.
_DEFAULT_NODE_TIMEOUT_S = 120


def _make_resilient_node(
    name: str,
    node_fn: Callable[[ResearchState], Awaitable[NodeUpdate]],
    timeout_s: int,
) -> Callable[[ResearchState], Awaitable[NodeUpdate]]:
    """Wrap a node with per-node timeout and graceful error handling.

    Critical nodes (query_planner, literature_search, synthesis) propagate
    all errors. Non-critical analysis nodes return an empty NodeUpdate on
    failure, allowing the pipeline to continue with partial data.

    Args:
        name: Node name for logging and critical/non-critical classification.
        node_fn: The async node function to wrap.
        timeout_s: Maximum execution time in seconds.

    Returns:
        Wrapped async node function with timeout and error handling.
    """
    critical = name in _CRITICAL_NODES

    async def wrapper(state: ResearchState) -> NodeUpdate:
        start = time.monotonic()
        try:
            async with asyncio.timeout(timeout_s):
                result = await node_fn(state)
            elapsed = time.monotonic() - start
            logger.debug("Node '%s' completed in %.1fs", name, elapsed)
            return result
        except TimeoutError:
            elapsed = time.monotonic() - start
            logger.error(
                "Node '%s' timed out after %.1fs (limit: %ds)",
                name,
                elapsed,
                timeout_s,
            )
            if critical:
                raise NodeTimeoutError(name, timeout_s) from None
            return NodeUpdate(current_node=name)
        except NodeTimeoutError:
            raise
        except Exception as exc:
            elapsed = time.monotonic() - start
            logger.error(
                "Node '%s' failed after %.1fs: %s",
                name,
                elapsed,
                exc,
            )
            if critical:
                raise
            return NodeUpdate(current_node=name)

    wrapper.__name__ = f"_resilient_{name}"
    wrapper.__qualname__ = f"_make_resilient_node.<locals>._resilient_{name}"
    return wrapper


def _passthrough(state: ResearchState) -> NodeUpdate:
    """No-op join barrier node.

    Exists only as a synchronization point after parallel fan-out.
    LangGraph merges state from both parallel branches before continuing.
    """
    return NodeUpdate(current_node="analysis_join")


def build_graph(config: AgentConfig, mcp: ResearchKBClient) -> CompiledStateGraph:  # type: ignore[type-arg]
    """Construct the research analysis graph.

    Nodes are wrapped in closures to inject config and MCP client.
    This keeps the state schema clean -- only research data flows through
    the graph, not infrastructure concerns.

    Args:
        config: Agent configuration (models, limits).
        mcp: Connected MCP client.

    Returns:
        Compiled StateGraph ready for invocation.
    """

    timeouts = config.node_timeouts

    def _timeout(name: str) -> int:
        return timeouts.get(name, _DEFAULT_NODE_TIMEOUT_S)

    # Wrap nodes to inject dependencies
    async def _query_planner(state: ResearchState) -> NodeUpdate:
        return await query_planner(state, config)

    async def _literature_search(state: ResearchState) -> NodeUpdate:
        return await literature_search(state, config, mcp)

    async def _concept_explorer(state: ResearchState) -> NodeUpdate:
        return await concept_explorer(state, config, mcp)

    async def _citation_analyzer(state: ResearchState) -> NodeUpdate:
        return await citation_analyzer(state, config, mcp)

    async def _assumption_auditor(state: ResearchState) -> NodeUpdate:
        return await assumption_auditor(state, config, mcp)

    async def _connection_explorer(state: ResearchState) -> NodeUpdate:
        return await connection_explorer(state, config, mcp)

    async def _synthesis_writer(state: ResearchState) -> NodeUpdate:
        return await synthesis_writer(state, config)

    # Build graph — each node wrapped with timeout + error handling
    graph: StateGraph[ResearchState] = StateGraph(ResearchState)

    graph.add_node(
        "query_planner",
        _make_resilient_node("query_planner", _query_planner, _timeout("query_planner")),
    )
    graph.add_node(
        "literature_search",
        _make_resilient_node(
            "literature_search", _literature_search, _timeout("literature_search")
        ),
    )
    graph.add_node(
        "concept_explorer",
        _make_resilient_node("concept_explorer", _concept_explorer, _timeout("concept_explorer")),
    )
    graph.add_node(
        "citation_analyzer",
        _make_resilient_node(
            "citation_analyzer", _citation_analyzer, _timeout("citation_analyzer")
        ),
    )
    graph.add_node("analysis_join", _passthrough)
    graph.add_node(
        "assumption_auditor",
        _make_resilient_node(
            "assumption_auditor", _assumption_auditor, _timeout("assumption_auditor")
        ),
    )
    graph.add_node(
        "connection_explorer",
        _make_resilient_node(
            "connection_explorer",
            _connection_explorer,
            _timeout("connection_explorer"),
        ),
    )
    graph.add_node(
        "synthesis",
        _make_resilient_node("synthesis", _synthesis_writer, _timeout("synthesis")),
    )

    # Sequential: planner → literature search
    graph.set_entry_point("query_planner")
    graph.add_edge("query_planner", "literature_search")

    # Fan-out: literature_search → {concept_explorer, citation_analyzer}
    graph.add_edge("literature_search", "concept_explorer")
    graph.add_edge("literature_search", "citation_analyzer")

    # Join barrier: both parallel branches merge state here
    graph.add_edge("concept_explorer", "analysis_join")
    graph.add_edge("citation_analyzer", "analysis_join")

    # Conditional: analysis_join → assumption_auditor OR connection_explorer
    graph.add_conditional_edges(
        "analysis_join",
        _should_audit_assumptions,
        {
            "assumption_auditor": "assumption_auditor",
            "connection_explorer": "connection_explorer",
        },
    )
    graph.add_edge("assumption_auditor", "connection_explorer")
    graph.add_edge("connection_explorer", "synthesis")
    graph.add_edge("synthesis", END)

    return graph.compile()


async def run_research(query: str, config: AgentConfig | None = None) -> dict[str, Any]:
    """Execute the full research pipeline.

    Args:
        query: Research question to analyze.
        config: Optional agent config (defaults to env-based).

    Returns:
        Final state dict with report and all intermediate results.
    """
    if not query or not query.strip():
        raise PlannerError("Query must not be empty.")

    if config is None:
        config = AgentConfig()

    logger.info("Starting research agent for: %s", query)

    async with ResearchKBClient(config.mcp) as mcp:
        # Pre-pipeline: fetch KB context (lightweight, no LLM)
        kb_domains, kb_stats_summary = await _fetch_kb_context(mcp)
        logger.info("KB context: %d domains, stats='%s'", len(kb_domains), kb_stats_summary)

        graph = build_graph(config, mcp)
        initial_state = ResearchState(
            query=query,
            kb_domains=kb_domains,
            kb_stats_summary=kb_stats_summary,
        )
        final_state = await graph.ainvoke(
            initial_state,
            config={
                "run_name": "research_agent",
                "metadata": {"query": query[:200]},
                "tags": ["research-agent"],
            },
        )

    logger.info("Research complete. Report length: %d chars", len(final_state.get("report", "")))
    return dict(final_state)


# ── Streaming support ──────────────────────────────────────────────────


@dataclass
class StreamEvent:
    """A streaming progress event emitted during pipeline execution.

    Attributes:
        event_type: One of 'node_end', 'report_chunk', or 'complete'.
        node_name: The graph node that produced this event.
        data: Human-readable summary or report content.
    """

    event_type: str  # "node_end" | "report_chunk" | "complete"
    node_name: str
    data: str


def _summarize_update(node_name: str, update: dict[str, Any]) -> str:
    """Map a node update to a human-readable progress message.

    Args:
        node_name: Graph node name (e.g., 'query_planner', 'synthesis').
        update: The state update dict emitted by the node.

    Returns:
        Short progress summary string.
    """
    summaries: dict[str, str] = {
        "query_planner": f"Planned {len(update.get('sub_tasks', []))} sub-tasks",
        "literature_search": f"Found {len(update.get('search_results', []))} search results",
        "concept_explorer": f"Explored {len(update.get('concepts', []))} concepts",
        "citation_analyzer": f"Analyzed {len(update.get('citations', []))} citation chains",
        "analysis_join": "Analysis complete (parallel join)",
        "assumption_auditor": f"Audited {len(update.get('assumption_audits', []))} methods",
        "connection_explorer": (
            f"Traced {len(update.get('connection_explanations', []))} concept connections"
        ),
        "synthesis": f"Generated report ({len(update.get('report', ''))} chars)",
    }
    return summaries.get(node_name, f"Completed {node_name}")


async def stream_research(
    query: str, config: AgentConfig | None = None
) -> AsyncGenerator[StreamEvent, None]:
    """Execute the research pipeline with streaming progress events.

    Yields StreamEvent for each node completion.  Uses LangGraph's native
    ``astream(stream_mode="updates")`` which yields ``{node_name: update_dict}``
    after each node finishes.

    Args:
        query: Research question to analyze.
        config: Optional agent config (defaults to env-based).

    Yields:
        StreamEvent instances: node_end for each node, report_chunk for the
        final report, and complete as the terminal event.
    """
    if not query or not query.strip():
        raise PlannerError("Query must not be empty.")

    if config is None:
        config = AgentConfig()

    logger.info("Starting streaming research agent for: %s", query)

    async with ResearchKBClient(config.mcp) as mcp:
        # Pre-pipeline: fetch KB context (lightweight, no LLM)
        kb_domains, kb_stats_summary = await _fetch_kb_context(mcp)
        logger.info("KB context: %d domains, stats='%s'", len(kb_domains), kb_stats_summary)

        graph = build_graph(config, mcp)
        initial_state = ResearchState(
            query=query,
            kb_domains=kb_domains,
            kb_stats_summary=kb_stats_summary,
        )

        run_config: dict[str, Any] = {
            "run_name": "research_agent_stream",
            "metadata": {"query": query[:200]},
            "tags": ["research-agent", "streaming"],
        }

        report = ""
        async for chunk in graph.astream(
            initial_state,
            config=run_config,  # type: ignore[arg-type]
            stream_mode="updates",
        ):
            # chunk is dict[str, dict] — one key per node that completed
            for node_name, update in chunk.items():
                summary = _summarize_update(node_name, update)
                yield StreamEvent(event_type="node_end", node_name=node_name, data=summary)

                # Emit report content when synthesis completes
                if node_name == "synthesis" and "report" in update:
                    report = update["report"]
                    yield StreamEvent(event_type="report_chunk", node_name="synthesis", data=report)

    yield StreamEvent(event_type="complete", node_name="", data=f"Report: {len(report)} chars")
