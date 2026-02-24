"""LangGraph StateGraph -- orchestrates the multi-agent research pipeline.

Architecture:
    Query Planner -> Literature Search -> Concept Explorer -> Citation Analyzer
                                                            -> Assumption Auditor
                 -> Synthesis Writer

    The graph uses conditional edges to:
    1. Skip assumption auditing if no methods were identified
    2. Route to synthesis after all analysis nodes complete

Design decisions:
    - Pydantic BaseModel state for rich validation and type support
    - Sequential main pipeline with potential parallel analysis
    - Conditional routing avoids unnecessary MCP calls
    - Config and MCP client injected via closure (not in state)
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from research_agent.config import AgentConfig
from research_agent.mcp_client import ResearchKBClient
from research_agent.nodes.assumption_auditor import assumption_auditor
from research_agent.nodes.citation_analyzer import citation_analyzer
from research_agent.nodes.concept_explorer import concept_explorer
from research_agent.nodes.literature_search import literature_search
from research_agent.nodes.query_planner import query_planner
from research_agent.nodes.synthesis import synthesis_writer
from research_agent.state import NodeUpdate, ResearchState

logger = logging.getLogger(__name__)


def _should_audit_assumptions(state: ResearchState) -> str:
    """Route to assumption auditor only if methods were identified.

    Args:
        state: Current state after concept exploration.

    Returns:
        'assumption_auditor' if methods exist, 'synthesis' otherwise.
    """
    methods = []
    for task in state.sub_tasks:
        methods.extend(task.methods_to_audit)

    if methods:
        return "assumption_auditor"
    return "synthesis"


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

    async def _synthesis_writer(state: ResearchState) -> NodeUpdate:
        return await synthesis_writer(state, config)

    # Build graph
    graph: StateGraph[ResearchState] = StateGraph(ResearchState)

    # Add nodes
    graph.add_node("query_planner", _query_planner)
    graph.add_node("literature_search", _literature_search)
    graph.add_node("concept_explorer", _concept_explorer)
    graph.add_node("citation_analyzer", _citation_analyzer)
    graph.add_node("assumption_auditor", _assumption_auditor)
    graph.add_node("synthesis", _synthesis_writer)

    # Define edges
    graph.set_entry_point("query_planner")
    graph.add_edge("query_planner", "literature_search")
    graph.add_edge("literature_search", "concept_explorer")
    graph.add_edge("concept_explorer", "citation_analyzer")

    # Conditional: citation_analyzer -> assumption_auditor OR synthesis
    graph.add_conditional_edges(
        "citation_analyzer",
        _should_audit_assumptions,
        {
            "assumption_auditor": "assumption_auditor",
            "synthesis": "synthesis",
        },
    )
    graph.add_edge("assumption_auditor", "synthesis")
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
    if config is None:
        config = AgentConfig()

    logger.info("Starting research agent for: %s", query)

    async with ResearchKBClient(config.mcp) as mcp:
        graph = build_graph(config, mcp)
        initial_state = ResearchState(query=query)
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
