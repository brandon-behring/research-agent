"""Query Planner node -- decomposes a research question into sub-tasks.

Uses Haiku for speed: this is a routing/planning step, not a reasoning step.
The planner identifies what to search, what concepts to explore, and which
methods (if any) need assumption auditing.

Uses ``with_structured_output()`` for guaranteed JSON schema compliance,
eliminating fragile regex/JSON parsing.
"""

from __future__ import annotations

import asyncio
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from research_agent.config import AgentConfig
from research_agent.exceptions import PlannerError
from research_agent.llm import create_llm
from research_agent.state import NodeUpdate, ResearchState, SubTask

logger = logging.getLogger(__name__)

# Timeout for the planner LLM call — planning is a routing step, should be fast.
_PLANNER_TIMEOUT_SECONDS = 30

_SYSTEM_PROMPT_TEMPLATE = """\
You are a research query planner. Given a research question, decompose it
into concrete sub-tasks for a multi-agent research system.

{kb_context}

For each sub-task, specify:
1. description: What to investigate
2. search_queries: 1-3 specific search queries for the knowledge base
3. concepts_to_explore: Key concepts to look up in the knowledge graph (by name)
4. methods_to_audit: Statistical methods whose assumptions should be checked
5. search_domain: Domain filter -- use one of the available domains or "" (cross-domain)
6. search_context: Search weighting strategy:
   - "auditing": emphasize assumption-checking content (use when auditing methods)
   - "building": emphasize implementation/construction content (use when building models)
   - "balanced": default, equal weighting
7. connections_to_explain: Pairs of concept names whose relationship should be traced.
   Use this when two concepts are mentioned together but their connection needs explanation.
   Example: [["DML", "cross-fitting"], ["unconfoundedness", "overlap"]]

Be specific with search queries -- the KB uses hybrid search (BM25 + vector + graph),
so both keywords and natural language work.

If the query is about a topic NOT in the KB, still create sub-tasks with reasonable
search queries -- the system will report what it finds (which may be sparse)."""

_DEFAULT_KB_DESCRIPTION = """The system has access to a causal inference knowledge base covering:
- Causal inference methods (DML, IV, DiD, RDD, propensity score, synthetic control)
- Time series analysis
- RAG/LLM systems
- Statistical methodology"""


def _build_system_prompt(state: ResearchState) -> str:
    """Build planner system prompt, injecting KB context when available.

    When kb_domains and kb_stats_summary are populated (from pre-pipeline
    list_domains + stats calls), the planner gets informed domain selection.
    Falls back to a static description when KB context is unavailable.

    Args:
        state: Current state with optional kb_domains and kb_stats_summary.

    Returns:
        Formatted system prompt string.
    """
    if state.kb_domains or state.kb_stats_summary:
        lines = ["The system has access to a research knowledge base."]
        if state.kb_stats_summary:
            lines.append(f"KB contains {state.kb_stats_summary}.")
        if state.kb_domains:
            domains_str = ", ".join(state.kb_domains)
            lines.append(f"Available KB domains: {domains_str}")
        kb_context = "\n".join(lines)
    else:
        kb_context = _DEFAULT_KB_DESCRIPTION

    return _SYSTEM_PROMPT_TEMPLATE.format(kb_context=kb_context)


class PlannerOutput(BaseModel):
    """Structured output from the query planner LLM."""

    sub_tasks: list[SubTask] = Field(description="Decomposed research sub-tasks")
    rationale: str = Field(description="Brief explanation of the decomposition strategy")


async def query_planner(state: ResearchState, config: AgentConfig) -> NodeUpdate:
    """Decompose a research question into sub-tasks.

    Args:
        state: Current graph state with the user's query.
        config: Agent configuration (model selection).

    Returns:
        NodeUpdate with ``sub_tasks`` and ``planning_rationale``.

    Raises:
        PlannerError: If planning fails after timeout or LLM error.
    """
    logger.info("Planning research for: %s", state.query)

    llm = create_llm(
        config.models.planning,
        max_tokens=2048,
        temperature=0.0,
    ).with_structured_output(PlannerOutput)

    system_prompt = _build_system_prompt(state)

    try:
        async with asyncio.timeout(_PLANNER_TIMEOUT_SECONDS):
            result = await llm.ainvoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=f"Research question: {state.query}"),
                ]
            )
            if not isinstance(result, PlannerOutput):
                raise PlannerError(f"Expected PlannerOutput, got {type(result).__name__}")
    except TimeoutError as e:
        raise PlannerError(
            f"Query planning timed out after {_PLANNER_TIMEOUT_SECONDS}s for: {state.query}"
        ) from e
    except Exception as e:
        raise PlannerError(f"Query planning failed: {e}") from e

    logger.info("Created %d sub-tasks", len(result.sub_tasks))
    return NodeUpdate(
        sub_tasks=result.sub_tasks,
        planning_rationale=result.rationale,
        current_node="query_planner",
    )
