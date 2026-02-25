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

SYSTEM_PROMPT = """You are a research query planner. Given a research question, decompose it
into concrete sub-tasks for a multi-agent research system.

The system has access to a causal inference knowledge base with 478 sources covering:
- Causal inference methods (DML, IV, DiD, RDD, propensity score, synthetic control)
- Time series analysis
- RAG/LLM systems
- Statistical methodology

For each sub-task, specify:
1. description: What to investigate
2. search_queries: 1-3 specific search queries for the knowledge base
3. concepts_to_explore: Key concepts to look up in the knowledge graph (by name)
4. methods_to_audit: Statistical methods whose assumptions should be checked

Be specific with search queries -- the KB uses hybrid search (BM25 + vector + graph),
so both keywords and natural language work.

If the query is about a topic NOT in the KB, still create sub-tasks with reasonable
search queries -- the system will report what it finds (which may be sparse)."""


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

    try:
        async with asyncio.timeout(_PLANNER_TIMEOUT_SECONDS):
            result = await llm.ainvoke(
                [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=f"Research question: {state.query}"),
                ]
            )
            assert isinstance(result, PlannerOutput)
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
