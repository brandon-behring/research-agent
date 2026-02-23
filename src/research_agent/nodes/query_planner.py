"""Query Planner node — decomposes a research question into sub-tasks.

Uses Haiku for speed: this is a routing/planning step, not a reasoning step.
The planner identifies what to search, what concepts to explore, and which
methods (if any) need assumption auditing.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from research_agent.config import AgentConfig
from research_agent.state import ResearchState, SubTask

logger = logging.getLogger(__name__)

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

Return JSON array of sub-tasks. Be specific with search queries — the KB uses
hybrid search (BM25 + vector + graph), so both keywords and natural language work.

Example output:
[
  {
    "description": "Find foundational papers on double machine learning",
    "search_queries": ["double machine learning Chernozhukov", "DML cross-fitting"],
    "concepts_to_explore": ["double machine learning", "cross-fitting"],
    "methods_to_audit": ["DML"]
  },
  {
    "description": "Understand identification assumptions",
    "search_queries": ["unconfoundedness assumption causal inference"],
    "concepts_to_explore": ["unconfoundedness", "conditional independence"],
    "methods_to_audit": []
  }
]

If the query is about a topic NOT in the KB, still create sub-tasks with reasonable
search queries — the system will report what it finds (which may be sparse)."""


async def query_planner(state: ResearchState, config: AgentConfig) -> dict[str, Any]:
    """Decompose a research question into sub-tasks.

    Args:
        state: Current graph state with the user's query.
        config: Agent configuration (model selection).

    Returns:
        Dict with 'sub_tasks' and 'planning_rationale' updates.
    """
    logger.info("Planning research for: %s", state.query)

    llm = ChatAnthropic(
        model=config.models.planning,
        max_tokens=2048,
        temperature=0.0,
    )

    response = await llm.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Research question: {state.query}"),
    ])

    # Parse JSON from response
    content = response.content
    if isinstance(content, list):
        content = content[0].get("text", "") if content else ""

    try:
        # Extract JSON from response (handle markdown code blocks)
        json_str = content
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0]
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0]

        tasks_data = json.loads(json_str.strip())
        sub_tasks = [
            SubTask(
                description=t.get("description", ""),
                search_queries=t.get("search_queries", []),
                concepts_to_explore=t.get("concepts_to_explore", []),
                methods_to_audit=t.get("methods_to_audit", []),
            )
            for t in tasks_data
        ]
    except (json.JSONDecodeError, IndexError, KeyError) as e:
        logger.warning("Failed to parse planner output: %s. Using fallback.", e)
        # Fallback: single task with the original query
        sub_tasks = [
            SubTask(
                description=state.query,
                search_queries=[state.query],
                concepts_to_explore=[],
                methods_to_audit=[],
            )
        ]

    logger.info("Created %d sub-tasks", len(sub_tasks))
    return {
        "sub_tasks": sub_tasks,
        "planning_rationale": content,
        "current_node": "query_planner",
    }
