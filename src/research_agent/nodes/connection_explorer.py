"""Connection Explorer node -- traces paths between concept pairs.

For each concept pair identified by the planner (via connections_to_explain),
calls research-kb's explain_connection tool to find graph paths with evidence.
Uses graph-only mode (use_llm=False) for deterministic, fast results (~250ms).
Our Sonnet synthesis handles the narrative.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from research_agent.config import AgentConfig
from research_agent.exceptions import MCPToolError
from research_agent.mcp_client import ResearchKBClient
from research_agent.state import NodeUpdate, ResearchState

logger = logging.getLogger(__name__)

# Timeout for all connection calls combined.
_CONNECTION_TIMEOUT_SECONDS = 30


def _parse_connection_response(raw: str) -> dict[str, Any]:
    """Parse explain_connection JSON response into a structured dict.

    Expected JSON schema::

        {
            "concept_a": "DML", "concept_b": "cross-fitting",
            "path_length": 1,
            "path_explanation": "DML directly uses cross-fitting...",
            "path": [
                {
                    "concept_name": "DML", "concept_type": "METHOD",
                    "evidence": [{"text": "...", "source": "..."}]
                }
            ]
        }

    Args:
        raw: JSON string from research_kb_explain_connection.

    Returns:
        Structured dict with path info, or minimal dict on parse failure.
    """
    try:
        data = json.loads(raw)
        return {
            "concept_a": data.get("concept_a", ""),
            "concept_b": data.get("concept_b", ""),
            "path_length": data.get("path_length", 0),
            "path_explanation": data.get("path_explanation", ""),
            "path": data.get("path", []),
        }
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning("Failed to parse connection response: %s", exc)
        return {}


async def _explore_one_connection(
    mcp: ResearchKBClient,
    concept_a: str,
    concept_b: str,
) -> dict[str, Any]:
    """Explore one concept pair connection.

    Args:
        mcp: Connected MCP client.
        concept_a: Source concept name.
        concept_b: Target concept name.

    Returns:
        Parsed connection dict, or empty dict on failure.
    """
    try:
        raw = await mcp.explain_connection(
            concept_a=concept_a,
            concept_b=concept_b,
            style="research",
            use_llm=False,
        )
        result = _parse_connection_response(raw)
        if result:
            logger.info(
                "Connection %s → %s: %d hops",
                concept_a,
                concept_b,
                result.get("path_length", 0),
            )
        return result
    except (MCPToolError, RuntimeError) as e:
        logger.warning("explain_connection failed for %s → %s: %s", concept_a, concept_b, e)
        return {}


async def connection_explorer(
    state: ResearchState,
    config: AgentConfig,
    mcp: ResearchKBClient,
) -> NodeUpdate:
    """Trace conceptual paths between concept pairs from the planner.

    Collects all connections_to_explain from sub-tasks, deduplicates pairs,
    and calls explain_connection for each. Graceful degradation: failed
    pairs are skipped with a warning.

    Args:
        state: Current state with sub_tasks populated.
        config: Agent configuration.
        mcp: Connected MCP client.

    Returns:
        NodeUpdate with ``connection_explanations``.
    """
    logger.info("Exploring concept connections")

    # Flatten and deduplicate connection pairs from all sub-tasks
    seen_pairs: set[tuple[str, str]] = set()
    unique_pairs: list[tuple[str, str]] = []
    for task in state.sub_tasks:
        for pair in task.connections_to_explain:
            if len(pair) >= 2:
                # Normalize: sort alphabetically for dedup
                key = (min(pair[0], pair[1]), max(pair[0], pair[1]))
                if key not in seen_pairs:
                    seen_pairs.add(key)
                    unique_pairs.append((pair[0], pair[1]))

    if not unique_pairs:
        logger.info("No concept connections to explore")
        return NodeUpdate(
            connection_explanations=[],
            current_node="connection_explorer",
        )

    explanations: list[dict[str, Any]] = []

    try:
        async with asyncio.timeout(_CONNECTION_TIMEOUT_SECONDS):
            results = await asyncio.gather(
                *[_explore_one_connection(mcp, a, b) for a, b in unique_pairs],
                return_exceptions=True,
            )
            for i, result in enumerate(results):
                if isinstance(result, BaseException):
                    a, b = unique_pairs[i]
                    logger.warning("Unexpected error for %s → %s: %s", a, b, result)
                elif result:  # Skip empty dicts from failed parses
                    explanations.append(result)

    except TimeoutError:
        logger.warning(
            "Connection exploration timed out with %d/%d pairs",
            len(explanations),
            len(unique_pairs),
        )

    logger.info("Explored %d concept connections", len(explanations))

    return NodeUpdate(
        connection_explanations=explanations,
        current_node="connection_explorer",
    )
