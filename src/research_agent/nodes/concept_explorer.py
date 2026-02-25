"""Concept Explorer node -- traverses the knowledge graph.

For each concept identified by the planner, retrieves concept details
and explores the graph neighborhood to discover related concepts.
Also extracts concept IDs from search results for deeper exploration.
"""

from __future__ import annotations

import asyncio
import logging
import re

from research_agent.config import AgentConfig
from research_agent.exceptions import MCPToolError
from research_agent.mcp_client import ResearchKBClient
from research_agent.state import ConceptInfo, NodeUpdate, ResearchState

logger = logging.getLogger(__name__)

# Timeout for all graph_neighborhood calls combined.
_EXPLORATION_TIMEOUT_SECONDS = 45

# Default hops for graph_neighborhood traversal.
_DEFAULT_HOPS = 2

# Default neighbor limit per concept.
_DEFAULT_NEIGHBOR_LIMIT = 30


def _parse_concept_detail(markdown: str) -> ConceptInfo | None:
    """Parse a concept detail response into ConceptInfo.

    Expected format from research-kb::

        ## Concept Name
        **Type:** METHOD
        **ID:** `concept-xyz-001`

        ### Description
        A framework for...

        ### Relationships (N total)
        - REQUIRES -> `concept-abc-001`

    Args:
        markdown: Raw markdown from research_kb_get_concept.

    Returns:
        ConceptInfo or None if parsing fails.
    """
    if not markdown or not markdown.strip():
        return None

    concept_id = ""
    name = ""
    concept_type = ""
    description = ""
    relationships: list[dict[str, str]] = []

    id_match = re.search(r"ID:\s*`([^`]+)`", markdown)
    if id_match:
        concept_id = id_match.group(1)

    name_match = re.search(r"## (.+)", markdown)
    if name_match:
        name = name_match.group(1).strip()

    type_match = re.search(r"Type:\*?\*?\s*(\w+)", markdown)
    if type_match:
        concept_type = type_match.group(1)

    # Extract description (between Description header and Relationships header)
    desc_match = re.search(
        r"### Description\s*\n(.*?)(?=### Relationships|\Z)",
        markdown,
        re.DOTALL,
    )
    if desc_match:
        description = desc_match.group(1).strip()

    # Extract relationships
    rel_pattern = re.compile(r"- (\w+)\s*\u2192\s*`([^`]+)`")
    for match in rel_pattern.finditer(markdown):
        relationships.append(
            {
                "type": match.group(1),
                "target_id": match.group(2),
            }
        )

    if not concept_id and not name:
        return None

    try:
        return ConceptInfo(
            concept_id=concept_id,
            name=name,
            concept_type=concept_type,
            description=description,
            relationships=relationships,
        )
    except Exception as e:
        logger.warning("Failed to construct ConceptInfo for '%s': %s", name, e)
        return None


async def _explore_one(
    mcp: ResearchKBClient,
    name: str,
) -> ConceptInfo | None:
    """Explore graph neighborhood for a single concept.

    Args:
        mcp: Connected MCP client.
        name: Concept name to explore.

    Returns:
        ConceptInfo or None if exploration failed.
    """
    try:
        raw = await mcp.graph_neighborhood(
            concept_name=name,
            hops=_DEFAULT_HOPS,
            limit=_DEFAULT_NEIGHBOR_LIMIT,
        )
        logger.info("Explored neighborhood for concept: %s", name)

        # Extract concept_id from neighborhood header if present
        # Format: *Type: METHOD | ID: `concept-dml-001`*
        concept_id = ""
        id_match = re.search(r"ID:\s*`([^`]+)`", raw)
        if id_match:
            concept_id = id_match.group(1)

        return ConceptInfo(
            concept_id=concept_id,
            name=name,
            neighborhood_summary=raw,
        )
    except (MCPToolError, RuntimeError) as e:
        logger.warning("Failed to explore concept '%s': %s", name, e)
        return None


async def concept_explorer(
    state: ResearchState,
    config: AgentConfig,
    mcp: ResearchKBClient,
) -> NodeUpdate:
    """Explore concepts in the knowledge graph.

    Fires all graph_neighborhood calls concurrently via asyncio.gather
    with return_exceptions=True for graceful degradation.

    Args:
        state: Current state with sub_tasks and search_results.
        config: Agent configuration.
        mcp: Connected MCP client.

    Returns:
        NodeUpdate with ``concepts`` and ``concept_map_summary``.
    """
    logger.info("Exploring knowledge graph concepts")

    explored_names: set[str] = set()

    # Collect concept names from all sub-tasks
    concept_names: list[str] = []
    for task in state.sub_tasks:
        concept_names.extend(task.concepts_to_explore)

    # Deduplicate
    unique_names = []
    for name in concept_names:
        lower = name.lower()
        if lower not in explored_names:
            explored_names.add(lower)
            unique_names.append(name)

    # Limit to avoid excessive API calls
    unique_names = unique_names[: config.max_concepts]

    concepts: list[ConceptInfo] = []

    try:
        async with asyncio.timeout(_EXPLORATION_TIMEOUT_SECONDS):
            # Fan out graph_neighborhood calls concurrently
            neighborhood_results = await asyncio.gather(
                *[_explore_one(mcp, name) for name in unique_names],
                return_exceptions=True,
            )
            for i, r in enumerate(neighborhood_results):
                if isinstance(r, BaseException):
                    logger.warning("Unexpected error exploring '%s': %s", unique_names[i], r)
                elif r is not None:
                    concepts.append(r)

    except TimeoutError:
        logger.warning("Concept exploration timed out with %d concepts", len(concepts))

    summary = f"Explored {len(unique_names)} concepts, found {len(concepts)} total entries."
    logger.info(summary)

    return NodeUpdate(
        concepts=concepts,
        concept_map_summary=summary,
        current_node="concept_explorer",
    )
