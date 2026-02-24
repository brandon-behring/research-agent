"""Concept Explorer node -- traverses the knowledge graph.

For each concept identified by the planner, retrieves concept details
and explores the graph neighborhood to discover related concepts.
Also extracts concept IDs from search results for deeper exploration.
"""

from __future__ import annotations

import logging
import re

from research_agent.config import AgentConfig
from research_agent.mcp_client import ResearchKBClient
from research_agent.state import ConceptInfo, NodeUpdate, ResearchState

logger = logging.getLogger(__name__)


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
    rel_pattern = re.compile(r"- (\w+)\s*→\s*`([^`]+)`")
    for match in rel_pattern.finditer(markdown):
        relationships.append({
            "type": match.group(1),
            "target_id": match.group(2),
        })

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


async def concept_explorer(
    state: ResearchState,
    config: AgentConfig,
    mcp: ResearchKBClient,
) -> NodeUpdate:
    """Explore concepts in the knowledge graph.

    Strategy:
        1. Collect concept names from planner sub-tasks
        2. Explore graph neighborhoods for each concept
        3. If concepts have IDs (from search results), fetch details

    Args:
        state: Current state with sub_tasks and search_results.
        config: Agent configuration.
        mcp: Connected MCP client.

    Returns:
        NodeUpdate with ``concepts`` and ``concept_map_summary``.
    """
    logger.info("Exploring knowledge graph concepts")

    concepts: list[ConceptInfo] = []
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

    for name in unique_names:
        try:
            raw = await mcp.graph_neighborhood(
                concept_name=name,
                hops=2,
                limit=30,
            )

            info = ConceptInfo(
                concept_id="",
                name=name,
                neighborhood_summary=raw,
            )
            concepts.append(info)
            logger.info("Explored neighborhood for concept: %s", name)

        except Exception as e:
            logger.warning("Failed to explore concept '%s': %s", name, e)

    # Also try to get details for any concept IDs found in search results
    concept_ids_seen: set[str] = set()
    for result in state.search_results[:5]:  # Top 5 results
        if result.source_id:
            # Extract concepts mentioned in the content
            id_pattern = re.compile(r"Concept ID:\s*`([^`]+)`")
            for match in id_pattern.finditer(result.content):
                cid = match.group(1)
                if cid not in concept_ids_seen:
                    concept_ids_seen.add(cid)
                    try:
                        raw = await mcp.get_concept(cid)
                        parsed = _parse_concept_detail(raw)
                        if parsed:
                            concepts.append(parsed)
                    except Exception as e:
                        logger.warning("Failed to get concept %s: %s", cid, e)

    summary = f"Explored {len(unique_names)} concepts, found {len(concepts)} total entries."
    logger.info(summary)

    return NodeUpdate(
        concepts=concepts,
        concept_map_summary=summary,
        current_node="concept_explorer",
    )
