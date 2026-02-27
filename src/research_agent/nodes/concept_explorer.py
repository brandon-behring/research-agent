"""Concept Explorer node -- traverses the knowledge graph.

For each concept identified by the planner, retrieves concept details
and explores the graph neighborhood to discover related concepts.
Also extracts concept IDs from search results for deeper exploration.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from research_agent.config import AgentConfig
from research_agent.exceptions import MCPToolError
from research_agent.mcp_client import ResearchKBClient
from research_agent.parsing import parse_json_first
from research_agent.state import ConceptInfo, NodeUpdate, ResearchState

logger = logging.getLogger(__name__)

# Timeout for all graph_neighborhood calls combined.
_EXPLORATION_TIMEOUT_SECONDS = 45

# Default hops for graph_neighborhood traversal.
_DEFAULT_HOPS = 2

# Default neighbor limit per concept.
_DEFAULT_NEIGHBOR_LIMIT = 30

# Maximum auto-discovered methods from graph (ASSUMPTION/THEOREM types).
_MAX_DISCOVERED_METHODS = 3


def _parse_concept_detail_json(raw: str) -> ConceptInfo | None:
    """Parse JSON concept detail response into ConceptInfo.

    Expected JSON schema::

        {
            "concept_id": "...", "name": "...", "concept_type": "METHOD",
            "definition": "...",
            "relationships": [{"type": "REQUIRES", "target_id": "..."}]
        }

    Note: JSON key ``definition`` maps to ``ConceptInfo.description``.

    Args:
        raw: JSON string from research_kb_get_concept.

    Returns:
        ConceptInfo or None if required fields are missing.

    Raises:
        json.JSONDecodeError: If raw is not valid JSON.
    """
    data = json.loads(raw)
    return ConceptInfo(
        concept_id=data.get("concept_id", ""),
        name=data.get("name", ""),
        concept_type=data.get("concept_type", ""),
        description=data.get("definition", ""),
        relationships=[
            {"type": r.get("type", ""), "target_id": r.get("target_id", "")}
            for r in data.get("relationships", [])
        ],
    )


def _parse_concept_detail_markdown(markdown: str) -> ConceptInfo | None:
    """Parse a concept detail markdown response into ConceptInfo.

    Retained as fallback when JSON parsing fails.

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


def _parse_concept_detail(raw: str) -> ConceptInfo | None:
    """Parse concept detail with JSON-first strategy and markdown fallback.

    Args:
        raw: Raw response string (JSON or markdown) from research-kb.

    Returns:
        ConceptInfo or None if parsing fails entirely.
    """
    if not raw or not raw.strip():
        return None
    return parse_json_first(
        raw,
        _parse_concept_detail_json,
        _parse_concept_detail_markdown,
        context="concept detail",
    )


def _build_neighborhood_summary(data: dict[str, Any]) -> str:
    """Build a human-readable neighborhood summary from JSON data.

    Produces formatted markdown suitable for synthesis LLM context
    (``synthesis.py:112-113`` feeds ``c.neighborhood_summary`` directly).

    Args:
        data: Parsed JSON dict from research_kb_graph_neighborhood.

    Returns:
        Formatted markdown summary string.
    """
    center = data.get("center", {})
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    type_counts = data.get("relationship_type_counts", {})

    lines = [
        f"## Graph Neighborhood: {center.get('name', 'Unknown')}",
        f"*Type: {center.get('type', '')} | ID: `{center.get('id', '')}`*",
        f"\n**{len(nodes)} connected concepts, {len(edges)} relationships**",
        "\n### Connected Concepts",
    ]
    for node in nodes:
        lines.append(f"- {node.get('name', '?')} [{node.get('type', '')}]")

    if type_counts:
        lines.append("\n### Relationships")
        for rtype, count in type_counts.items():
            lines.append(f"- {rtype}: {count}")

    return "\n".join(lines)


@dataclass
class _ExploreResult:
    """Internal result from _explore_one, carrying concept + discovered neighbors."""

    concept: ConceptInfo
    assumption_neighbors: list[str]  # ASSUMPTION/THEOREM names from neighborhood


async def _explore_one(
    mcp: ResearchKBClient,
    name: str,
) -> _ExploreResult | None:
    """Explore graph neighborhood and enrich with structured concept detail.

    Two-step retrieval:
        1. ``graph_neighborhood`` — discovers connected concepts and extracts concept_id
        2. ``get_concept`` — fetches structured fields (type, description, relationships)

    Step 2 is sequential (needs concept_id from step 1), but the outer
    ``asyncio.gather`` in ``concept_explorer()`` still fans out across concepts.

    Args:
        mcp: Connected MCP client.
        name: Concept name to explore.

    Returns:
        _ExploreResult with concept info + discovered neighbors, or None on failure.
    """
    try:
        raw = await mcp.graph_neighborhood(
            concept_name=name,
            hops=_DEFAULT_HOPS,
            limit=_DEFAULT_NEIGHBOR_LIMIT,
        )
        logger.info("Explored neighborhood for concept: %s", name)

        # Extract concept_id and build summary — JSON-first with markdown fallback
        concept_id = ""
        neighborhood_summary = raw  # fallback: raw response as-is
        assumption_neighbors: list[str] = []
        try:
            data = json.loads(raw)
            concept_id = data.get("center", {}).get("id", "")
            neighborhood_summary = _build_neighborhood_summary(data)
            # Collect ASSUMPTION/THEOREM neighbors for auto-discovery
            for node in data.get("nodes", []):
                if node.get("type") in ("ASSUMPTION", "THEOREM") and node.get("name"):
                    assumption_neighbors.append(node["name"])
        except (json.JSONDecodeError, KeyError, TypeError):
            id_match = re.search(r"ID:\s*`([^`]+)`", raw)
            if id_match:
                concept_id = id_match.group(1)

        # Enrich with structured concept detail if we have an ID
        detail: ConceptInfo | None = None
        if concept_id:
            try:
                detail_raw = await mcp.get_concept(concept_id, include_relationships=True)
                detail = _parse_concept_detail(detail_raw)
            except (MCPToolError, RuntimeError) as exc:
                logger.warning(
                    "get_concept failed for '%s' (%s): %s — falling back to neighborhood only",
                    name,
                    concept_id,
                    exc,
                )

        concept = ConceptInfo(
            concept_id=concept_id,
            name=detail.name if detail and detail.name else name,
            concept_type=detail.concept_type if detail else "",
            description=detail.description if detail else "",
            relationships=detail.relationships if detail else [],
            neighborhood_summary=neighborhood_summary,
        )
        return _ExploreResult(concept=concept, assumption_neighbors=assumption_neighbors)
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
    discovered_methods: list[str] = []
    seen_discovered: set[str] = set()

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
                    concepts.append(r.concept)
                    # Collect ASSUMPTION/THEOREM neighbors for auto-discovery
                    for neighbor_name in r.assumption_neighbors:
                        if len(discovered_methods) >= _MAX_DISCOVERED_METHODS:
                            break
                        name_lower = neighbor_name.lower()
                        if name_lower not in seen_discovered:
                            seen_discovered.add(name_lower)
                            discovered_methods.append(neighbor_name)
                            logger.info("Auto-discovered method from graph: %s", neighbor_name)

    except TimeoutError:
        logger.warning("Concept exploration timed out with %d concepts", len(concepts))

    summary = f"Explored {len(unique_names)} concepts, found {len(concepts)} total entries."
    if discovered_methods:
        methods_str = ", ".join(discovered_methods)
        summary += f" Auto-discovered {len(discovered_methods)} methods: {methods_str}."
    logger.info(summary)

    return NodeUpdate(
        concepts=concepts,
        concept_map_summary=summary,
        discovered_methods=discovered_methods,
        current_node="concept_explorer",
    )
