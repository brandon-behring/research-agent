"""Literature Search node — executes search queries against research-kb.

Runs all search queries from the planner's sub-tasks, collecting results
into a deduplicated list. Uses hybrid search (primary) with fast_search
fallback if the main search returns sparse results.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from research_agent.config import AgentConfig
from research_agent.mcp_client import ResearchKBClient
from research_agent.state import ResearchState, SearchResult

logger = logging.getLogger(__name__)


def _parse_search_results(markdown: str) -> list[SearchResult]:
    """Parse research-kb's markdown search output into structured results.

    Args:
        markdown: Raw markdown from research_kb_search.

    Returns:
        List of SearchResult objects extracted from the markdown.
    """
    results: list[SearchResult] = []

    # Split on "### N." headers
    sections = re.split(r"### \d+\.", markdown)

    for section in sections[1:]:  # Skip header before first result
        lines = section.strip().split("\n")
        if not lines:
            continue

        title = lines[0].strip()
        content_lines: list[str] = []
        source_id = ""
        score = 0.0
        authors = ""
        year = ""
        chunk_id = ""

        for line in lines[1:]:
            if line.startswith("*Score:"):
                score_match = re.search(r"Score:\s*([\d.]+)", line)
                if score_match:
                    score = float(score_match.group(1))
            elif "Source ID:" in line:
                id_match = re.search(r"Source ID:\s*`([^`]+)`", line)
                if id_match:
                    source_id = id_match.group(1)
                cid_match = re.search(r"Chunk ID:\s*`([^`]+)`", line)
                if cid_match:
                    chunk_id = cid_match.group(1)
            elif line.startswith("> "):
                content_lines.append(line[2:])
            elif "(" in line and ")" in line and not line.startswith("*"):
                # Author line: "Author et al. (2024) [Type]"
                authors = line.strip()
                year_match = re.search(r"\((\d{4})\)", line)
                if year_match:
                    year = year_match.group(1)

        results.append(SearchResult(
            title=title,
            content="\n".join(content_lines),
            source_id=source_id,
            score=score,
            authors=authors,
            year=year,
            chunk_id=chunk_id,
        ))

    return results


async def literature_search(
    state: ResearchState,
    config: AgentConfig,
    mcp: ResearchKBClient,
) -> dict[str, Any]:
    """Execute all search queries from sub-tasks against research-kb.

    Args:
        state: Current state with sub_tasks populated.
        config: Agent configuration.
        mcp: Connected MCP client.

    Returns:
        Dict with 'search_results' and 'search_summary' updates.
    """
    logger.info("Starting literature search across %d sub-tasks", len(state.sub_tasks))

    all_results: list[SearchResult] = []
    seen_source_ids: set[str] = set()

    for task in state.sub_tasks:
        for query in task.search_queries:
            try:
                raw = await mcp.search(
                    query=query,
                    limit=config.max_search_results,
                    context_type="balanced",
                )
                parsed = _parse_search_results(raw)

                # Deduplicate by source_id
                for result in parsed:
                    if result.source_id and result.source_id not in seen_source_ids:
                        seen_source_ids.add(result.source_id)
                        all_results.append(result)
                    elif not result.source_id:
                        all_results.append(result)

                logger.info("Query '%s': found %d results", query, len(parsed))

            except Exception as e:
                logger.error("Search failed for '%s': %s", query, e)
                # Try fast_search as fallback
                try:
                    raw = await mcp.fast_search(query=query, limit=5)
                    parsed = _parse_search_results(raw)
                    for result in parsed:
                        if result.source_id not in seen_source_ids:
                            seen_source_ids.add(result.source_id)
                            all_results.append(result)
                    logger.info("Fast search fallback for '%s': %d results", query, len(parsed))
                except Exception as e2:
                    logger.error("Fast search also failed for '%s': %s", query, e2)

    # Sort by score descending
    all_results.sort(key=lambda r: r.score, reverse=True)

    summary = (
        f"Found {len(all_results)} unique results across "
        f"{sum(len(t.search_queries) for t in state.sub_tasks)} queries."
    )
    logger.info(summary)

    return {
        "search_results": all_results,
        "search_summary": summary,
        "current_node": "literature_search",
    }
