"""Literature Search node -- executes search queries against research-kb.

Runs all search queries from the planner's sub-tasks, collecting results
into a deduplicated list. Uses hybrid search (primary) with fast_search
fallback in two scenarios:
    1. Primary search raises an exception (MCPToolError, RuntimeError)
    2. Primary search returns fewer than ``_MIN_SPARSE_THRESHOLD`` results
"""

from __future__ import annotations

import asyncio
import logging
import re

from research_agent.config import AgentConfig
from research_agent.exceptions import MCPToolError, SearchError
from research_agent.mcp_client import ResearchKBClient
from research_agent.state import NodeUpdate, ResearchState, SearchResult

logger = logging.getLogger(__name__)

# Timeout for the entire search operation across all queries.
_SEARCH_TIMEOUT_SECONDS = 90

# Max concurrent search requests — research-kb's connection pool maxes at ~10.
_CONCURRENCY_LIMIT = 3

# Minimum results from primary search before triggering fast_search supplement.
_MIN_SPARSE_THRESHOLD = 2


def _parse_search_results(markdown: str) -> list[SearchResult]:
    """Parse research-kb's markdown search output into structured results.

    Expected format from research-kb::

        ### 1. Title Here
        Author et al. (2024) [Paper]
        > Content snippet...
        *Score: 0.892 | FTS: 0.756 | Vector: 0.891 | Graph: 0.823*
        *Source ID: `src-001` | Chunk ID: `chk-001`*

    Args:
        markdown: Raw markdown from research_kb_search.

    Returns:
        List of SearchResult objects extracted from the markdown.
    """
    if not markdown or not markdown.strip():
        return []

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
                    try:
                        raw_score = float(score_match.group(1))
                        score = min(max(raw_score, 0.0), 1.0)
                    except ValueError:
                        logger.warning("Non-numeric score in line: %s", line)
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

        try:
            results.append(
                SearchResult(
                    title=title,
                    content="\n".join(content_lines),
                    source_id=source_id,
                    score=score,
                    authors=authors,
                    year=year,
                    chunk_id=chunk_id,
                )
            )
        except Exception as e:
            logger.warning("Failed to construct SearchResult for '%s': %s", title, e)

    return results


async def _search_one(
    mcp: ResearchKBClient,
    query: str,
    config: AgentConfig,
) -> tuple[list[SearchResult], str]:
    """Execute a single search query with fast_search fallback.

    Args:
        mcp: Connected MCP client.
        query: Search query string.
        config: Agent configuration (for max_search_results).

    Returns:
        Tuple of (parsed results, query string) for post-gather dedup.
    """
    try:
        raw = await mcp.search(
            query=query,
            limit=config.max_search_results,
            context_type="balanced",
        )
        results = _parse_search_results(raw)

        # Sparse-result fallback: supplement with fast_search if too few results
        if len(results) < _MIN_SPARSE_THRESHOLD:
            logger.info(
                "Sparse results for '%s' (%d < %d), supplementing with fast_search",
                query,
                len(results),
                _MIN_SPARSE_THRESHOLD,
            )
            try:
                fast_raw = await mcp.fast_search(query=query, limit=5)
                results.extend(_parse_search_results(fast_raw))
            except (MCPToolError, RuntimeError) as e2:
                logger.warning("Fast search supplement failed for '%s': %s", query, e2)

        return results, query
    except (MCPToolError, RuntimeError) as e:
        logger.error("Search failed for '%s': %s", query, e)
        try:
            raw = await mcp.fast_search(query=query, limit=5)
            logger.info("Fast search fallback for '%s'", query)
            return _parse_search_results(raw), query
        except (MCPToolError, RuntimeError) as e2:
            logger.error("Fast search also failed for '%s': %s", query, e2)
            return [], query


async def literature_search(
    state: ResearchState,
    config: AgentConfig,
    mcp: ResearchKBClient,
) -> NodeUpdate:
    """Execute all search queries from sub-tasks against research-kb.

    Fires all queries concurrently via asyncio.gather, then deduplicates
    results by source_id post-gather.

    Args:
        state: Current state with sub_tasks populated.
        config: Agent configuration.
        mcp: Connected MCP client.

    Returns:
        NodeUpdate with ``search_results`` and ``search_summary``.
    """
    logger.info("Starting literature search across %d sub-tasks", len(state.sub_tasks))

    all_queries = [q for task in state.sub_tasks for q in task.search_queries]
    all_results: list[SearchResult] = []
    seen_source_ids: set[str] = set()

    sem = asyncio.Semaphore(_CONCURRENCY_LIMIT)

    async def _bounded_search(q: str) -> tuple[list[SearchResult], str]:
        async with sem:
            return await _search_one(mcp, q, config)

    try:
        async with asyncio.timeout(_SEARCH_TIMEOUT_SECONDS):
            batch_results = await asyncio.gather(
                *[_bounded_search(q) for q in all_queries],
                return_exceptions=True,
            )

            # Deduplicate across all query results post-gather
            for i, raw in enumerate(batch_results):
                if isinstance(raw, BaseException):
                    logger.warning(
                        "Unexpected error searching '%s': %s", all_queries[i], raw
                    )
                    continue
                parsed, query = raw
                for result in parsed:
                    if result.source_id and result.source_id not in seen_source_ids:
                        seen_source_ids.add(result.source_id)
                        all_results.append(result)
                    elif not result.source_id:
                        all_results.append(result)
                logger.info("Query '%s': found %d results", query, len(parsed))

    except TimeoutError as e:
        logger.warning(
            "Literature search timed out after %ds with %d results",
            _SEARCH_TIMEOUT_SECONDS,
            len(all_results),
        )
        if not all_results:
            raise SearchError("Literature search timed out with no results") from e

    # Sort by score descending
    all_results.sort(key=lambda r: r.score, reverse=True)

    summary = f"Found {len(all_results)} unique results across {len(all_queries)} queries."
    logger.info(summary)

    return NodeUpdate(
        search_results=all_results,
        search_summary=summary,
        current_node="literature_search",
    )
