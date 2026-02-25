"""Citation Analyzer node -- maps citation networks and finds related papers.

Takes source IDs from search results, builds citation chains (who cites whom),
and discovers related papers through bibliographic coupling (shared references).
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from research_agent.config import AgentConfig
from research_agent.mcp_client import ResearchKBClient
from research_agent.state import CitationInfo, NodeUpdate, ResearchState

logger = logging.getLogger(__name__)


def _parse_citation_network(markdown: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Parse citation network markdown into citing/cited-by lists.

    Expected format::

        ### Citing This Source (N)
        - **Title** (2021)
          - Author
          - ID: `src-xxx`

        ### Cited By This Source (N)
        ...

    Args:
        markdown: Raw markdown from research_kb_citation_network.

    Returns:
        Tuple of (citing_sources, cited_by_sources).
    """
    if not markdown or not markdown.strip():
        return [], []

    citing: list[dict[str, str]] = []
    cited_by: list[dict[str, str]] = []

    current_section = None

    for line in markdown.split("\n"):
        if "Citing This Source" in line:
            current_section = "citing"
        elif "Cited By This Source" in line:
            current_section = "cited_by"
        elif line.startswith("- **") and current_section:
            title_match = re.match(r"- \*\*(.+?)\*\*\s*\((\d{4})\)", line)
            if title_match:
                entry = {"title": title_match.group(1), "year": title_match.group(2)}
                id_match = re.search(r"ID:\s*`([^`]+)`", line)
                if id_match:
                    entry["source_id"] = id_match.group(1)
                if current_section == "citing":
                    citing.append(entry)
                else:
                    cited_by.append(entry)

    return citing, cited_by


def _parse_biblio_coupling(markdown: str) -> list[dict[str, Any]]:
    """Parse bibliographic coupling markdown.

    Expected format::

        - **Title** (2021)
          - Author
          - Coupling: **45.2%** (8 shared refs)
          - ID: `src-xxx`

    Args:
        markdown: Raw markdown from research_kb_biblio_coupling.

    Returns:
        List of similar papers with coupling scores.
    """
    if not markdown or not markdown.strip():
        return []

    papers: list[dict[str, Any]] = []

    for line in markdown.split("\n"):
        if line.startswith("- **"):
            title_match = re.match(r"- \*\*(.+?)\*\*\s*\((\d{4})\)", line)
            coupling_match = re.search(r"Coupling:\s*\*\*(.+?)%\*\*", line)
            id_match = re.search(r"ID:\s*`([^`]+)`", line)

            if title_match:
                paper: dict[str, Any] = {
                    "title": title_match.group(1),
                    "year": title_match.group(2),
                }
                if coupling_match:
                    try:
                        paper["coupling_pct"] = float(coupling_match.group(1))
                    except ValueError:
                        logger.warning("Non-numeric coupling in: %s", line)
                if id_match:
                    paper["source_id"] = id_match.group(1)
                papers.append(paper)

    return papers


async def citation_analyzer(
    state: ResearchState,
    config: AgentConfig,
    mcp: ResearchKBClient,
) -> NodeUpdate:
    """Analyze citation networks for top search results.

    Strategy:
        1. Take source IDs from top search results
        2. Get citation network (who cites / is cited by)
        3. Find bibliographically similar papers

    Args:
        state: Current state with search_results populated.
        config: Agent configuration.
        mcp: Connected MCP client.

    Returns:
        NodeUpdate with ``citations`` and ``citation_summary``.
    """
    logger.info("Analyzing citation networks")

    citations: list[CitationInfo] = []
    analyzed_ids: set[str] = set()

    # Analyze top search results by score
    top_results = [r for r in state.search_results if r.source_id][:5]

    try:
        async with asyncio.timeout(45):
            for result in top_results:
                if result.source_id in analyzed_ids:
                    continue
                analyzed_ids.add(result.source_id)

                citing: list[dict[str, str]] = []
                cited_by: list[dict[str, str]] = []
                similar: list[dict[str, Any]] = []

                # Pair-parallel: citation_network + biblio_coupling for same source
                # Sequential across results to avoid retry-storm from citation_network bug
                try:
                    net_raw, bib_raw = await asyncio.gather(
                        mcp.citation_network(
                            source_id=result.source_id,
                            limit=config.max_citations,
                        ),
                        mcp.biblio_coupling(
                            source_id=result.source_id,
                            limit=10,
                        ),
                        return_exceptions=True,
                    )

                    if not isinstance(net_raw, BaseException):
                        citing, cited_by = _parse_citation_network(net_raw)
                        logger.info(
                            "Citation network for '%s': %d citing, %d cited-by",
                            result.title[:50],
                            len(citing),
                            len(cited_by),
                        )
                    else:
                        logger.warning(
                            "Citation network failed for %s: %s", result.source_id, net_raw
                        )

                    if not isinstance(bib_raw, BaseException):
                        similar = _parse_biblio_coupling(bib_raw)
                        logger.info(
                            "Found %d bibliographically similar papers for '%s'",
                            len(similar),
                            result.title[:50],
                        )
                    else:
                        logger.warning(
                            "Biblio coupling failed for %s: %s", result.source_id, bib_raw
                        )
                except Exception as e:
                    logger.warning("Citation analysis failed for %s: %s", result.source_id, e)

                citations.append(
                    CitationInfo(
                        source_id=result.source_id,
                        source_title=result.title,
                        citing=citing,
                        cited_by=cited_by,
                        similar_papers=similar,
                    )
                )

    except TimeoutError:
        logger.warning("Citation analysis timed out with %d sources analyzed", len(citations))

    total_citing = sum(len(c.citing) for c in citations)
    total_cited = sum(len(c.cited_by) for c in citations)
    total_similar = sum(len(c.similar_papers) for c in citations)
    summary = (
        f"Analyzed {len(citations)} sources: "
        f"{total_citing} citing, {total_cited} cited-by, "
        f"{total_similar} similar papers."
    )
    logger.info(summary)

    return NodeUpdate(
        citations=citations,
        citation_summary=summary,
        current_node="citation_analyzer",
    )
