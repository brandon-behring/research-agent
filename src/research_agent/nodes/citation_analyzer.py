"""Citation Analyzer node — maps citation networks and finds related papers.

Takes source IDs from search results, builds citation chains (who cites whom),
and discovers related papers through bibliographic coupling (shared references).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from research_agent.config import AgentConfig
from research_agent.mcp_client import ResearchKBClient
from research_agent.state import CitationInfo, ResearchState

logger = logging.getLogger(__name__)


def _parse_citation_network(markdown: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Parse citation network markdown into citing/cited-by lists.

    Args:
        markdown: Raw markdown from research_kb_citation_network.

    Returns:
        Tuple of (citing_sources, cited_by_sources).
    """
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

    Args:
        markdown: Raw markdown from research_kb_biblio_coupling.

    Returns:
        List of similar papers with coupling scores.
    """
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
                    paper["coupling_pct"] = float(coupling_match.group(1))
                if id_match:
                    paper["source_id"] = id_match.group(1)
                papers.append(paper)

    return papers


async def citation_analyzer(
    state: ResearchState,
    config: AgentConfig,
    mcp: ResearchKBClient,
) -> dict[str, Any]:
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
        Dict with 'citations' and 'citation_summary' updates.
    """
    logger.info("Analyzing citation networks")

    citations: list[CitationInfo] = []
    analyzed_ids: set[str] = set()

    # Analyze top search results by score
    top_results = [r for r in state.search_results if r.source_id][:5]

    for result in top_results:
        if result.source_id in analyzed_ids:
            continue
        analyzed_ids.add(result.source_id)

        info = CitationInfo(source_id=result.source_id, source_title=result.title)

        # Get citation network
        try:
            raw = await mcp.citation_network(
                source_id=result.source_id,
                limit=config.max_citations,
            )
            citing, cited_by = _parse_citation_network(raw)
            info.citing = citing
            info.cited_by = cited_by
            logger.info(
                "Citation network for '%s': %d citing, %d cited-by",
                result.title[:50],
                len(citing),
                len(cited_by),
            )
        except Exception as e:
            logger.warning("Citation network failed for %s: %s", result.source_id, e)

        # Get bibliographic coupling
        try:
            raw = await mcp.biblio_coupling(
                source_id=result.source_id,
                limit=10,
            )
            info.similar_papers = _parse_biblio_coupling(raw)
            logger.info(
                "Found %d bibliographically similar papers for '%s'",
                len(info.similar_papers),
                result.title[:50],
            )
        except Exception as e:
            logger.warning("Biblio coupling failed for %s: %s", result.source_id, e)

        citations.append(info)

    total_citing = sum(len(c.citing) for c in citations)
    total_cited = sum(len(c.cited_by) for c in citations)
    total_similar = sum(len(c.similar_papers) for c in citations)
    summary = (
        f"Analyzed {len(citations)} sources: "
        f"{total_citing} citing, {total_cited} cited-by, "
        f"{total_similar} similar papers."
    )
    logger.info(summary)

    return {
        "citations": citations,
        "citation_summary": summary,
        "current_node": "citation_analyzer",
    }
