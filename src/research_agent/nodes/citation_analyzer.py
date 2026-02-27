"""Citation Analyzer node -- maps citation networks and finds related papers.

Takes source IDs from search results, builds citation chains (who cites whom),
and discovers related papers through bibliographic coupling (shared references).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from research_agent.config import AgentConfig
from research_agent.mcp_client import ResearchKBClient
from research_agent.parsing import parse_json_first
from research_agent.state import CitationInfo, NodeUpdate, ResearchState, SearchResult

logger = logging.getLogger(__name__)

# Timeout for all citation analysis (network + biblio coupling).
_CITATION_TIMEOUT_SECONDS = 45

# Maximum search results to analyze for citation networks.
_TOP_RESULTS_LIMIT = 5

# Max concurrent source analyses. Conservative: each source fires 2 MCP calls,
# so limit=2 means max 4 concurrent MCP calls. Avoids retry storms.
_CITATION_CONCURRENCY_LIMIT = 2


def _parse_citation_network_json(raw: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Parse JSON citation network into citing/cited-by lists.

    Expected JSON schema::

        {
            "citing": [{"title": "...", "year": 2021, "source_id": "..."}],
            "cited_by": [{"title": "...", "year": 2017, "source_id": "..."}]
        }

    Args:
        raw: JSON string from research_kb_citation_network.

    Returns:
        Tuple of (citing_sources, cited_by_sources).

    Raises:
        json.JSONDecodeError: If raw is not valid JSON.
    """
    data = json.loads(raw)
    citing = [
        {
            "title": c.get("title", ""),
            "year": str(c.get("year", "")),
            "source_id": c.get("source_id", ""),
        }
        for c in data.get("citing", [])
    ]
    cited_by = [
        {
            "title": c.get("title", ""),
            "year": str(c.get("year", "")),
            "source_id": c.get("source_id", ""),
        }
        for c in data.get("cited_by", [])
    ]
    return citing, cited_by


def _parse_citation_network_markdown(
    markdown: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Parse citation network markdown into citing/cited-by lists.

    Retained as fallback when JSON parsing fails. Uses a state machine
    to track the current entry across lines.

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
    citing: list[dict[str, str]] = []
    cited_by: list[dict[str, str]] = []

    current_section: str | None = None
    current_entry: dict[str, str] | None = None

    def _flush_entry() -> None:
        """Append current_entry to the appropriate section list."""
        nonlocal current_entry
        if current_entry is not None and current_section is not None:
            target = citing if current_section == "citing" else cited_by
            target.append(current_entry)
            current_entry = None

    for line in markdown.split("\n"):
        if "Citing This Source" in line:
            _flush_entry()
            current_section = "citing"
        elif "Cited By This Source" in line:
            _flush_entry()
            current_section = "cited_by"
        elif line.startswith("- **") and current_section:
            _flush_entry()
            title_match = re.match(r"- \*\*(.+?)\*\*\s*\((\d{4})\)", line)
            if title_match:
                current_entry = {"title": title_match.group(1), "year": title_match.group(2)}
                # Also check the title line itself for inline ID
                id_match = re.search(r"ID:\s*`([^`]+)`", line)
                if id_match:
                    current_entry["source_id"] = id_match.group(1)
        elif current_entry is not None and line.strip().startswith("- "):
            # Continuation line: extract fields
            stripped = line.strip()
            id_match = re.search(r"ID:\s*`([^`]+)`", stripped)
            if id_match:
                current_entry["source_id"] = id_match.group(1)

    _flush_entry()
    return citing, cited_by


def _parse_citation_network(raw: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Parse citation network with JSON-first strategy and markdown fallback.

    Args:
        raw: Raw response string (JSON or markdown) from research-kb.

    Returns:
        Tuple of (citing_sources, cited_by_sources).
    """
    if not raw or not raw.strip():
        return [], []
    return parse_json_first(
        raw,
        _parse_citation_network_json,
        _parse_citation_network_markdown,
        context="citation network",
    )


def _parse_biblio_coupling_json(raw: str) -> list[dict[str, Any]]:
    """Parse JSON bibliographic coupling response.

    Expected JSON schema::

        {
            "similar": [
                {
                    "title": "...", "year": 2021, "source_id": "...",
                    "coupling_strength": 0.452, "shared_references": 8
                }
            ]
        }

    Note: ``coupling_strength`` (0.0-1.0) is converted to ``coupling_pct`` (0-100)
    for downstream compatibility (``synthesis.py:135``).

    Args:
        raw: JSON string from research_kb_biblio_coupling.

    Returns:
        List of similar papers with coupling scores.

    Raises:
        json.JSONDecodeError: If raw is not valid JSON.
    """
    data = json.loads(raw)
    papers: list[dict[str, Any]] = []
    for item in data.get("similar", []):
        paper: dict[str, Any] = {
            "title": item.get("title", ""),
            "year": str(item.get("year", "")),
        }
        if "source_id" in item:
            paper["source_id"] = item["source_id"]
        if "coupling_strength" in item:
            paper["coupling_pct"] = item["coupling_strength"] * 100  # 0.452 → 45.2
        if "shared_references" in item:
            paper["shared_references"] = item["shared_references"]
        papers.append(paper)
    return papers


def _parse_biblio_coupling_markdown(markdown: str) -> list[dict[str, Any]]:
    """Parse bibliographic coupling markdown.

    Retained as fallback when JSON parsing fails. Uses a state machine
    to track the current paper across lines.

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
    papers: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for line in markdown.split("\n"):
        if line.startswith("- **"):
            if current is not None:
                papers.append(current)
            title_match = re.match(r"- \*\*(.+?)\*\*\s*\((\d{4})\)", line)
            if title_match:
                current = {
                    "title": title_match.group(1),
                    "year": title_match.group(2),
                }
                # Also check the title line itself for inline fields
                coupling_match = re.search(r"Coupling:\s*\*\*(.+?)%\*\*", line)
                id_match = re.search(r"ID:\s*`([^`]+)`", line)
                if coupling_match:
                    try:
                        current["coupling_pct"] = float(coupling_match.group(1))
                    except ValueError:
                        logger.warning("Non-numeric coupling in: %s", line)
                if id_match:
                    current["source_id"] = id_match.group(1)
            else:
                current = None
        elif current is not None and line.strip().startswith("- "):
            # Continuation line: extract fields
            stripped = line.strip()
            coupling_match = re.search(r"Coupling:\s*\*\*(.+?)%\*\*", stripped)
            id_match = re.search(r"ID:\s*`([^`]+)`", stripped)
            if coupling_match:
                try:
                    current["coupling_pct"] = float(coupling_match.group(1))
                except ValueError:
                    logger.warning("Non-numeric coupling in: %s", stripped)
            if id_match:
                current["source_id"] = id_match.group(1)

    if current is not None:
        papers.append(current)

    return papers


def _parse_biblio_coupling(raw: str) -> list[dict[str, Any]]:
    """Parse biblio coupling with JSON-first strategy and markdown fallback.

    Args:
        raw: Raw response string (JSON or markdown) from research-kb.

    Returns:
        List of similar papers with coupling scores.
    """
    if not raw or not raw.strip():
        return []
    return parse_json_first(
        raw,
        _parse_biblio_coupling_json,
        _parse_biblio_coupling_markdown,
        context="biblio coupling",
    )


async def _analyze_one_source(
    mcp: ResearchKBClient,
    source_id: str,
    source_title: str,
    config: AgentConfig,
) -> CitationInfo:
    """Analyze citation network + biblio coupling for one source.

    Fires both MCP calls in parallel (pair-parallel) for a single source.

    Args:
        mcp: Connected MCP client.
        source_id: Source ID to analyze.
        source_title: Source title for logging/output.
        config: Agent configuration (for max_citations).

    Returns:
        CitationInfo with citing, cited_by, and similar_papers.
    """
    citing: list[dict[str, str]] = []
    cited_by: list[dict[str, str]] = []
    similar: list[dict[str, Any]] = []

    try:
        net_raw, bib_raw = await asyncio.gather(
            mcp.citation_network(source_id=source_id, limit=config.max_citations),
            mcp.biblio_coupling(source_id=source_id, limit=10),
            return_exceptions=True,
        )

        if not isinstance(net_raw, BaseException):
            citing, cited_by = _parse_citation_network(net_raw)
            logger.info(
                "Citation network for '%s': %d citing, %d cited-by",
                source_title[:50],
                len(citing),
                len(cited_by),
            )
        else:
            logger.warning("Citation network failed for %s: %s", source_id, net_raw)

        if not isinstance(bib_raw, BaseException):
            similar = _parse_biblio_coupling(bib_raw)
            logger.info(
                "Found %d similar papers for '%s'",
                len(similar),
                source_title[:50],
            )
        else:
            logger.warning("Biblio coupling failed for %s: %s", source_id, bib_raw)
    except Exception as e:
        logger.warning("Citation analysis failed for %s: %s", source_id, e)

    return CitationInfo(
        source_id=source_id,
        source_title=source_title,
        citing=citing,
        cited_by=cited_by,
        similar_papers=similar,
    )


async def citation_analyzer(
    state: ResearchState,
    config: AgentConfig,
    mcp: ResearchKBClient,
) -> NodeUpdate:
    """Analyze citation networks for top search results.

    Uses semaphore-bounded concurrency across sources (``_CITATION_CONCURRENCY_LIMIT``),
    with pair-parallel MCP calls within each source. This is 2x faster than the
    previous sequential approach while keeping concurrency conservative enough
    to avoid retry storms.

    Args:
        state: Current state with search_results populated.
        config: Agent configuration.
        mcp: Connected MCP client.

    Returns:
        NodeUpdate with ``citations`` and ``citation_summary``.
    """
    logger.info("Analyzing citation networks")

    citations: list[CitationInfo] = []

    # Deduplicate source_ids, take top N by score
    top_results = [r for r in state.search_results if r.source_id][:_TOP_RESULTS_LIMIT]
    unique_results = []
    seen: set[str] = set()
    for r in top_results:
        if r.source_id not in seen:
            seen.add(r.source_id)
            unique_results.append(r)

    sem = asyncio.Semaphore(_CITATION_CONCURRENCY_LIMIT)

    async def _bounded_analyze(result: SearchResult) -> CitationInfo:
        async with sem:
            return await _analyze_one_source(mcp, result.source_id, result.title, config)

    try:
        async with asyncio.timeout(_CITATION_TIMEOUT_SECONDS):
            batch = await asyncio.gather(
                *[_bounded_analyze(r) for r in unique_results],
                return_exceptions=True,
            )
            for i, item in enumerate(batch):
                if isinstance(item, BaseException):
                    logger.warning(
                        "Unexpected error for '%s': %s",
                        unique_results[i].source_id,
                        item,
                    )
                else:
                    citations.append(item)

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
