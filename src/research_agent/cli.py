"""CLI entry point for the research agent.

Usage:
    research-agent "What are the assumptions of double machine learning?"
    research-agent --verbose -o report.md "Compare DML and IV"
    research-agent --clear-cache
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

from research_agent.cache import ReportCache, compute_cache_key
from research_agent.config import AgentConfig
from research_agent.exceptions import ResearchAgentError
from research_agent.graph import run_research, stream_research


def _extract_metadata(result: dict[str, Any]) -> dict[str, Any]:
    """Extract cache metadata from pipeline result.

    Uses getattr for safe access — works with both Pydantic models
    (real pipeline) and plain dicts (tests).

    Args:
        result: Final state dict from run_research().

    Returns:
        Dict with source_count, concept_names, methods_audited.
    """
    return {
        "source_count": len(result.get("search_results", [])),
        "concept_names": [getattr(c, "name", "") for c in result.get("concepts", [])],
        "methods_audited": [
            getattr(a, "method_name", "") for a in result.get("assumption_audits", [])
        ],
    }


def _output_report(report: str, args: argparse.Namespace) -> None:
    """Write report to file or stdout.

    Args:
        report: Report text to output.
        args: Parsed CLI arguments (checks args.output).
    """
    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(report)


async def _run_streaming(query: str, config: AgentConfig) -> dict[str, str]:
    """Run the pipeline in streaming mode, printing progress to stderr.

    Args:
        query: Research question to analyze.
        config: Agent configuration.

    Returns:
        Dict with 'report' key, matching run_research() contract.
    """
    report = ""
    async for event in stream_research(query, config):
        if event.event_type == "node_end":
            print(f"  [{event.node_name}] {event.data}", file=sys.stderr)
        elif event.event_type == "report_chunk":
            report = event.data
        elif event.event_type == "complete":
            print(f"  {event.data}", file=sys.stderr)
    return {"report": report}


def main() -> None:
    """CLI entry point.

    Exit codes:
        0: Success
        1: Research agent error (known failure)
        2: Unexpected error
        130: Keyboard interrupt
    """
    parser = argparse.ArgumentParser(
        description="Multi-agent research analysis powered by LangGraph and MCP",
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="",
        help="Research question to analyze",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Write report to file (default: stdout)",
    )
    parser.add_argument(
        "--stream",
        "-s",
        action="store_true",
        help="Stream progress events to stderr as pipeline executes",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass report cache for this query",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear all cached reports and exit",
    )

    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    config = AgentConfig()

    # Handle --clear-cache early exit (no query required)
    if args.clear_cache:
        with ReportCache(Path(config.cache_db_path).expanduser(), config.cache_ttl_hours) as cache:
            count = cache.clear()
        print(f"Cleared {count} cached reports.", file=sys.stderr)
        return

    # Validate inputs (after --clear-cache which doesn't need a query)
    if not args.query or not args.query.strip():
        parser.error("Query must not be empty.")

    if args.output:
        output_path = Path(args.output)
        if not output_path.parent.exists():
            parser.error(f"Output directory does not exist: {output_path.parent}")

    # ── Single with-block: cache lifecycle managed by context manager ─
    with ReportCache(
        Path(config.cache_db_path).expanduser(),
        config.cache_ttl_hours,
        enabled=config.cache_enabled and not args.no_cache,
    ) as cache:
        cache_key = compute_cache_key(
            args.query,
            config.max_search_results,
            config.max_concepts,
            config.max_citations,
            config.models.synthesis,
        )
        entry = cache.get(cache_key)
        if entry is not None:
            print("[cached]", file=sys.stderr)
            _output_report(entry.report, args)
            return

        # ── Run pipeline ──────────────────────────────────────────────
        try:
            if args.stream:
                result = asyncio.run(_run_streaming(args.query, config))
            else:
                result = asyncio.run(run_research(args.query, config))
        except ResearchAgentError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        except KeyboardInterrupt:
            print("\nInterrupted.", file=sys.stderr)
            sys.exit(130)
        except Exception as e:
            print(f"Unexpected error: {e}", file=sys.stderr)
            sys.exit(2)

        report = result.get("report", "No report generated.")

        # ── Cache write ───────────────────────────────────────────────
        metadata = _extract_metadata(result)
        config_summary = {
            "max_search_results": config.max_search_results,
            "max_concepts": config.max_concepts,
            "max_citations": config.max_citations,
            "synthesis_model": config.models.synthesis,
        }
        cache.put(
            cache_key,
            args.query,
            report,
            json.dumps(metadata),
            json.dumps(config_summary),
        )
        cache.evict_expired()

    _output_report(report, args)


if __name__ == "__main__":
    main()
