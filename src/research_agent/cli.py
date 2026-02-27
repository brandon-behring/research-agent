"""CLI entry point for the research agent.

Usage:
    research-agent "What are the assumptions of double machine learning?"
    research-agent --verbose -o report.md "Compare DML and IV"
    research-agent --clear-cache
    research-agent --health-check
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
from research_agent.mcp_client import ResearchKBClient


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


def _build_json_output(result: dict[str, Any], config: AgentConfig) -> dict[str, Any]:
    """Build structured JSON output from pipeline result.

    Assembles report, metadata, and configuration into a single JSON-serializable
    dict for machine-readable output (``--json`` flag).

    Args:
        result: Final state dict from run_research().
        config: Agent configuration used for this run.

    Returns:
        Dict with report, metadata, and config keys.
    """
    metadata = _extract_metadata(result)
    metadata["concept_count"] = len(result.get("concepts", []))
    metadata["citation_count"] = len(result.get("citations", []))
    metadata["confidence_level"] = result.get("confidence_assessment", "")
    metadata["kb_stats"] = result.get("kb_stats_summary", "")
    metadata["kb_domains"] = result.get("kb_domains", [])
    metadata["similar_concepts_count"] = len(result.get("similar_concepts", []))
    metadata["cross_domain_matches_count"] = len(result.get("cross_domain_matches", []))
    metadata["connection_count"] = len(result.get("connection_explanations", []))

    config_summary = {
        "max_search_results": config.max_search_results,
        "max_concepts": config.max_concepts,
        "max_citations": config.max_citations,
        "synthesis_model": config.models.synthesis,
        "planning_model": config.models.planning,
    }

    return {
        "report": result.get("report", ""),
        "metadata": metadata,
        "config": config_summary,
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
            timing = ""
            if event.duration_ms is not None:
                timing = f" ({event.duration_ms / 1000:.1f}s)"
            print(
                f"  [{event.node_name}] {event.data}{timing}",
                file=sys.stderr,
            )
        elif event.event_type == "report_chunk":
            report = event.data
        elif event.event_type == "complete":
            timing = ""
            if event.duration_ms is not None:
                timing = f" in {event.duration_ms / 1000:.1f}s"
            print(f"  {event.data}{timing}", file=sys.stderr)
    return {"report": report}


async def _run_health_check(config: AgentConfig) -> bool:
    """Connect to MCP, run a test search, and report status.

    Args:
        config: Agent configuration (MCP transport settings).

    Returns:
        True if health check passes, False otherwise.
    """
    import time

    print("Health check: connecting to research-kb...", file=sys.stderr)
    start = time.monotonic()
    try:
        async with ResearchKBClient(config.mcp) as mcp:
            elapsed_connect = time.monotonic() - start
            print(
                f"  Connected ({elapsed_connect:.1f}s)",
                file=sys.stderr,
            )

            search_start = time.monotonic()
            result = await mcp.fast_search("test", limit=1)
            elapsed_search = time.monotonic() - search_start
            # Result should be non-empty string
            ok = bool(result and result.strip())
            status = "OK" if ok else "WARN: empty response"
            print(
                f"  fast_search: {status} ({elapsed_search:.1f}s)",
                file=sys.stderr,
            )

            # Optional: fetch stats for corpus info
            try:
                stats_raw = await mcp.stats()
                if stats_raw:
                    # Extract first few lines for summary
                    summary = stats_raw.strip().split("\n")[0]
                    print(f"  KB: {summary}", file=sys.stderr)
            except Exception:
                pass  # Stats are optional for health check

        total = time.monotonic() - start
        print(f"  Total: {total:.1f}s — HEALTHY", file=sys.stderr)
        return True
    except Exception as e:
        total = time.monotonic() - start
        print(
            f"  UNHEALTHY after {total:.1f}s: {e}",
            file=sys.stderr,
        )
        return False


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
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Check MCP connection and exit (no query required)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output structured JSON instead of markdown report",
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

    # Handle --health-check early exit (no query required)
    if args.health_check:
        ok = asyncio.run(_run_health_check(config))
        sys.exit(0 if ok else 1)

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

    if args.json:
        json_output = _build_json_output(result, config)
        output_str = json.dumps(json_output, indent=2)
        if args.output:
            with open(args.output, "w") as f:
                f.write(output_str)
            print(f"JSON report written to {args.output}", file=sys.stderr)
        else:
            print(output_str)
    else:
        _output_report(report, args)


if __name__ == "__main__":
    main()
