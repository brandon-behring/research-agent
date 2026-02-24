"""CLI entry point for the research agent.

Usage:
    research-agent "What are the assumptions of double machine learning?"
    research-agent --verbose -o report.md "Compare DML and IV"
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from research_agent.config import AgentConfig
from research_agent.exceptions import ResearchAgentError
from research_agent.graph import run_research, stream_research


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

    args = parser.parse_args()

    # Validate inputs
    if not args.query.strip():
        parser.error("Query must not be empty.")

    if args.output:
        output_path = Path(args.output)
        if not output_path.parent.exists():
            parser.error(f"Output directory does not exist: {output_path.parent}")

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    try:
        config = AgentConfig()
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

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(report)


if __name__ == "__main__":
    main()
