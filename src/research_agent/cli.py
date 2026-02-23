"""CLI entry point for the research agent.

Usage:
    research-agent "What are the assumptions of double machine learning?"
    research-agent --model sonnet "Compare DML and instrumental variables"
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from research_agent.config import AgentConfig
from research_agent.graph import run_research


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-agent research analysis powered by LangGraph and MCP",
    )
    parser.add_argument(
        "query",
        help="Research question to analyze",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--output", "-o",
        help="Write report to file (default: stdout)",
    )

    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    config = AgentConfig.from_env()
    result = asyncio.run(run_research(args.query, config))

    report = result.get("report", "No report generated.")

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(report)


if __name__ == "__main__":
    main()
