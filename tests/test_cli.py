"""Tests for CLI entry point.

Tests argument parsing, error handling, and exit codes.
Uses subprocess for integration-level tests.
"""

from __future__ import annotations

import subprocess
import sys


class TestCLIArgParsing:
    """Test CLI argument parsing via subprocess."""

    def test_help_flag(self) -> None:
        """--help prints usage and exits 0."""
        result = subprocess.run(
            [sys.executable, "-m", "research_agent.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "research question" in result.stdout.lower()

    def test_no_args_exits_error(self) -> None:
        """No arguments exits with error (missing query)."""
        result = subprocess.run(
            [sys.executable, "-m", "research_agent.cli"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode != 0

    def test_verbose_flag_accepted(self) -> None:
        """--verbose is a recognized flag (even if query fails later)."""
        result = subprocess.run(
            [sys.executable, "-m", "research_agent.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert "--verbose" in result.stdout or "-v" in result.stdout

    def test_output_flag_accepted(self) -> None:
        """--output is a recognized flag."""
        result = subprocess.run(
            [sys.executable, "-m", "research_agent.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert "--output" in result.stdout or "-o" in result.stdout

    def test_stream_flag_accepted(self) -> None:
        """--stream is a recognized flag."""
        result = subprocess.run(
            [sys.executable, "-m", "research_agent.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert "--stream" in result.stdout or "-s" in result.stdout


class TestCLIErrorHandling:
    """Test error handling paths."""

    def test_empty_query_rejected(self) -> None:
        """Empty query string is rejected."""
        result = subprocess.run(
            [sys.executable, "-m", "research_agent.cli", ""],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Should either exit 2 (argparse error) or exit 1 (validation error)
        assert result.returncode != 0

    def test_invalid_output_dir(self) -> None:
        """Output to nonexistent directory is rejected."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "research_agent.cli",
                "--output",
                "/nonexistent/path/report.md",
                "test query",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode != 0
