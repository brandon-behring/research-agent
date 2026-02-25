"""Tests for CLI entry point.

Tests argument parsing, error handling, exit codes, and main() function paths.
Combines subprocess tests (arg parsing) with mocked unit tests (exit codes).
"""

from __future__ import annotations

import subprocess
import sys
from io import StringIO
from unittest.mock import AsyncMock, patch

import pytest

from research_agent.cli import main
from research_agent.exceptions import ResearchAgentError


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

    def test_no_cache_flag_accepted(self) -> None:
        """--no-cache is a recognized flag."""
        result = subprocess.run(
            [sys.executable, "-m", "research_agent.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert "--no-cache" in result.stdout

    def test_clear_cache_flag_accepted(self) -> None:
        """--clear-cache is a recognized flag."""
        result = subprocess.run(
            [sys.executable, "-m", "research_agent.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert "--clear-cache" in result.stdout


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


class TestCLIUnitLevel:
    """Unit tests for main() exit paths using mocked dependencies."""

    def test_successful_run_prints_report(self) -> None:
        """Successful run prints report to stdout."""
        mock_result = {"report": "# Test Report\n\nFindings here."}
        with (
            patch("research_agent.cli.run_research", new_callable=AsyncMock) as mock_run,
            patch("sys.argv", ["cli", "--no-cache", "What is DML?"]),
            patch("sys.stdout", new_callable=StringIO) as mock_stdout,
        ):
            mock_run.return_value = mock_result
            main()
            assert "Test Report" in mock_stdout.getvalue()

    def test_output_file_writes_report(self, tmp_path: pytest.TempPathFactory) -> None:
        """--output writes report to file."""
        out_file = tmp_path / "report.md"  # type: ignore[operator]
        mock_result = {"report": "# File Report"}
        with (
            patch("research_agent.cli.run_research", new_callable=AsyncMock) as mock_run,
            patch(
                "sys.argv",
                ["cli", "--no-cache", "--output", str(out_file), "What is DML?"],
            ),
        ):
            mock_run.return_value = mock_result
            main()
            assert out_file.read_text() == "# File Report"

    def test_research_agent_error_exits_1(self) -> None:
        """ResearchAgentError → exit code 1."""
        with (
            patch(
                "research_agent.cli.run_research",
                new_callable=AsyncMock,
                side_effect=ResearchAgentError("Test error"),
            ),
            patch("sys.argv", ["cli", "--no-cache", "test query"]),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == 1

    def test_generic_exception_exits_2(self) -> None:
        """Generic exception → exit code 2."""
        with (
            patch(
                "research_agent.cli.run_research",
                new_callable=AsyncMock,
                side_effect=ValueError("Unexpected"),
            ),
            patch("sys.argv", ["cli", "--no-cache", "test query"]),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == 2

    def test_keyboard_interrupt_exits_130(self) -> None:
        """KeyboardInterrupt → exit code 130."""
        with (
            patch(
                "research_agent.cli.run_research",
                new_callable=AsyncMock,
                side_effect=KeyboardInterrupt(),
            ),
            patch("sys.argv", ["cli", "--no-cache", "test query"]),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == 130

    def test_verbose_sets_debug_logging(self) -> None:
        """--verbose sets logging level to DEBUG."""
        mock_result = {"report": "Report"}
        with (
            patch("research_agent.cli.run_research", new_callable=AsyncMock) as mock_run,
            patch("sys.argv", ["cli", "--no-cache", "--verbose", "test"]),
            patch("logging.basicConfig") as mock_log_config,
            patch("sys.stdout", new_callable=StringIO),
        ):
            mock_run.return_value = mock_result
            main()
            # basicConfig should have been called with DEBUG level (10)
            call_kwargs = mock_log_config.call_args[1]
            assert call_kwargs["level"] == 10

    def test_streaming_mode_collects_report(self) -> None:
        """--stream mode collects report via stream_research."""
        mock_result = {"report": "# Streamed Report"}
        with (
            patch(
                "research_agent.cli._run_streaming",
                new_callable=AsyncMock,
            ) as mock_stream,
            patch("sys.argv", ["cli", "--no-cache", "--stream", "test query"]),
            patch("sys.stdout", new_callable=StringIO) as mock_stdout,
        ):
            mock_stream.return_value = mock_result
            main()
            assert "Streamed Report" in mock_stdout.getvalue()
