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


class TestHealthCheck:
    """Tests for --health-check CLI flag."""

    def test_health_check_flag_accepted(self) -> None:
        """--health-check is a recognized flag."""
        result = subprocess.run(
            [sys.executable, "-m", "research_agent.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert "--health-check" in result.stdout

    def test_health_check_success_exits_0(self) -> None:
        """Successful health check exits 0."""
        with (
            patch(
                "research_agent.cli._run_health_check",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch("sys.argv", ["cli", "--health-check"]),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == 0

    def test_health_check_failure_exits_1(self) -> None:
        """Failed health check exits 1."""
        with (
            patch(
                "research_agent.cli._run_health_check",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch("sys.argv", ["cli", "--health-check"]),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == 1

    @pytest.mark.asyncio
    async def test_run_health_check_success(self) -> None:
        """_run_health_check returns True when MCP responds."""
        from research_agent.cli import _run_health_check
        from research_agent.config import AgentConfig, MCPConfig

        config = AgentConfig(
            mcp=MCPConfig(transport="stdio", research_kb_path="/fake"),
        )
        mock_mcp = AsyncMock()
        mock_mcp.fast_search.return_value = "some results"
        mock_mcp.stats.return_value = "## Stats\n- Sources: 100"
        with (
            patch(
                "research_agent.cli.ResearchKBClient.__aenter__",
                return_value=mock_mcp,
            ),
            patch(
                "research_agent.cli.ResearchKBClient.__aexit__",
                return_value=None,
            ),
        ):
            result = await _run_health_check(config)
        assert result is True
        mock_mcp.fast_search.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_health_check_connection_failure(self) -> None:
        """_run_health_check returns False on connection failure."""
        from research_agent.cli import _run_health_check
        from research_agent.config import AgentConfig, MCPConfig

        config = AgentConfig(
            mcp=MCPConfig(transport="stdio", research_kb_path="/fake"),
        )
        with patch(
            "research_agent.cli.ResearchKBClient.__aenter__",
            side_effect=RuntimeError("connection refused"),
        ):
            result = await _run_health_check(config)
        assert result is False


class TestRunStreaming:
    """Tests for _run_streaming() async helper."""

    @pytest.mark.asyncio
    async def test_collects_report_from_stream_events(self) -> None:
        """_run_streaming collects report and prints progress."""
        from research_agent.cli import _run_streaming
        from research_agent.config import AgentConfig
        from research_agent.graph import StreamEvent

        events = [
            StreamEvent(
                event_type="node_end",
                node_name="query_planner",
                data="Planned 2 sub-tasks",
                duration_ms=150,
            ),
            StreamEvent(
                event_type="node_end",
                node_name="synthesis",
                data="Generated report",
            ),
            StreamEvent(
                event_type="report_chunk",
                node_name="synthesis",
                data="# Final Report",
            ),
            StreamEvent(
                event_type="complete",
                node_name="",
                data="Report: 14 chars",
                duration_ms=3000,
            ),
        ]

        async def fake_stream(query, config):
            for e in events:
                yield e

        config = AgentConfig()
        with patch(
            "research_agent.cli.stream_research",
            side_effect=fake_stream,
        ):
            result = await _run_streaming("test query", config)

        assert result["report"] == "# Final Report"


class TestJSONOutput:
    """Tests for --json structured output."""

    def test_json_flag_accepted(self) -> None:
        """--json is a recognized flag."""
        result = subprocess.run(
            [sys.executable, "-m", "research_agent.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert "--json" in result.stdout

    def test_json_output_is_valid_json(self) -> None:
        """--json produces valid JSON output."""
        import json

        mock_result = {
            "report": "# Test Report",
            "search_results": [],
            "concepts": [],
            "assumption_audits": [],
            "citations": [],
            "confidence_assessment": "high",
            "kb_stats_summary": "100 sources",
            "kb_domains": ["causal_inference"],
            "similar_concepts": [],
            "cross_domain_matches": [],
            "connection_explanations": [],
        }
        with (
            patch(
                "research_agent.cli.run_research",
                new_callable=AsyncMock,
            ) as mock_run,
            patch(
                "sys.argv",
                ["cli", "--no-cache", "--json", "What is DML?"],
            ),
            patch("sys.stdout", new_callable=StringIO) as mock_stdout,
        ):
            mock_run.return_value = mock_result
            main()
            output = mock_stdout.getvalue()

        parsed = json.loads(output)
        assert parsed["report"] == "# Test Report"
        assert "metadata" in parsed
        assert "config" in parsed
        assert parsed["metadata"]["source_count"] == 0
        assert parsed["metadata"]["confidence_level"] == "high"
        assert parsed["metadata"]["kb_domains"] == ["causal_inference"]

    def test_json_output_to_file(self, tmp_path: pytest.TempPathFactory) -> None:
        """--json --output writes JSON to file."""
        import json

        out_file = tmp_path / "report.json"  # type: ignore[operator]
        mock_result = {
            "report": "# File Report",
            "search_results": [],
            "concepts": [],
            "assumption_audits": [],
            "citations": [],
        }
        with (
            patch(
                "research_agent.cli.run_research",
                new_callable=AsyncMock,
            ) as mock_run,
            patch(
                "sys.argv",
                [
                    "cli",
                    "--no-cache",
                    "--json",
                    "--output",
                    str(out_file),
                    "What is DML?",
                ],
            ),
        ):
            mock_run.return_value = mock_result
            main()
        parsed = json.loads(out_file.read_text())
        assert parsed["report"] == "# File Report"

    def test_build_json_output_structure(self) -> None:
        """_build_json_output returns complete structure."""
        from research_agent.cli import _build_json_output
        from research_agent.config import AgentConfig

        result = {
            "report": "# Report",
            "search_results": [1, 2, 3],
            "concepts": [type("C", (), {"name": "DML"})()],
            "citations": [1, 2],
            "assumption_audits": [
                type("A", (), {"method_name": "IV"})(),
            ],
            "confidence_assessment": "medium",
            "kb_stats_summary": "500 sources",
            "kb_domains": ["stats", "econ"],
            "similar_concepts": [{"name": "X"}],
            "cross_domain_matches": [],
            "connection_explanations": [{"path": "A→B"}],
        }
        config = AgentConfig()
        output = _build_json_output(result, config)

        assert output["report"] == "# Report"
        assert output["metadata"]["source_count"] == 3
        assert output["metadata"]["concept_count"] == 1
        assert output["metadata"]["citation_count"] == 2
        assert output["metadata"]["methods_audited"] == ["IV"]
        assert output["metadata"]["confidence_level"] == "medium"
        assert output["metadata"]["kb_stats"] == "500 sources"
        assert output["metadata"]["similar_concepts_count"] == 1
        assert output["metadata"]["connection_count"] == 1
        assert "max_search_results" in output["config"]
        assert "planning_model" in output["config"]


class TestErrorFormatting:
    """Tests for _format_error actionable hints."""

    def test_mcp_connection_error_hint(self) -> None:
        """MCPConnectionError includes connection hints."""
        from research_agent.cli import _format_error
        from research_agent.exceptions import MCPConnectionError

        err = MCPConnectionError("connection refused")
        msg = _format_error(err)
        assert "RESEARCH_KB_PATH" in msg
        assert "health-check" in msg

    def test_search_error_no_results_hint(self) -> None:
        """SearchError with 'no results' includes search hints."""
        from research_agent.cli import _format_error
        from research_agent.exceptions import SearchError

        err = SearchError("Literature search timed out with no results")
        msg = _format_error(err)
        assert "broader search terms" in msg

    def test_node_timeout_error_hint(self) -> None:
        """NodeTimeoutError includes timeout hints."""
        from research_agent.cli import _format_error
        from research_agent.exceptions import NodeTimeoutError

        err = NodeTimeoutError("synthesis", 180)
        msg = _format_error(err)
        assert "synthesis" in msg
        assert "180s" in msg
        assert "NODE_TIMEOUTS" in msg

    def test_generic_error_no_hint(self) -> None:
        """Generic exception has no special hint."""
        from research_agent.cli import _format_error

        err = ValueError("something broke")
        msg = _format_error(err)
        assert "something broke" in msg
        assert "Hint:" not in msg

    def test_verbose_includes_traceback(self) -> None:
        """verbose=True appends traceback info."""
        from research_agent.cli import _format_error

        try:
            raise ValueError("test error")
        except ValueError as e:
            msg = _format_error(e, verbose=True)
        assert "Traceback" in msg
        assert "test error" in msg
