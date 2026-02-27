"""Tests for the exception hierarchy."""

from __future__ import annotations

from research_agent.exceptions import (
    LLMParsingError,
    MCPConnectionError,
    MCPToolError,
    PlannerError,
    ResearchAgentError,
    SearchError,
    SynthesisError,
)


class TestExceptionHierarchy:
    """All exceptions inherit from ResearchAgentError."""

    def test_mcp_connection_error(self) -> None:
        assert issubclass(MCPConnectionError, ResearchAgentError)

    def test_mcp_tool_error(self) -> None:
        assert issubclass(MCPToolError, ResearchAgentError)

    def test_llm_parsing_error(self) -> None:
        assert issubclass(LLMParsingError, ResearchAgentError)

    def test_planner_error(self) -> None:
        assert issubclass(PlannerError, ResearchAgentError)

    def test_search_error(self) -> None:
        assert issubclass(SearchError, ResearchAgentError)

    def test_synthesis_error(self) -> None:
        assert issubclass(SynthesisError, ResearchAgentError)


class TestMCPToolError:
    """MCPToolError captures structured error info."""

    def test_attributes(self) -> None:
        err = MCPToolError(tool_name="research_kb_search", detail="timeout")
        assert err.tool_name == "research_kb_search"
        assert err.detail == "timeout"

    def test_message_format(self) -> None:
        err = MCPToolError(tool_name="research_kb_search", detail="not found")
        assert "research_kb_search" in str(err)
        assert "not found" in str(err)

    def test_catchable_as_base(self) -> None:
        """Can catch MCPToolError as ResearchAgentError."""
        try:
            raise MCPToolError(tool_name="t", detail="d")
        except ResearchAgentError:
            pass  # Expected


class TestResearchAgentError:
    """Base exception behaves as standard Exception."""

    def test_message(self) -> None:
        err = ResearchAgentError("something broke")
        assert str(err) == "something broke"

    def test_is_exception(self) -> None:
        assert issubclass(ResearchAgentError, Exception)
