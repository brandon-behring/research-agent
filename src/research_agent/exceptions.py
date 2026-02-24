"""Exception hierarchy for the research agent.

Provides specific exception types for different failure modes,
enabling targeted error handling instead of broad ``except Exception`` catches.

Hierarchy::

    ResearchAgentError (base)
    +-- MCPConnectionError    — cannot connect to research-kb
    +-- MCPToolError          — MCP tool call failed (includes tool name)
    +-- LLMParsingError       — structured output parsing failed
    +-- PlannerError          — query decomposition failed
    +-- SearchError           — literature search failed
    +-- SynthesisError        — report generation failed
"""

from __future__ import annotations


class ResearchAgentError(Exception):
    """Base exception for all research agent errors."""


class MCPConnectionError(ResearchAgentError):
    """Failed to connect to the MCP server."""


class MCPToolError(ResearchAgentError):
    """An MCP tool call returned an error.

    Attributes:
        tool_name: The MCP tool that failed.
        detail: Error detail from the server.
    """

    def __init__(self, tool_name: str, detail: str) -> None:
        self.tool_name = tool_name
        self.detail = detail
        super().__init__(f"MCP tool '{tool_name}' failed: {detail}")


class LLMParsingError(ResearchAgentError):
    """Structured output parsing from the LLM failed."""


class PlannerError(ResearchAgentError):
    """Query planner node failed to decompose the research question."""


class SearchError(ResearchAgentError):
    """Literature search or concept search failed."""


class SynthesisError(ResearchAgentError):
    """Report synthesis failed."""
