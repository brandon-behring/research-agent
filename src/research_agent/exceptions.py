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
    +-- NodeTimeoutError      — pipeline node exceeded its time limit
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


class NodeTimeoutError(ResearchAgentError):
    """A pipeline node exceeded its time limit.

    Attributes:
        node_name: The graph node that timed out.
        timeout_s: The timeout limit in seconds.
    """

    def __init__(self, node_name: str, timeout_s: int) -> None:
        self.node_name = node_name
        self.timeout_s = timeout_s
        super().__init__(f"Node '{node_name}' timed out after {timeout_s}s")
