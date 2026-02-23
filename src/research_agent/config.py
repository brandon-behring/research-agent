"""Configuration for model selection and MCP endpoints.

Design decisions:
    - Haiku for planning (fast, cheap — routing doesn't need Opus)
    - Sonnet for synthesis (strong reasoning for final report)
    - MCP endpoint configurable via env vars for Docker flexibility
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """LLM model selection per node type.

    Rationale: Cost/latency optimization. Planning nodes need speed,
    synthesis needs reasoning depth. This mirrors production patterns
    where you don't call the most expensive model for every step.
    """

    planning: str = "claude-haiku-4-5-20251001"
    synthesis: str = "claude-sonnet-4-6"
    analysis: str = "claude-sonnet-4-6"


@dataclass(frozen=True)
class MCPConfig:
    """Research-KB MCP server connection configuration.

    Supports two transports:
        - stdio: spawn research-kb as subprocess (local dev)
        - http: connect to running research-kb service (Docker)
    """

    transport: str = "stdio"
    research_kb_path: str = ""
    http_url: str = "http://research-kb:8000"

    @classmethod
    def from_env(cls) -> "MCPConfig":
        """Load configuration from environment variables."""
        transport = os.environ.get("MCP_TRANSPORT", "stdio")
        return cls(
            transport=transport,
            research_kb_path=os.environ.get("RESEARCH_KB_PATH", ""),
            http_url=os.environ.get("RESEARCH_KB_URL", "http://research-kb:8000"),
        )


@dataclass(frozen=True)
class AgentConfig:
    """Top-level agent configuration."""

    models: ModelConfig = ModelConfig()
    mcp: MCPConfig = MCPConfig()
    max_search_results: int = 10
    max_concepts: int = 15
    max_citations: int = 20

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Load full configuration from environment."""
        return cls(
            models=ModelConfig(
                planning=os.environ.get("PLANNING_MODEL", "claude-haiku-4-5-20251001"),
                synthesis=os.environ.get("SYNTHESIS_MODEL", "claude-sonnet-4-6"),
                analysis=os.environ.get("ANALYSIS_MODEL", "claude-sonnet-4-6"),
            ),
            mcp=MCPConfig.from_env(),
            max_search_results=int(os.environ.get("MAX_SEARCH_RESULTS", "10")),
            max_concepts=int(os.environ.get("MAX_CONCEPTS", "15")),
            max_citations=int(os.environ.get("MAX_CITATIONS", "20")),
        )
