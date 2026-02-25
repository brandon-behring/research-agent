"""Configuration for model selection and MCP endpoints.

Uses pydantic-settings BaseSettings for automatic environment variable loading.
Explicit constructor overrides still work for tests: ``AgentConfig(max_search_results=5)``.

Design decisions:
    - Haiku for planning (fast, cheap -- routing doesn't need Opus)
    - Sonnet for synthesis (strong reasoning for final report)
    - MCP endpoint configurable via env vars for Docker flexibility
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class ModelConfig(BaseSettings):
    """LLM model selection per node type.

    Rationale: Cost/latency optimization. Planning nodes need speed,
    synthesis needs reasoning depth.
    """

    planning: str = Field(
        default="claude-haiku-4-5-20251001",
        alias="PLANNING_MODEL",
        description="Model for query decomposition (fast, cheap)",
    )
    synthesis: str = Field(
        default="claude-sonnet-4-6",
        alias="SYNTHESIS_MODEL",
        description="Model for report synthesis (strong reasoning)",
    )

    model_config = {"frozen": True, "populate_by_name": True}


class MCPConfig(BaseSettings):
    """Research-KB MCP server connection configuration.

    Supports two transports:
        - stdio: spawn research-kb as subprocess (local dev)
        - http: connect to running research-kb service (Docker)
    """

    transport: Literal["stdio", "http"] = Field(
        default="stdio",
        alias="MCP_TRANSPORT",
        description="MCP transport protocol",
    )
    research_kb_path: str = Field(
        default="",
        alias="RESEARCH_KB_PATH",
        description="Path to research-kb repo root (stdio mode)",
    )
    http_url: str = Field(
        default="http://research-kb:8000",
        alias="RESEARCH_KB_URL",
        description="HTTP endpoint for research-kb (Docker mode)",
    )
    mcp_path: str = Field(
        default="/mcp",
        alias="MCP_PATH",
        description="MCP endpoint path appended to http_url",
    )
    research_kb_python: str = Field(
        default="",
        alias="RESEARCH_KB_PYTHON",
        description="Python executable for stdio transport (default: {kb_path}/venv/bin/python)",
    )

    model_config = {"frozen": True, "populate_by_name": True}


class AgentConfig(BaseSettings):
    """Top-level agent configuration.

    All fields load from environment variables automatically.
    Explicit overrides in constructor take precedence (for tests).
    """

    models: ModelConfig = Field(default_factory=ModelConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    max_search_results: int = Field(
        default=10,
        ge=1,
        le=50,
        alias="MAX_SEARCH_RESULTS",
        description="Maximum results per search query",
    )
    max_concepts: int = Field(
        default=15,
        ge=1,
        le=50,
        alias="MAX_CONCEPTS",
        description="Maximum concepts to explore",
    )
    max_citations: int = Field(
        default=20,
        ge=1,
        le=50,
        alias="MAX_CITATIONS",
        description="Maximum citation chains per source",
    )
    synthesis_timeout: int = Field(
        default=120,
        ge=30,
        le=300,
        alias="SYNTHESIS_TIMEOUT",
        description="Timeout in seconds for synthesis LLM call",
    )
    cache_enabled: bool = Field(
        default=True,
        alias="CACHE_ENABLED",
        description="Enable SQLite report cache",
    )
    cache_db_path: str = Field(
        default="~/.cache/research-agent/cache.db",
        alias="CACHE_DB_PATH",
        description="Path to SQLite cache database",
    )
    cache_ttl_hours: float = Field(
        default=24.0,
        gt=0.0,
        le=720.0,
        alias="CACHE_TTL_HOURS",
        description="Hours before cached reports expire",
    )

    model_config = {"frozen": True, "populate_by_name": True}
