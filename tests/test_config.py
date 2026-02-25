"""Tests for agent configuration.

Tests env var loading (monkeypatched), bounds validation,
transport Literal rejection, and frozen immutability.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from research_agent.config import AgentConfig, MCPConfig, ModelConfig


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_defaults(self) -> None:
        """Default models are set correctly."""
        config = ModelConfig()
        assert "haiku" in config.planning
        assert "sonnet" in config.synthesis
        assert "sonnet" in config.synthesis

    def test_explicit_overrides(self) -> None:
        """Constructor overrides work for tests."""
        config = ModelConfig(
            planning="test-model",
            synthesis="test-model-2",
        )
        assert config.planning == "test-model"
        assert config.synthesis == "test-model-2"

    def test_frozen(self) -> None:
        """ModelConfig is immutable."""
        config = ModelConfig()
        with pytest.raises(ValidationError):
            config.planning = "changed"


class TestMCPConfig:
    """Tests for MCPConfig."""

    def test_defaults(self) -> None:
        """Default transport is stdio."""
        config = MCPConfig()
        assert config.transport == "stdio"
        assert config.http_url == "http://research-kb:8000"

    def test_accepts_stdio(self) -> None:
        """stdio is a valid transport."""
        config = MCPConfig(transport="stdio")
        assert config.transport == "stdio"

    def test_accepts_http(self) -> None:
        """http is a valid transport."""
        config = MCPConfig(transport="http")
        assert config.transport == "http"

    def test_rejects_invalid_transport(self) -> None:
        """Invalid transport rejected by Literal validation."""
        with pytest.raises(ValidationError):
            MCPConfig(transport="grpc")

    def test_frozen(self) -> None:
        """MCPConfig is immutable."""
        config = MCPConfig()
        with pytest.raises(ValidationError):
            config.transport = "http"

    def test_mcp_path_default(self) -> None:
        """Default mcp_path is /mcp."""
        config = MCPConfig()
        assert config.mcp_path == "/mcp"

    def test_mcp_path_override(self) -> None:
        """Custom mcp_path is accepted."""
        config = MCPConfig(mcp_path="/v1/mcp")
        assert config.mcp_path == "/v1/mcp"

    def test_mcp_path_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Loads mcp_path from MCP_PATH env var."""
        monkeypatch.setenv("MCP_PATH", "/custom/endpoint")
        config = MCPConfig()
        assert config.mcp_path == "/custom/endpoint"

    def test_env_var_loading(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Loads transport from MCP_TRANSPORT env var."""
        monkeypatch.setenv("MCP_TRANSPORT", "http")
        monkeypatch.setenv("RESEARCH_KB_URL", "http://localhost:9999")
        config = MCPConfig()
        assert config.transport == "http"
        assert config.http_url == "http://localhost:9999"


class TestAgentConfig:
    """Tests for AgentConfig."""

    def test_defaults(self) -> None:
        """Default values are sensible."""
        config = AgentConfig()
        assert config.max_search_results == 10
        assert config.max_concepts == 15
        assert config.max_citations == 20

    def test_explicit_overrides(self) -> None:
        """Constructor overrides for test configs."""
        config = AgentConfig(
            max_search_results=5,
            max_concepts=3,
            max_citations=5,
        )
        assert config.max_search_results == 5
        assert config.max_concepts == 3
        assert config.max_citations == 5

    def test_bounds_min(self) -> None:
        """Values below minimum are rejected."""
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            AgentConfig(max_search_results=0)

    def test_bounds_max(self) -> None:
        """Values above maximum are rejected."""
        with pytest.raises(ValidationError, match="less than or equal to 50"):
            AgentConfig(max_search_results=100)

    def test_frozen(self) -> None:
        """AgentConfig is immutable."""
        config = AgentConfig()
        with pytest.raises(ValidationError):
            config.max_search_results = 5

    def test_env_var_loading(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Loads max_search_results from MAX_SEARCH_RESULTS env var."""
        monkeypatch.setenv("MAX_SEARCH_RESULTS", "7")
        config = AgentConfig()
        assert config.max_search_results == 7

    def test_nested_model_config(self) -> None:
        """Nested ModelConfig is accessible."""
        config = AgentConfig(
            models=ModelConfig(planning="test-planner"),
        )
        assert config.models.planning == "test-planner"

    def test_nested_mcp_config(self) -> None:
        """Nested MCPConfig is accessible."""
        config = AgentConfig(
            mcp=MCPConfig(transport="http"),
        )
        assert config.mcp.transport == "http"
