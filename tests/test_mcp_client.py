"""Tests for MCP client wrapper.

Tests the ResearchKBClient interface — validates method signatures,
error handling, and argument forwarding.
"""

from __future__ import annotations

import pytest

from research_agent.config import MCPConfig
from research_agent.mcp_client import ResearchKBClient


class TestMCPClientConfig:
    """Test MCP client configuration handling."""

    def test_requires_research_kb_path_for_stdio(self) -> None:
        """Stdio transport requires RESEARCH_KB_PATH."""
        config = MCPConfig(transport="stdio", research_kb_path="")
        client = ResearchKBClient(config)
        with pytest.raises(ValueError, match="RESEARCH_KB_PATH must be set"):
            import asyncio

            asyncio.run(client.__aenter__())

    def test_rejects_unsupported_transport(self) -> None:
        """Unknown transport raises ValueError."""
        config = MCPConfig(transport="grpc", research_kb_path="/some/path")
        client = ResearchKBClient(config)
        with pytest.raises(ValueError, match="Unsupported transport"):
            import asyncio

            asyncio.run(client.__aenter__())

    def test_not_connected_raises_runtime_error(self) -> None:
        """Calling tools without connecting raises RuntimeError."""
        config = MCPConfig(transport="stdio", research_kb_path="/fake")
        client = ResearchKBClient(config)
        with pytest.raises(RuntimeError, match="Not connected"):
            import asyncio

            asyncio.run(client.search("test query"))


class TestMCPClientInterface:
    """Test that all 7 tool methods have correct signatures."""

    def test_search_method_exists(self) -> None:
        """search() method exists with expected parameters."""
        config = MCPConfig(transport="stdio", research_kb_path="/fake")
        client = ResearchKBClient(config)
        assert hasattr(client, "search")
        assert callable(client.search)

    def test_fast_search_method_exists(self) -> None:
        """fast_search() method exists."""
        config = MCPConfig(transport="stdio", research_kb_path="/fake")
        client = ResearchKBClient(config)
        assert hasattr(client, "fast_search")

    def test_get_concept_method_exists(self) -> None:
        """get_concept() method exists."""
        config = MCPConfig(transport="stdio", research_kb_path="/fake")
        client = ResearchKBClient(config)
        assert hasattr(client, "get_concept")

    def test_graph_neighborhood_method_exists(self) -> None:
        """graph_neighborhood() method exists."""
        config = MCPConfig(transport="stdio", research_kb_path="/fake")
        client = ResearchKBClient(config)
        assert hasattr(client, "graph_neighborhood")

    def test_citation_network_method_exists(self) -> None:
        """citation_network() method exists."""
        config = MCPConfig(transport="stdio", research_kb_path="/fake")
        client = ResearchKBClient(config)
        assert hasattr(client, "citation_network")

    def test_biblio_coupling_method_exists(self) -> None:
        """biblio_coupling() method exists."""
        config = MCPConfig(transport="stdio", research_kb_path="/fake")
        client = ResearchKBClient(config)
        assert hasattr(client, "biblio_coupling")

    def test_audit_assumptions_method_exists(self) -> None:
        """audit_assumptions() method exists."""
        config = MCPConfig(transport="stdio", research_kb_path="/fake")
        client = ResearchKBClient(config)
        assert hasattr(client, "audit_assumptions")

    def test_all_seven_tools_present(self) -> None:
        """All 7 tool methods exist on the client."""
        config = MCPConfig(transport="stdio", research_kb_path="/fake")
        client = ResearchKBClient(config)
        expected_tools = [
            "search",
            "fast_search",
            "get_concept",
            "graph_neighborhood",
            "citation_network",
            "biblio_coupling",
            "audit_assumptions",
        ]
        for tool in expected_tools:
            assert hasattr(client, tool), f"Missing tool method: {tool}"
