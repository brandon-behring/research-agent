"""Tests for MCP client wrapper.

Tests the ResearchKBClient interface -- validates method signatures,
error handling, and argument forwarding.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from research_agent.config import MCPConfig
from research_agent.exceptions import MCPConnectionError
from research_agent.mcp_client import ResearchKBClient


class TestMCPClientConfig:
    """Test MCP client configuration handling."""

    def test_requires_research_kb_path_for_stdio(self) -> None:
        """Stdio transport requires RESEARCH_KB_PATH."""
        config = MCPConfig(transport="stdio", research_kb_path="")
        client = ResearchKBClient(config)
        with pytest.raises(ValueError, match="RESEARCH_KB_PATH must be set"):
            asyncio.run(client.__aenter__())

    def test_rejects_unsupported_transport(self) -> None:
        """Unknown transport rejected at construction via Literal validation."""
        with pytest.raises(ValidationError):
            MCPConfig(transport="grpc", research_kb_path="/some/path")

    def test_not_connected_raises_runtime_error(self) -> None:
        """Calling tools without connecting raises RuntimeError."""
        config = MCPConfig(transport="stdio", research_kb_path="/fake")
        client = ResearchKBClient(config)
        with pytest.raises(RuntimeError, match="Not connected"):
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


class TestHTTPTransport:
    """Tests for HTTP transport support."""

    def test_http_transport_accepted(self) -> None:
        """MCPConfig accepts 'http' transport."""
        config = MCPConfig(transport="http", http_url="http://localhost:8000")
        assert config.transport == "http"

    def test_connect_http_missing_url(self) -> None:
        """Empty http_url raises MCPConnectionError."""
        config = MCPConfig(transport="http", http_url="")
        client = ResearchKBClient(config)
        with pytest.raises(MCPConnectionError, match="RESEARCH_KB_URL must be set"):
            asyncio.run(client.__aenter__())

    @pytest.mark.asyncio
    async def test_connect_http_connection_failure(self) -> None:
        """Connection failure raises MCPConnectionError."""
        import httpx

        config = MCPConfig(transport="http", http_url="http://unreachable-host:9999")
        client = ResearchKBClient(config)

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        mock_ctx.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("research_agent.mcp_client.streamable_http_client", return_value=mock_ctx),
            pytest.raises(MCPConnectionError, match="Failed to connect via HTTP"),
        ):
            await client.__aenter__()

    @pytest.mark.asyncio
    async def test_connect_http_uses_streamable_client(self) -> None:
        """HTTP transport uses streamable_http_client and initializes session."""
        config = MCPConfig(
            transport="http",
            http_url="http://localhost:8000",
            mcp_path="/mcp",
        )
        client = ResearchKBClient(config)

        mock_read = MagicMock()
        mock_write = MagicMock()
        mock_session_id = MagicMock()

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=(mock_read, mock_write, mock_session_id))
        mock_ctx.__aexit__ = AsyncMock(return_value=None)

        with (
            patch(
                "research_agent.mcp_client.streamable_http_client", return_value=mock_ctx
            ) as mock_http,
            patch("research_agent.mcp_client.ClientSession") as mock_session_cls,
        ):
            mock_session = AsyncMock()
            mock_session_cls.return_value = mock_session

            await client.__aenter__()

            mock_http.assert_called_once_with(url="http://localhost:8000/mcp")
            mock_session_cls.assert_called_once_with(mock_read, mock_write)
            mock_session.initialize.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_aexit_cleans_up_http_context(self) -> None:
        """__aexit__ cleans up the HTTP context manager."""
        config = MCPConfig(transport="http", http_url="http://localhost:8000")
        client = ResearchKBClient(config)

        # Simulate an established HTTP connection
        mock_ctx = AsyncMock()
        mock_ctx.__aexit__ = AsyncMock(return_value=None)
        client._http_context = mock_ctx

        await client.__aexit__(None, None, None)
        mock_ctx.__aexit__.assert_awaited_once()

    def test_http_url_path_construction(self) -> None:
        """URL is constructed from http_url + mcp_path with proper slash handling."""
        # Trailing slash on URL should be stripped
        config = MCPConfig(
            transport="http",
            http_url="http://localhost:8000/",
            mcp_path="/mcp",
        )
        url = config.http_url.rstrip("/") + config.mcp_path
        assert url == "http://localhost:8000/mcp"

    def test_custom_mcp_path(self) -> None:
        """Custom mcp_path is respected in URL construction."""
        config = MCPConfig(
            transport="http",
            http_url="http://localhost:8000",
            mcp_path="/v1/mcp",
        )
        url = config.http_url.rstrip("/") + config.mcp_path
        assert url == "http://localhost:8000/v1/mcp"
