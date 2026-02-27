"""Tests for MCP client wrapper.

Tests the ResearchKBClient interface -- validates method signatures,
error handling, argument forwarding, and _call_tool retry logic.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from research_agent.config import MCPConfig
from research_agent.exceptions import MCPConnectionError, MCPToolError
from research_agent.mcp_client import ResearchKBClient


class TestMCPClientConfig:
    """Test MCP client configuration handling."""

    def test_requires_research_kb_path_for_stdio(self) -> None:
        """Stdio transport requires RESEARCH_KB_PATH."""
        config = MCPConfig(transport="stdio", research_kb_path="")
        client = ResearchKBClient(config)
        with pytest.raises(MCPConnectionError, match="RESEARCH_KB_PATH must be set"):
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


class TestCustomPythonPath:
    """Tests for RESEARCH_KB_PYTHON configuration."""

    def test_custom_python_path_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """RESEARCH_KB_PYTHON env var overrides default venv/bin/python."""
        monkeypatch.setenv("RESEARCH_KB_PYTHON", "/usr/bin/python3")
        config = MCPConfig(transport="stdio", research_kb_path="/fake/path")
        assert config.research_kb_python == "/usr/bin/python3"

    def test_custom_python_path_constructor(self) -> None:
        """Constructor override for research_kb_python."""
        config = MCPConfig(
            transport="stdio",
            research_kb_path="/fake/path",
            research_kb_python="/opt/conda/bin/python",
        )
        assert config.research_kb_python == "/opt/conda/bin/python"

    def test_default_python_path_empty(self) -> None:
        """Default research_kb_python is empty string."""
        config = MCPConfig(transport="stdio", research_kb_path="/fake/path")
        assert config.research_kb_python == ""


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
            # enter_async_context calls __aenter__, which must return the session
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_cls.return_value = mock_session

            await client.__aenter__()

            mock_http.assert_called_once_with(url="http://localhost:8000/mcp")
            mock_session_cls.assert_called_once_with(mock_read, mock_write)
            mock_session.initialize.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_aexit_cleans_up_http_context(self) -> None:
        """__aexit__ cleans up all context managers via AsyncExitStack."""
        config = MCPConfig(transport="http", http_url="http://localhost:8000")
        client = ResearchKBClient(config)

        mock_read = MagicMock()
        mock_write = MagicMock()
        mock_session_id = MagicMock()

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=(mock_read, mock_write, mock_session_id))
        mock_ctx.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("research_agent.mcp_client.streamable_http_client", return_value=mock_ctx),
            patch("research_agent.mcp_client.ClientSession") as mock_session_cls,
        ):
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_cls.return_value = mock_session

            async with client:
                pass

            # Both session and transport should be cleaned up
            mock_session.__aexit__.assert_awaited_once()
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


class TestToolArgumentForwarding:
    """Verify each tool method forwards correct arguments to _call_tool."""

    @pytest.fixture
    def connected_client(self) -> tuple[ResearchKBClient, AsyncMock]:
        """Client with injected mock session."""
        config = MCPConfig(transport="stdio", research_kb_path="/fake")
        client = ResearchKBClient(config)
        session = AsyncMock()
        text_block = SimpleNamespace(text='{"results": []}')
        session.call_tool.return_value = SimpleNamespace(content=[text_block], isError=False)
        client._session = session
        return client, session

    @pytest.mark.asyncio
    async def test_search_forwards_all_args(
        self, connected_client: tuple[ResearchKBClient, AsyncMock]
    ) -> None:
        """search() forwards all parameters including optional domain."""
        client, session = connected_client
        await client.search(
            query="DML",
            limit=5,
            domain="causal_inference",
            context_type="auditing",
            use_graph=False,
            use_rerank=False,
        )
        session.call_tool.assert_awaited_once_with(
            "research_kb_search",
            {
                "query": "DML",
                "limit": 5,
                "domain": "causal_inference",
                "context_type": "auditing",
                "use_graph": False,
                "use_rerank": False,
                "output_format": "json",
            },
        )

    @pytest.mark.asyncio
    async def test_search_omits_domain_when_none(
        self, connected_client: tuple[ResearchKBClient, AsyncMock]
    ) -> None:
        """search() excludes domain from args when None."""
        client, session = connected_client
        await client.search(query="DML", limit=10)
        args = session.call_tool.call_args[0][1]
        assert "domain" not in args
        assert args["query"] == "DML"
        assert args["output_format"] == "json"

    @pytest.mark.asyncio
    async def test_fast_search_forwards_args(
        self, connected_client: tuple[ResearchKBClient, AsyncMock]
    ) -> None:
        """fast_search() forwards query, limit, and optional domain."""
        client, session = connected_client
        await client.fast_search(query="IV", limit=3, domain="causal_inference")
        session.call_tool.assert_awaited_once_with(
            "research_kb_fast_search",
            {"query": "IV", "limit": 3, "domain": "causal_inference", "output_format": "json"},
        )

    @pytest.mark.asyncio
    async def test_fast_search_omits_domain_when_none(
        self, connected_client: tuple[ResearchKBClient, AsyncMock]
    ) -> None:
        """fast_search() excludes domain from args when None."""
        client, session = connected_client
        await client.fast_search(query="DML", limit=5)
        args = session.call_tool.call_args[0][1]
        assert "domain" not in args

    @pytest.mark.asyncio
    async def test_get_concept_forwards_args(
        self, connected_client: tuple[ResearchKBClient, AsyncMock]
    ) -> None:
        """get_concept() forwards concept_id and include_relationships."""
        client, session = connected_client
        await client.get_concept("concept-dml-001", include_relationships=False)
        session.call_tool.assert_awaited_once_with(
            "research_kb_get_concept",
            {
                "concept_id": "concept-dml-001",
                "include_relationships": False,
                "output_format": "json",
            },
        )

    @pytest.mark.asyncio
    async def test_graph_neighborhood_forwards_args(
        self, connected_client: tuple[ResearchKBClient, AsyncMock]
    ) -> None:
        """graph_neighborhood() forwards concept_name, hops, limit."""
        client, session = connected_client
        await client.graph_neighborhood(concept_name="DML", hops=3, limit=20)
        session.call_tool.assert_awaited_once_with(
            "research_kb_graph_neighborhood",
            {"concept_name": "DML", "hops": 3, "limit": 20, "output_format": "json"},
        )

    @pytest.mark.asyncio
    async def test_citation_network_forwards_args(
        self, connected_client: tuple[ResearchKBClient, AsyncMock]
    ) -> None:
        """citation_network() forwards source_id and limit."""
        client, session = connected_client
        await client.citation_network(source_id="src-001", limit=15)
        session.call_tool.assert_awaited_once_with(
            "research_kb_citation_network",
            {"source_id": "src-001", "limit": 15, "output_format": "json"},
        )

    @pytest.mark.asyncio
    async def test_biblio_coupling_forwards_args(
        self, connected_client: tuple[ResearchKBClient, AsyncMock]
    ) -> None:
        """biblio_coupling() forwards source_id, limit, and min_coupling."""
        client, session = connected_client
        await client.biblio_coupling(source_id="src-001", limit=5, min_coupling=0.2)
        session.call_tool.assert_awaited_once_with(
            "research_kb_biblio_coupling",
            {
                "source_id": "src-001",
                "limit": 5,
                "min_coupling": 0.2,
                "output_format": "json",
            },
        )

    @pytest.mark.asyncio
    async def test_audit_assumptions_forwards_args(
        self, connected_client: tuple[ResearchKBClient, AsyncMock]
    ) -> None:
        """audit_assumptions() forwards method_name and include_docstring."""
        client, session = connected_client
        await client.audit_assumptions(method_name="DML", include_docstring=False)
        session.call_tool.assert_awaited_once_with(
            "research_kb_audit_assumptions",
            {"method_name": "DML", "include_docstring": False, "output_format": "json"},
        )


class TestCallTool:
    """Tests for _call_tool method -- retry logic and error handling."""

    @pytest.mark.asyncio
    async def test_successful_call_returns_text(self) -> None:
        """Successful MCP tool call returns concatenated text content."""
        config = MCPConfig(transport="stdio", research_kb_path="/fake")
        client = ResearchKBClient(config)

        # Inject a mock session directly
        mock_session = AsyncMock()
        text_block = SimpleNamespace(text="Result text")
        mock_session.call_tool.return_value = SimpleNamespace(
            content=[text_block],
            isError=False,
        )
        client._session = mock_session

        result = await client._call_tool("test_tool", {"arg": "value"})
        assert result == "Result text"
        mock_session.call_tool.assert_awaited_once_with("test_tool", {"arg": "value"})

    @pytest.mark.asyncio
    async def test_error_response_raises_mcp_tool_error(self) -> None:
        """isError=True raises MCPToolError."""
        config = MCPConfig(transport="stdio", research_kb_path="/fake")
        client = ResearchKBClient(config)

        mock_session = AsyncMock()
        text_block = SimpleNamespace(text="Tool error detail")
        mock_session.call_tool.return_value = SimpleNamespace(
            content=[text_block],
            isError=True,
        )
        client._session = mock_session

        with pytest.raises(MCPToolError, match="test_tool"):
            await client._call_tool("test_tool", {})

    @pytest.mark.asyncio
    async def test_retries_on_transient_mcp_tool_error(self) -> None:
        """Retries up to 3 times on MCPToolError, then re-raises."""
        config = MCPConfig(transport="stdio", research_kb_path="/fake")
        client = ResearchKBClient(config)

        mock_session = AsyncMock()
        text_block = SimpleNamespace(text="Transient failure")
        mock_session.call_tool.return_value = SimpleNamespace(
            content=[text_block],
            isError=True,
        )
        client._session = mock_session

        with pytest.raises(MCPToolError):
            await client._call_tool("flaky_tool", {})

        # Should have retried 3 times total (tenacity stop_after_attempt(3))
        assert mock_session.call_tool.await_count == 3
