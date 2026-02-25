"""Thin MCP client wrapping research-kb's tools.

Architecture:
    This module abstracts the MCP transport (stdio or HTTP) so agent nodes
    call clean Python methods without knowing how research-kb is connected.

    Seven tools exposed (of research-kb's 20+):
        1. search           -- 4-signal hybrid search (BM25 + vector + graph + PageRank)
        2. fast_search      -- lightweight vector-only fallback
        3. get_concept      -- retrieve concept details from knowledge graph
        4. graph_neighborhood -- explore related concepts within N hops
        5. citation_network -- find citing/cited-by chains for a source
        6. biblio_coupling  -- related papers via shared reference overlap
        7. audit_assumptions -- method assumption documentation

    Each method returns the raw markdown string from research-kb.
    Agent nodes parse what they need from the structured markdown.

Resilience:
    - Retries with exponential backoff on MCPToolError (3 attempts)
    - Specific exception types (MCPConnectionError, MCPToolError)
    - LangSmith tracing via @traceable (no-op without LANGCHAIN_API_KEY)
"""

from __future__ import annotations

import json
import logging
from contextlib import AsyncExitStack
from typing import Any

import httpx
from langsmith import traceable
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from research_agent.config import MCPConfig
from research_agent.exceptions import MCPConnectionError, MCPToolError

logger = logging.getLogger(__name__)


class ResearchKBClient:
    """Client for research-kb MCP server.

    Usage::

        async with ResearchKBClient(config) as client:
            results = await client.search("causal inference assumptions")

    Args:
        config: MCP connection configuration (stdio path or HTTP URL).
    """

    def __init__(self, config: MCPConfig) -> None:
        self._config = config
        self._session: ClientSession | None = None
        self._stack: AsyncExitStack | None = None

    async def __aenter__(self) -> ResearchKBClient:
        """Connect to the MCP server.

        Uses AsyncExitStack for proper LIFO cleanup ordering:
        session exits before transport, preventing anyio task group scope errors.
        """
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()
        try:
            if self._config.transport == "stdio":
                await self._connect_stdio()
            elif self._config.transport == "http":
                await self._connect_http()
            else:
                await self._stack.aclose()
                raise MCPConnectionError(f"Unsupported transport: {self._config.transport}")
        except MCPConnectionError:
            await self._stack.aclose()
            raise
        except Exception:
            await self._stack.aclose()
            raise
        return self

    async def __aexit__(self, *exc: Any) -> None:
        """Disconnect from the MCP server.

        AsyncExitStack ensures session closes before transport (LIFO order).

        The stdio transport uses an anyio task group whose cancel scope can
        raise RuntimeError during cleanup after the ClientSession's own task
        group has already exited. This is a known MCP SDK pattern issue --
        the work is complete by this point, so we log and suppress it.
        """
        if self._stack:
            try:
                await self._stack.__aexit__(*exc)
            except (RuntimeError, BaseExceptionGroup) as e:
                logger.debug("Transport cleanup error (non-fatal): %s", e)

    async def _connect_stdio(self) -> None:
        """Establish stdio connection to research-kb server.

        Raises:
            MCPConnectionError: If RESEARCH_KB_PATH is not set or connection fails.
        """
        assert self._stack is not None

        if not self._config.research_kb_path:
            raise MCPConnectionError(
                "RESEARCH_KB_PATH must be set for stdio transport. "
                "Point it to the research-kb repository root."
            )

        server_params = StdioServerParameters(
            command=f"{self._config.research_kb_path}/venv/bin/python",
            args=["-m", "research_kb_mcp.server"],
            cwd=self._config.research_kb_path,
        )

        try:
            read, write = await self._stack.enter_async_context(stdio_client(server_params))
            session = ClientSession(read, write)
            self._session = await self._stack.enter_async_context(session)
            await self._session.initialize()
            logger.info("Connected to research-kb MCP server via stdio")
        except Exception as e:
            raise MCPConnectionError(f"Failed to connect via stdio: {e}") from e

    async def _connect_http(self) -> None:
        """Establish HTTP connection to research-kb server.

        Uses the MCP streamable HTTP transport. The endpoint URL is constructed
        from ``http_url`` + ``mcp_path`` (configurable, default ``/mcp``).

        Raises:
            MCPConnectionError: If http_url is empty or connection fails.
        """
        assert self._stack is not None

        if not self._config.http_url:
            raise MCPConnectionError(
                "RESEARCH_KB_URL must be set for HTTP transport. "
                "Point it to the research-kb HTTP endpoint."
            )

        url = self._config.http_url.rstrip("/") + self._config.mcp_path
        logger.info("Connecting to research-kb MCP server at %s", url)

        try:
            read_stream, write_stream, _ = await self._stack.enter_async_context(
                streamable_http_client(url=url)
            )
            session = ClientSession(read_stream, write_stream)
            self._session = await self._stack.enter_async_context(session)
            await self._session.initialize()
            logger.info("Connected to research-kb MCP server via HTTP")
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError, OSError) as e:
            raise MCPConnectionError(f"Failed to connect via HTTP to {url}: {e}") from e
        except Exception as e:
            raise MCPConnectionError(f"Failed to connect via HTTP to {url}: {e}") from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(MCPToolError),
        reraise=True,
    )
    @traceable(name="mcp_call", run_type="tool")
    async def _call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Call an MCP tool and return the text content.

        Args:
            name: Tool name (e.g., 'research_kb_search').
            arguments: Tool arguments as a dict.

        Returns:
            The text content from the tool response.

        Raises:
            RuntimeError: If not connected.
            MCPToolError: If the tool call returns an error.
        """
        if self._session is None:
            raise RuntimeError("Not connected. Use 'async with ResearchKBClient(...)' pattern.")

        logger.debug("Calling MCP tool: %s(%s)", name, json.dumps(arguments, default=str))
        result = await self._session.call_tool(name, arguments)

        # MCP returns content as list of TextContent/ImageContent
        texts = [block.text for block in result.content if hasattr(block, "text")]
        output = "\n".join(texts)

        if result.isError:
            logger.error("MCP tool %s returned error: %s", name, output)
            raise MCPToolError(tool_name=name, detail=output)

        logger.debug("MCP tool %s returned %d chars", name, len(output))
        return output

    # -- Search tools -------------------------------------------------------

    async def search(
        self,
        query: str,
        limit: int = 10,
        domain: str | None = None,
        context_type: str = "balanced",
        use_graph: bool = True,
        use_rerank: bool = True,
    ) -> str:
        """Hybrid search: BM25 + vector + graph + PageRank.

        Args:
            query: Natural language search query.
            limit: Maximum results (1-50).
            domain: Optional domain filter ('causal_inference', 'time_series').
            context_type: Search weighting ('building', 'auditing', 'balanced').
            use_graph: Include knowledge graph signals.
            use_rerank: Apply cross-encoder reranking.

        Returns:
            Markdown-formatted search results with scores.
        """
        args: dict[str, Any] = {
            "query": query,
            "limit": limit,
            "context_type": context_type,
            "use_graph": use_graph,
            "use_rerank": use_rerank,
        }
        if domain:
            args["domain"] = domain
        return await self._call_tool("research_kb_search", args)

    async def fast_search(
        self,
        query: str,
        limit: int = 5,
        domain: str | None = None,
    ) -> str:
        """Lightweight vector-only search (~200ms).

        Args:
            query: Search query.
            limit: Maximum results (1-20).
            domain: Optional domain filter.

        Returns:
            Markdown-formatted results (vector similarity only).
        """
        args: dict[str, Any] = {"query": query, "limit": limit}
        if domain:
            args["domain"] = domain
        return await self._call_tool("research_kb_fast_search", args)

    # -- Concept tools -------------------------------------------------------

    async def get_concept(
        self,
        concept_id: str,
        include_relationships: bool = True,
    ) -> str:
        """Retrieve concept details from the knowledge graph.

        Args:
            concept_id: UUID of the concept.
            include_relationships: Include REQUIRES/USES/ADDRESSES edges.

        Returns:
            Markdown with concept description and relationships.
        """
        return await self._call_tool(
            "research_kb_get_concept",
            {"concept_id": concept_id, "include_relationships": include_relationships},
        )

    async def graph_neighborhood(
        self,
        concept_name: str,
        hops: int = 2,
        limit: int = 50,
    ) -> str:
        """Explore concepts connected within N hops.

        Args:
            concept_name: Name of concept (fuzzy matched).
            hops: Relationship hops to traverse (1-3).
            limit: Maximum connected concepts.

        Returns:
            Markdown with connected concepts and relationship summary.
        """
        return await self._call_tool(
            "research_kb_graph_neighborhood",
            {"concept_name": concept_name, "hops": hops, "limit": limit},
        )

    # -- Citation tools -------------------------------------------------------

    async def citation_network(
        self,
        source_id: str,
        limit: int = 20,
    ) -> str:
        """Find citation chains for a source.

        Args:
            source_id: UUID of the source paper.
            limit: Maximum sources per direction.

        Returns:
            Markdown with citing/cited-by lists.
        """
        return await self._call_tool(
            "research_kb_citation_network",
            {"source_id": source_id, "limit": limit},
        )

    async def biblio_coupling(
        self,
        source_id: str,
        limit: int = 10,
        min_coupling: float = 0.1,
    ) -> str:
        """Find related papers via shared reference overlap (Jaccard similarity).

        Args:
            source_id: UUID of the source to find neighbors for.
            limit: Maximum results.
            min_coupling: Minimum Jaccard threshold (0.0-1.0).

        Returns:
            Markdown with bibliographically similar sources.
        """
        return await self._call_tool(
            "research_kb_biblio_coupling",
            {"source_id": source_id, "limit": limit, "min_coupling": min_coupling},
        )

    # -- Assumption tools -------------------------------------------------------

    async def audit_assumptions(
        self,
        method_name: str,
        include_docstring: bool = True,
    ) -> str:
        """Audit method assumptions (e.g., DML, IV, DiD, RDD).

        Args:
            method_name: Statistical method name or alias.
            include_docstring: Include Python docstring snippet.

        Returns:
            Markdown with assumptions, violation consequences, verification approaches.
        """
        return await self._call_tool(
            "research_kb_audit_assumptions",
            {"method_name": method_name, "include_docstring": include_docstring},
        )
