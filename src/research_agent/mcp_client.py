"""Thin MCP client wrapping research-kb's tools.

Architecture:
    This module abstracts the MCP transport (stdio or HTTP) so agent nodes
    call clean Python methods without knowing how research-kb is connected.

    Thirteen tools exposed (of research-kb's 22):

    JSON tools (output_format='json'):
        1. search           -- 4-signal hybrid search (BM25 + vector + graph + PageRank)
        2. fast_search      -- lightweight vector-only fallback
        3. get_concept      -- retrieve concept details from knowledge graph
        4. graph_neighborhood -- explore related concepts within N hops
        5. citation_network -- find citing/cited-by chains for a source
        6. biblio_coupling  -- related papers via shared reference overlap
        7. audit_assumptions -- method assumption documentation
        8. explain_connection -- trace path between concepts with evidence

    Markdown tools (no output_format parameter):
        9.  get_source            -- full source metadata (title, authors, year, type)
        10. find_similar_concepts -- embedding-based concept similarity
        11. cross_domain_concepts -- cross-domain concept bridging
        12. list_domains          -- available KB domains
        13. stats                 -- corpus size and composition

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
            logger.exception("Unexpected error during MCP connection setup")
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
        if self._stack is None:
            raise RuntimeError("_connect_stdio() called outside __aenter__ context")

        if not self._config.research_kb_path:
            raise MCPConnectionError(
                "RESEARCH_KB_PATH must be set for stdio transport. "
                "Point it to the research-kb repository root."
            )

        python_cmd = (
            self._config.research_kb_python or f"{self._config.research_kb_path}/.venv/bin/python"
        )
        server_params = StdioServerParameters(
            command=python_cmd,
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
        if self._stack is None:
            raise RuntimeError("_connect_http() called outside __aenter__ context")

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
        use_citations: bool = True,
        citation_weight: float = 0.15,
        use_expand: bool = True,
    ) -> str:
        """Hybrid search: BM25 + vector + graph + PageRank.

        Args:
            query: Natural language search query.
            limit: Maximum results (1-50).
            domain: Optional domain filter ('causal_inference', 'time_series').
            context_type: Search weighting ('building', 'auditing', 'balanced').
            use_graph: Include knowledge graph signals.
            use_rerank: Apply cross-encoder reranking.
            use_citations: Include citation PageRank signal.
            citation_weight: Weight for citation signal (0.0-1.0).
            use_expand: Expand query via HyDE.

        Returns:
            JSON string with search results and scores.
        """
        args: dict[str, Any] = {
            "query": query,
            "limit": limit,
            "context_type": context_type,
            "use_graph": use_graph,
            "use_rerank": use_rerank,
            "use_citations": use_citations,
            "citation_weight": citation_weight,
            "use_expand": use_expand,
            "output_format": "json",
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
            JSON string with results (vector similarity only).
        """
        args: dict[str, Any] = {"query": query, "limit": limit, "output_format": "json"}
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
            JSON string with concept description and relationships.
        """
        return await self._call_tool(
            "research_kb_get_concept",
            {
                "concept_id": concept_id,
                "include_relationships": include_relationships,
                "output_format": "json",
            },
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
            JSON string with connected concepts and relationship summary.
        """
        return await self._call_tool(
            "research_kb_graph_neighborhood",
            {"concept_name": concept_name, "hops": hops, "limit": limit, "output_format": "json"},
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
            JSON string with citing/cited-by lists.
        """
        return await self._call_tool(
            "research_kb_citation_network",
            {"source_id": source_id, "limit": limit, "output_format": "json"},
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
            JSON string with bibliographically similar sources.
        """
        return await self._call_tool(
            "research_kb_biblio_coupling",
            {
                "source_id": source_id,
                "limit": limit,
                "min_coupling": min_coupling,
                "output_format": "json",
            },
        )

    # -- Assumption tools -------------------------------------------------------

    async def audit_assumptions(
        self,
        method_name: str,
        include_docstring: bool = True,
        domain: str | None = None,
        scope: str = "general",
    ) -> str:
        """Audit method assumptions (e.g., DML, IV, DiD, RDD).

        Args:
            method_name: Statistical method name or alias.
            include_docstring: Include Python docstring snippet.
            domain: Optional domain context ('causal_inference', 'time_series').
            scope: Audit scope ('general' or 'applied').

        Returns:
            JSON string with assumptions, violation consequences, verification approaches.
        """
        args: dict[str, Any] = {
            "method_name": method_name,
            "include_docstring": include_docstring,
            "output_format": "json",
        }
        if domain:
            args["domain"] = domain
        if scope != "general":
            args["scope"] = scope
        return await self._call_tool("research_kb_audit_assumptions", args)

    # -- Connection tools -------------------------------------------------------

    async def explain_connection(
        self,
        concept_a: str,
        concept_b: str,
        style: str = "research",
        max_evidence_per_step: int = 2,
        use_llm: bool = False,
    ) -> str:
        """Trace path between concepts with evidence (graph-only, no LLM synthesis).

        Args:
            concept_a: Source concept name.
            concept_b: Target concept name.
            style: Explanation style ('research', 'teaching', 'brief').
            max_evidence_per_step: Evidence chunks per path step.
            use_llm: Whether to use LLM for narrative (False = graph-only).

        Returns:
            JSON string with path steps and evidence chunks.
        """
        return await self._call_tool(
            "research_kb_explain_connection",
            {
                "concept_a": concept_a,
                "concept_b": concept_b,
                "style": style,
                "max_evidence_per_step": max_evidence_per_step,
                "use_llm": use_llm,
                "output_format": "json",
            },
        )

    # -- Markdown tools (no output_format parameter) ---------------------------

    async def get_source(
        self,
        source_id: str,
        include_chunks: bool = False,
        chunk_limit: int = 10,
    ) -> str:
        """Retrieve full source metadata (title, authors, year, type, DOI).

        Args:
            source_id: UUID of the source document.
            include_chunks: Whether to include content chunks.
            chunk_limit: Maximum chunks if include_chunks is True.

        Returns:
            Markdown string with source metadata.
        """
        args: dict[str, Any] = {"source_id": source_id}
        if include_chunks:
            args["include_chunks"] = True
            args["chunk_limit"] = chunk_limit
        return await self._call_tool("research_kb_get_source", args)

    async def find_similar_concepts(
        self,
        concept_id: str,
        limit: int = 10,
        threshold: float = 0.8,
    ) -> str:
        """Find embedding-similar concepts in the knowledge graph.

        Args:
            concept_id: UUID of the source concept.
            limit: Maximum similar concepts to return.
            threshold: Minimum cosine similarity threshold (0.0-1.0).

        Returns:
            Markdown string with similar concepts and similarity scores.
        """
        return await self._call_tool(
            "research_kb_find_similar_concepts",
            {"concept_id": concept_id, "limit": limit, "threshold": threshold},
        )

    async def cross_domain_concepts(
        self,
        source_domain: str,
        target_domain: str,
        concept_name: str | None = None,
        concept_id: str | None = None,
        similarity_threshold: float = 0.85,
        limit: int = 10,
    ) -> str:
        """Find concepts that bridge two knowledge domains.

        Args:
            source_domain: Source domain name (e.g., 'causal_inference').
            target_domain: Target domain name (e.g., 'time_series').
            concept_name: Optional concept name to anchor the search.
            concept_id: Optional concept ID to anchor the search.
            similarity_threshold: Minimum similarity for cross-domain matches.
            limit: Maximum matches to return.

        Returns:
            Markdown string with cross-domain concept mappings.
        """
        args: dict[str, Any] = {
            "source_domain": source_domain,
            "target_domain": target_domain,
            "similarity_threshold": similarity_threshold,
            "limit": limit,
        }
        if concept_id:
            args["concept_id"] = concept_id
        elif concept_name:
            args["concept_name"] = concept_name
        return await self._call_tool("research_kb_cross_domain_concepts", args)

    async def list_domains(self) -> str:
        """List all available domains in the knowledge base.

        Returns:
            Markdown string with domain names and descriptions.
        """
        return await self._call_tool("research_kb_list_domains", {})

    async def stats(self) -> str:
        """Get corpus statistics (source count, chunk count, concept count).

        Returns:
            Markdown string with corpus composition summary.
        """
        return await self._call_tool("research_kb_stats", {})
