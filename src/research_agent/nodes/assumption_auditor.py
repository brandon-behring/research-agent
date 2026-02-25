"""Assumption Auditor node -- audits statistical method assumptions.

For each method identified by the planner, calls research-kb's assumption
auditing tool to get formal statements, violation consequences, and
verification approaches. Critical for causal inference research where
identification depends on assumptions.
"""

from __future__ import annotations

import asyncio
import logging

from research_agent.config import AgentConfig
from research_agent.mcp_client import ResearchKBClient
from research_agent.state import AssumptionAudit, NodeUpdate, ResearchState

logger = logging.getLogger(__name__)

# Timeout for all assumption audit calls combined.
_AUDIT_TIMEOUT_SECONDS = 45


async def assumption_auditor(
    state: ResearchState,
    config: AgentConfig,
    mcp: ResearchKBClient,
) -> NodeUpdate:
    """Audit assumptions for all methods identified by the planner.

    Args:
        state: Current state with sub_tasks containing methods_to_audit.
        config: Agent configuration.
        mcp: Connected MCP client.

    Returns:
        NodeUpdate with ``assumption_audits`` and ``assumption_summary``.
    """
    logger.info("Starting assumption audits")

    audited_methods: set[str] = set()

    # Collect and deduplicate methods from sub-tasks
    unique_methods: list[str] = []
    for task in state.sub_tasks:
        for method in task.methods_to_audit:
            method_lower = method.lower()
            if method_lower not in audited_methods:
                audited_methods.add(method_lower)
                unique_methods.append(method)

    audits: list[AssumptionAudit] = []

    try:
        async with asyncio.timeout(_AUDIT_TIMEOUT_SECONDS):
            # Fan out all audit calls concurrently
            raw_results = await asyncio.gather(
                *[
                    mcp.audit_assumptions(method_name=m, include_docstring=True)
                    for m in unique_methods
                ],
                return_exceptions=True,
            )

            for method, raw in zip(unique_methods, raw_results, strict=True):
                if isinstance(raw, BaseException):
                    logger.warning("Assumption audit failed for '%s': %s", method, raw)
                    audits.append(
                        AssumptionAudit(
                            method_name=method,
                            raw_output=f"Audit failed: {raw}",
                        )
                    )
                else:
                    audits.append(
                        AssumptionAudit(
                            method_name=method,
                            raw_output=raw,
                        )
                    )
                    logger.info("Audited assumptions for: %s", method)

    except TimeoutError:
        logger.warning("Assumption auditing timed out with %d audits", len(audits))

    if not audits:
        summary = "No statistical methods identified for assumption auditing."
    else:
        summary = f"Audited {len(audits)} methods: {', '.join(audited_methods)}."

    logger.info(summary)

    return NodeUpdate(
        assumption_audits=audits,
        assumption_summary=summary,
        current_node="assumption_auditor",
    )
