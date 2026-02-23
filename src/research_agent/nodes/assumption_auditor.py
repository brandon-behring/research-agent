"""Assumption Auditor node — audits statistical method assumptions.

For each method identified by the planner, calls research-kb's assumption
auditing tool to get formal statements, violation consequences, and
verification approaches. Critical for causal inference research where
identification depends on assumptions.
"""

from __future__ import annotations

import logging
from typing import Any

from research_agent.config import AgentConfig
from research_agent.mcp_client import ResearchKBClient
from research_agent.state import AssumptionAudit, ResearchState

logger = logging.getLogger(__name__)


async def assumption_auditor(
    state: ResearchState,
    config: AgentConfig,
    mcp: ResearchKBClient,
) -> dict[str, Any]:
    """Audit assumptions for all methods identified by the planner.

    Args:
        state: Current state with sub_tasks containing methods_to_audit.
        config: Agent configuration.
        mcp: Connected MCP client.

    Returns:
        Dict with 'assumption_audits' and 'assumption_summary' updates.
    """
    logger.info("Starting assumption audits")

    audits: list[AssumptionAudit] = []
    audited_methods: set[str] = set()

    # Collect all methods from sub-tasks
    for task in state.sub_tasks:
        for method in task.methods_to_audit:
            method_lower = method.lower()
            if method_lower in audited_methods:
                continue
            audited_methods.add(method_lower)

            try:
                raw = await mcp.audit_assumptions(
                    method_name=method,
                    include_docstring=True,
                )
                audit = AssumptionAudit(
                    method_name=method,
                    raw_output=raw,
                )
                audits.append(audit)
                logger.info("Audited assumptions for: %s", method)

            except Exception as e:
                logger.warning("Assumption audit failed for '%s': %s", method, e)
                audits.append(AssumptionAudit(
                    method_name=method,
                    raw_output=f"Audit failed: {e}",
                ))

    if not audits:
        summary = "No statistical methods identified for assumption auditing."
    else:
        summary = f"Audited {len(audits)} methods: {', '.join(audited_methods)}."

    logger.info(summary)

    return {
        "assumption_audits": audits,
        "assumption_summary": summary,
        "current_node": "assumption_auditor",
    }
