---
name: research-agent-archived
description: research-agent (LangGraph 7-node pipeline) ARCHIVED 2026-06-12 (RS2/R5). It was never used in real work; capabilities re-homed to research-kb MCP + research_toolkit skills + synthesis-kb. Push-based memory sync ended at archival.
metadata:
  node_type: memory
  type: project
  originSessionId: 1b78bb6c-1471-47e2-8fd6-4d21b96dec31
---

**Status (2026-05-21)**: Slated for archival at M6 of the synthesis-kb build (see [[synthesis-kb-planned]]).

## Decision

Archive (not delete, not repurpose, not keep-as-is). Reasons:
- 4,811 LOC + 451 tests is real intellectual artifact, useful even unmaintained — parallel-fan-out pattern and per-node resilience worth preserving as documented prior art
- Deletion is irreversible
- Repurposing as a thin orchestrator risks a half-finished system
- Keep-as-is means maintaining a system the user does not use

## Migration map for research-agent's nodes

| node | new home |
|------|----------|
| `query_planner` | research_toolkit's `/research-plan` (exists) |
| `literature_search` | research-kb's `research_kb_search` (exists) |
| `concept_explorer` | `synthesis_kb_concept_neighborhood` + `synthesis_kb_find_similar_concepts` |
| `citation_analyzer` | research-kb's `research_kb_citation_network` + `synthesis_kb_biblio_coupling` |
| `assumption_auditor` | research-kb's `research_kb_audit_assumptions` + new synthesis-side audit |
| `connection_explorer` | `synthesis_kb_explain_bridge` |
| `synthesis` | research_toolkit's `/research-synthesize` (to be added) |

## Why it was unused

LangGraph orchestration (parallel fan-out, conditional routing, streaming) was designed for unattended-batch research. The actual use case turned out to be **interactive, modular, audit-heavy research** — better served by skills than by sealed pipelines. The architecture matched the intended problem but not the actual problem.

## On archival

At M6: add one-line README pointing to synthesis-kb. No code changes. Tests remain runnable for reference.

## Archived (2026-06-12, slice RS2)

Executed per the research-side review decision R5 (hub
`lever_of_archimedes/docs/plans/active/2026-06-research-side-design-review/decisions.yaml`), which
confirmed the May design's own §7.2/Q6 recommendation. This is the **final synced memory commit**:
immediately after it, the README archive banner lands, dependabot PRs + issue #13 close, and
`gh repo archive brandon-behring/research-agent` freezes the repo read-only on GitHub.

- **Push-based memory sync ends here** — [[reference-memory-synced-via-git]] becomes partly stale:
  the `~/.claude/projects/.../memory` symlink keeps working locally, but commits can no longer push.
- One local credential string was redacted from [[synthesis-kb-planned]] before this commit
  (the arrangement's no-secrets rule; the repo is public).
- Local clone stays at `~/Claude/research-agent` as the §7.2 reference copy (root cleanup = RS7).
- Successor stack: research-kb MCP (knowledge queries) · research_toolkit skills (dossier authoring) ·
  synthesis-kb (claim/concept synthesis — where the migration map above effectively landed; its
  search/provenance MCP tools are slice RS6).
