---
name: synthesis-kb-planned
description: A new sibling repo to research-kb is planned (synthesis-kb) that indexes synthesized dossiers; research-agent is to be archived. Authoritative design doc lives at ~/.claude/plans/i-want-to-think-foamy-summit.md.
metadata:
  node_type: memory
  type: project
  originSessionId: 1b78bb6c-1471-47e2-8fd6-4d21b96dec31
---

**Status (2026-05-27)**: Reaffirmed + reframed. synthesis-kb is still wanted, but the approach is now **design-&-de-risk-first** (not "M0 begins next"); scope is **unified all-domain** (agents + causal inference + ML security + computational math) serving **four goals** (books · mastery/recall · cross-domain discovery · public map); near-term win is **"corpus in order"** (all ~36 strict-live dossiers pass `cross_stage` validation as of 2026-05-27 — the work is consolidate caches + export stragglers + catalogue, not fixing broken dossiers). Sequencing: **primaries-first**. research-kb's query layer is on the **desktop** — see [[research-kb-desktop]]. Approved execution plan: `~/.claude/plans/dig-deep-and-understand-sparkling-cloud.md`.

**Status (2026-05-21, historical)**: Architecture approved, not yet built. M0 begins next.

## Target four-repo state

| Repo | Layer | Role |
|------|-------|------|
| `research-kb` | Primary literature | Untouched — source of truth for what literature says |
| `synthesis-kb` (new) | Personal synthesis | Indexes dossiers (claims, evidence, bridges, extracted concepts); MCP + Streamlit dashboard; mirrors research-kb's stack (Postgres + pgvector + Kuzu + FastMCP) |
| `research_toolkit` | Dossier production | Builds dossiers; produces TWO outputs per project (cache → research-kb, dossier → synthesis-kb) |
| `research-agent` | Legacy | Archive — README pointer to synthesis-kb, no maintenance (see [[research-agent-archived]]) |

## Build sequence

M0 (cache export side) → M1 (synthesis-kb scaffold + ingest one dossier) → M2 (search + biblio coupling) → M3 (concept graph + bridges) → M4 (Streamlit dashboard) → M5 (consistency audit) → M6 (retire research-agent).

M1 is the discipline check: abandon cheaply if ingesting one dossier and querying it does not feel useful.

## Key design decisions

- **Epistemic separation enforced at storage layer** — see [[feedback-epistemic-separation]]
- **Files canonical, DB derived** — `claim_graph.jsonl`, `evidence_ledger.yml`, `bib_ledger.yml` are inputs; concepts, bridges, embeddings are DB-only derived data
- **Re-ingestion: upsert by stable IDs** + soft delete via `status='stale'`. Claim history lives in git on canonical files.
- **Three-identifier evidence model** — display (bibkey), truth (cache_id + sha256 + byte offset), retrieval (research-kb UUIDs, re-resolvable)
- **Wrapper command** `/dossier-publish <project>` chains the four-step ingestion pipeline

## Future-revisit items (§13 of plan)

Bibkey resolution strategy, dossier-audit findings flow, cross-pipeline dedup, concept-name disambiguation across domains, multi-machine sync, book/publication integration, MCP HTTP transport, prompt versioning, Tier-2/3 escalation paths. Each has a trigger criterion specified.

## brandon-behring.dev integration

Long-term goal: public synthesis-map at `brandon-behring.dev/knowledge` consuming `synthesis_kb_export_graph` output via build-time sync. Tracking issue filed on `brandon-behring/brandon-behring.dev`. See [[reference-brandon-behring-dev]].

## Authoritative reference

Design foundation: `~/Claude/research-agent/docs/plans/active/synthesis_kb_migration_2026-05-21.md` (the canonical synthesis-kb design; the earlier `~/.claude/plans/i-want-to-think-foamy-summit.md` no longer exists). **Locked v2 (all-domain, four-purpose) design: `~/Claude/research-agent/docs/plans/active/2026-05-27_kb-system-design-all-domain.md`** (records the decisions the de-risk spike settled). Current approved execution plan (all-domain reframe + corpus-in-order): `~/.claude/plans/dig-deep-and-understand-sparkling-cloud.md`.
