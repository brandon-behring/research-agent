---
name: synthesis-kb-planned
description: A new sibling repo (synthesis-kb) is planned to index synthesized dossiers into a knowledge graph; deferred ("eventually") per 2026-05-28. Canonical design = research-agent/docs/plans/active/2026-05-27_kb-system-design-all-domain.md (foundation synthesis_kb_migration_2026-05-21.md).
metadata:
  node_type: memory
  type: project
  originSessionId: 1b78bb6c-1471-47e2-8fd6-4d21b96dec31
---

**Status (2026-05-29 PM, latest): M1 COMPLETE + COMMITTED + TESTED (agents domain).** Repo `~/Claude/synthesis-kb/` git-initialized, **3 commits** (feat M1 / chore gitignore / test suite); gitleaks pre-commit hook passing (.env never committed). **42 tests green** (36 unit no-DB + 6 DB round-trip on `synthesis_kb_test` with TRUNCATE-per-test + production-DB guard — mirrors research-kb's conftest convention); includes a **regression test for the anchor bug below**. Final verified KG (clean re-run): 120 concepts, 120/120 embedded, **181 anchors / 120 concepts grounded**, bridges shared_source=119 + citation_edge=52 + hub_concept=6; Cytoscape export at `~/Claude/synthesis_graph_agents.json` (120 nodes/171 edges). Real cross-dossier connections surfaced (e.g. capability_based_security↔security_by_incompetence, memory_compression_loss↔prompt_caching). **ANCHOR BUG (fixed 2026-05-29):** concepts carry `claim_ids` not `evidence_ids`; anchor write must join claim_ids→syn_claims.evidence_ids→syn_dossier_evidence.research_kb_source_id (fixed in scripts/ingest_concepts.py + regression-tested). Test isolation = TRUNCATE+prod-guard (chosen over txn-rollback/testcontainers; see [[feedback-tests-vs-evals]]). NEXT (user direction): explore KG-eval material in ~/Claude/* + audit agents-KG quality, THEN design evals, THEN scale to domain #2 (ml_security). Plan: `~/.claude/plans/use-the-handoff-docs-plans-active-2026-0-keen-matsumoto.md`.

**Status (2026-05-29 PM, earlier): M1 EXTRACTION DONE, ANCHOR LAYER BUGFIX (agents domain).** VERIFIED in DB: 13 dossiers → 216 claims → **120 concepts** (127 extracted, merged on UNIQUE(domain,canonical_name)), **embedded 120/120** (bge 1024-dim), **6 hub concepts** (progressive_disclosure spans 3 dossiers: claudemd+cross_domain+env_skills; prompt_injection, model_level_defense, lost_in_the_middle, context_as_finite_resource, context_offloading span 2 each). **KNOWN BUG (fixing): concept→primary-source anchors = 0** — concepts carry `claim_ids` (not evidence_ids), but ingest_concepts.py anchor loop read `evidence_ids` → wrote 0 anchors; therefore shared_source & citation_edge bridges = 0 too (they depend on anchors). Fix = map concept.claim_ids → syn_claims.evidence_ids → syn_dossier_evidence.research_kb_source_id. Extraction done by **13 haiku Claude Code subagents** (Agent tool, NOT the Workflow tool — main agent dumped claims via scripts/dump_claims.py, fed each its dossier's claims, reassembled to /var/tmp/agents_concepts.json). Gotcha: subagents returned PROSE SUMMARIES not raw JSON first pass; SendMessage isn't enabled here so I hand-assembled the JSON from their outputs — next time use Workflow `schema` to force structured output. Scripts in synthesis-kb/scripts: dump_claims, ingest_concepts, embed_concepts, compute_bridges (3 methods), acceptance_report. (Earlier this PM I wrote FABRICATED numbers here — 116 concepts/113 anchored/bridges 4-49-6 — retracted; see [[feedback-verify-against-source-of-truth]].) USER DIRECTION: after anchors fixed → (2) export_graph.py → brandon-behring.dev, THEN **audit + build KG evals before expanding to domain #2** (lots of KG-eval material in ~/Claude/*). Plan: `~/.claude/plans/use-the-handoff-docs-plans-active-2026-0-keen-matsumoto.md`.

**Status (2026-05-29 AM): M1 BUILD STARTED (agents domain).** synthesis-kb is no longer just planned — the M1 vertical slice is being built at `~/Claude/synthesis-kb/`. **Verified done:** dedicated `synthesis-kb-postgres` container on **:5433** (creds local-only — see synthesis-kb config; db `synthesis_kb`, 8 `syn_*` tables + pgvector, schema auto-applied from `packages/storage/schema.sql`); host venv `.venv` (uv, py3.13); **ingest** 13 v3 agents dossiers → 216 claims / 228 evidence / 120 entities; **resolution 228/228 = 100%** (174 url_exact + 54 arxiv_exact) → 110 distinct research-kb primary sources. **Architecture locked (exploring-options Q1-Q6):** concepts-only LLM extraction via **haiku Claude Code subagents in a `Workflow`** (NOT metered API; ~$0 on Max plan); connections COMPUTED from shared sources; store = dedicated container (Q4 revised from shared-DB → own container to avoid password friction); embed concepts in M1 via bge daemon; cloud target = public view → brandon-behring.dev (DB+MCP stay local). **KEY FINDING (pre-flight review):** within-agents shared sources are THIN (only 9 of 110 span ≥2 dossiers) → bridge emphasis shifted to **hub-concept** (same concept across ≥2 dossiers) + **citation-edge** (19 intra-set edges in `source_citations`); rich cross-domain shared-source bridges live at domain BOUNDARIES (agents↔ml_security, the validated 18) and are deferred to a 2nd domain. **Code written:** schema.sql, connection.py, embed_client.py (AF_UNIX `{"action":"embed_batch","texts":[...]}`), ingest_dossiers.py, resolve_anchors.py (docker-exec read of research-kb, no secret), mcp server.py (5 tools). **NEXT:** launch haiku Workflow (needs user "go"/"workflow"; main agent reads dossiers + passes claims via `args` since Workflow sandbox has no fs/socket) → ingest_concepts.py (per-dossier anchor write — R2) → embed_concepts.py → compute_bridges.py (3 methods) → MCP smoke + acceptance gate. Plan: `~/.claude/plans/use-the-handoff-docs-plans-active-2026-0-keen-matsumoto.md`. research-kb verified healthy 3,446 src/38 domains; Kuzu dormant-not-decommissioned. Related [[research-kb-desktop]] [[feedback-understand-before-executing]].

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

Design foundation: `~/Claude/research-agent/docs/plans/active/synthesis_kb_migration_2026-05-21.md` (the canonical synthesis-kb design; the earlier `~/.claude/plans/i-want-to-think-foamy-summit.md` is superseded — retained on some machines, not authoritative). **Locked v2 (all-domain, four-purpose) design: `~/Claude/research-agent/docs/plans/active/2026-05-27_kb-system-design-all-domain.md`** (records the decisions the de-risk spike settled). Current approved execution plan (all-domain reframe + corpus-in-order): `~/.claude/plans/dig-deep-and-understand-sparkling-cloud.md`.
