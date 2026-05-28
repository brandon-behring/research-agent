# Research KB System — Locked Design (v2, all-domain) · 2026-05-27

**Status: LOCKED.** The authoritative design for the unified personal research knowledge system.
**Extends** the foundation `synthesis_kb_migration_2026-05-21.md` — read that for the synthesis-kb schema
sketch (§3), MCP tool surface (§4), three-tier evidence identity model (§3.1), ingestion mechanics (§6),
and milestone detail (§8); all of it stands unless this doc says otherwise. This doc records what the
2026-05-27 session changed: **all-domain scope, four-purpose requirements, and the decisions the de-risk
spike settled.** Execution plan: `~/.claude/plans/dig-deep-and-understand-sparkling-cloud.md`. Corpus
catalogue: `~/Claude/research_INDEX.md`.

## 1. What the system is

Three layers, strict epistemic separation, unified across **all domains** (agents · causal/econ ·
ML-security · math · meta), with **domain as a lens**:

| Layer | Repo | Content | Delivers |
|---|---|---|---|
| **Primary** | research-kb (on **desktop**) | 842 cached primary docs → pgvector RAG + FTS; **citation / shared-source network** | what the literature says + cross-domain *source* bridges |
| **Synthesis** | synthesis-kb (new) | the 36 strict-live dossiers' claims/concepts; cross-domain bridges; evidence linked back to primaries | what *I've* synthesized + recall + *semantic* discovery |
| **View** | export → brandon-behring.dev (Cytoscape) | graph export + `--public` filter | the explorable / public map |

## 2. Goal → requirement (all four served simultaneously)

| Goal | Locked requirement |
|---|---|
| Books | atoms stay modular + a `content_map` links atom claim_ids → book chapter/section units; verbatim citation grounding |
| Mastery/recall | vector + FTS search over claims **and** primaries; `claim_provenance` to the evidence chain |
| Cross-domain discovery | bridges first-class, **two tiers** (§3); `domain` lens highlights cross-domain edges |
| Public map | `synthesis_kb_export_graph` + per-dossier/claim `visibility`; reuse the existing brandon-behring.dev Cytoscape stack |

## 3. Decisions the de-risk spike SETTLED (2026-05-27)

The throwaway all-domain compose over all 36 strict-live claim graphs (3,957 records, 996 claims) settled four previously-open questions:

1. **All-domain entity resolution = the existing toolkit scheme. No new ID namespacing needed.** The compose hit **0 cross-project claim/evidence ID collisions** across all 5 domains, and resolved shared nodes by deterministic id-recurrence (`ent_<bibkey>`, `src_<url-slug>`, `graph_cache_<sha>`). Lock: resolution is id / primary_url / sha256 recurrence, exactly as `_merge_projects.py` already does. The disjoint-prefix discipline holds corpus-wide. *(Kills the worry that unifying domains needs a new ID strategy.)*
2. **Cross-domain bridges ship in two tiers; tier-1 first.** Tier-1 = **shared primary source** (exact, cheap, from research-kb's citation net) — the spike found **18 real ones** (agents↔ML-security on the prompt-injection/agent-security literature; causal↔ML-sec on eval/benchmark sources). Tier-2 = **shared/similar concept** (semantic, synthesis-kb). Build tier-1 first — it is the validated MVP of the discovery value.
3. **Concept disambiguation (old Q13) is LOW-urgency and gets a conservative rule.** Because tier-1 works without concept resolution. Rule: concepts are **domain-tagged**; identical `canonical_name` across domains stays **distinct by default**, joined by a soft `same_term` edge surfaced for manual confirmation — **never auto-merged** (avoids the "convergence" causal-vs-math homonym error).
4. **Cheapest-mechanism-first is correct.** Source-level bridges need no embeddings or LLM, so the **primary citation network alone delivers core discovery value**. Embeddings/concepts are additive, not prerequisite — which is why primaries-first is the right sequence.

## 4. Schema deltas vs the 2026-05-21 sketch

The 2026-05-21 schema stands; add:
- **`domain`** tag on `dossiers`, `claims`, `dossier_concepts` (the lens; value set = research_INDEX.md domains).
- **`bridges.bridge_type`** ∈ `{shared_source, shared_concept, similar_concept, analogous_method, contradiction}`. `shared_source` is *exact* (computed in research-kb's citation net); the rest are semantic (synthesis-kb).
- **`content_map`** linking atom `claim_id` → book chapter/section units (serves Books; mirrors `claude-books/docs/research-program/content-map.md`).
- **`visibility`** (`public|private`, default private) on dossiers/claims for the public-map export.
- Three-tier evidence identity model **unchanged** — validated: 31/36 dossiers carry sha256 excerpt anchors (the 5 v2 get backfilled in normalization).

## 5. research-kb (primary layer) — build spec · DESKTOP

- Ingest `~/Claude/research_cache/` (842 docs: raw blob + extracted text + sha256) → `sources` + `chunks` + embeddings (pgvector) + FTS. Idempotent on `file_hash`/sha256.
- **Citation / shared-source network**: nodes = primary sources; edges = (a) sources shared across dossiers by `primary_url` (the spike's tier-1 bridges — exact), (b) extracted reference links where parseable. This is the layer the spike validated; **build it first.**
- Its Postgres + pgvector + Kuzu + FastMCP code lives on the **desktop** — see `[[research-kb-desktop]]`. First action: deploy there + point ingestion at the cache.

## 6. synthesis-kb (synthesis layer) — build spec

- Mirror research-kb's stack. Ingest the 36 strict-live dossiers via their `research_kb_export.jsonl` (the lossless envelope; toolkit already emits it) → `dossiers`/`claims`/`evidence`/`entities`/`dossier_concepts`/`bridges`.
- Resolve each evidence → research-kb primary via the three-tier identity (bibkey + sha256 → source/chunk UUID; mark `pending_resolution` if research-kb not yet populated).
- Tier-2 semantic bridges + concept graph layered after primaries land. MCP tool surface per 2026-05-21 §4.

## 7. Build sequence (primaries-first; de-risk gate PASSED ✅)

0. **[desktop]** deploy research-kb; point at `~/Claude/research_cache`. *(prerequisite — blocks 1,3)*
1. **research-kb primary ingestion + citation/shared-source network** — highest value first (delivers the validated source-bridges).
2. **Corpus normalization** (laptop, parallelizable now): consolidate 13 local caches · export ~20 stragglers · backfill anchors on 5 v2 dossiers. Feeds clean inputs to both KBs.
3. **synthesis-kb scaffold + ingest** the 36 dossiers; resolve evidence→primaries.
4. **Tier-2 semantic bridges + concept graph** (synthesis-kb).
5. **Visualization** export → brandon-behring.dev.
6. **Consistency audit** (deferred — design-doc M5).

## 8. Open decisions (carried, with triggers)

- **Multi-machine sync (Q14) — now ACTIVE & important.** research-kb runs on the desktop while the corpus + toolkit live on the laptop. Decide the model: **files canonical** (`claim_graph.jsonl`/`evidence_ledger.yml`/`cache_manifest.yml` + the 842-doc cache synced via git/cloud; each machine rebuilds its DB) vs. DB sync via `pg_dump`. Leaning files-canonical (matches the design's "files canonical, DB derived" invariant). Decide at desktop-deploy time.
- **Concept-extractor model tier** (Ollama / runpod / Claude) — decide before §4 (run the cost-estimate task on a claim sample).
- **bibkey ↔ source-UUID resolution (Q10)** — at research-kb ingestion (auto match by URL/DOI + sha256, manual override map for misses).
- **Consistency-audit mechanism** — M5.
- **Book integration (Q15)** — when the first dossier is assembled into a chapter.
