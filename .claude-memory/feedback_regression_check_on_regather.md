---
name: feedback-regression-check-on-regather
description: "When re-gathering or migrating dossiers (v2→v3, fresh re-runs, schema migrations), compare new vs. old as a regression check before treating new as authoritative — never silently lose curatorial work"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 4ff2c9b9-809c-44e0-b26b-a1f96a7ea267
---

When a dossier is re-gathered (e.g., v2→v3 schema migration, fresh re-run of `/research-plan`+`/research-gather`, or any work that *recreates* an existing curated artifact), **the new output is NOT authoritative by default**. Compare against the prior version first.

**Why:** the user has invested curatorial work in the v2/old version — specific source selections, scope decisions, claim framings. A fresh re-run by an agent may drop sources (different judgment under different prompting), drop claims (sub-areas the new agent didn't see as essential), or change interpretations. Silently treating the new version as canonical *launders* that curatorial loss invisible.

**How to apply:**
1. After every regather/migration completes, produce an explicit regression comparison report (use `~/Claude/research_toolkit/scripts/compare_v2_v3.py` or equivalent): sources in old not in new (potential regressions), sources in new not in old (enrichments), counts deltas, per-source claim coverage if comparable.
2. Default rules: enrichments → new is authoritative for that source; same source in both with new having anchors → new is authoritative; **losses (in-old-not-in-new) → surface for user judgment** — is this intentional dedup/scope change, or a regression to recover?
3. Never delete or supersede the v2/old artifact until the comparison shows zero unaccounted-for regressions or the user explicitly accepts each one.
4. Pattern matches the broader [[feedback-epistemic-separation]] discipline: never conflate primary literature with synthesized claims; here, never conflate "new artifact exists" with "new artifact preserves old artifact's work."

Related: [[feedback-understand-before-executing]].
