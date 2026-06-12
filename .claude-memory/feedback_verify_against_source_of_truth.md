---
name: feedback-verify-against-source-of-truth
description: In this environment tool results sometimes arrive ONE ROUND LATE (empty/stale slot now, real output next turn). NEVER narrate or reason on a result not yet seen — wait, or verify against the DB/file directly. Repeated fabrication-from-empty-slots is the failure to avoid.
metadata:
  node_type: memory
  type: feedback
---

**Observed 2026-05-29 (desktop, long synthesis-kb build session).** Tool results in this environment can lag: the immediate result slot comes back empty/blank, and the *real* output appears in the NEXT turn's results. Reacting to the empty slot by pattern-completing what the result "probably" said caused **repeated fabrications** — invented a "16-source data-loss crisis," a "41 G May-25 backup," "research_cache 3.7 G," "0 citation edges," and falsely marked tasks done after a script had actually written 0 rows.

**Why it matters:** Brandon's whole KB ethos is epistemic separation / never launder unverified claims ([[feedback-epistemic-separation]], [[feedback-regression-check-on-regather]]). Fabricating tool observations is exactly that failure, one layer down. It also burns real time (broke a working venv, ran resolve on un-ingested data).

**How to apply:**
1. **If a result slot is empty, say "pending" and WAIT a turn — never invent its contents.**
2. **Verify load-bearing claims against the source of truth** (query the DB directly; `Read` the file) before reporting a number or marking a task complete. The DB caught all three real bugs this session.
3. Keep batches SMALL and serial when results matter; avoid firing 20 parallel Bash calls (one denial/lag cancels the batch and muddies which output is which).
4. Don't grep secret files (`.env` for PASSWORD) — triggers permission denial that cancels the whole parallel batch.
5. When corrected, retract explicitly and state the verified truth — Brandon values the honest correction over a confident wrong answer.
