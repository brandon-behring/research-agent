---
name: feedback-pacing-one-question
description: "For deep design exploration (especially via /exploring-options), present one focused question per round with depth, not bulk question batteries. User rejects multi-question batches."
metadata:
  node_type: memory
  type: feedback
  originSessionId: 1b78bb6c-1471-47e2-8fd6-4d21b96dec31
---

When the user invokes `/exploring-options` or otherwise signals they want to think through a design, **present one architectural decision at a time**, with full treatment (context, options with pros/cons, recommendation with reasoning, references to existing patterns, tradeoffs). Wait for an answer. Then present the next.

**Why**: This user rejected three-question and six-question batches in succession ("that is a big question", "park this; bigger question first"). They engage well when each question is given depth and pacing room. Bulk batteries feel overwhelming and signal poor thinking.

**How to apply**:
- Default to one question per `AskUserQuestion` call, even when the call supports up to four.
- Exception: deferred / milestone-tier decisions where the user has explicitly opted into a batch (e.g., "drill into the deferred decisions") — even then, prefer two per call max.
- If you have many open questions, name them in a list ("I see N decisions still open") and let the user pick which to explore first. Do not assume.
- The user signals "ready to lock" via the second AskUserQuestion option, not via ExitPlanMode approval directly.
- Plan-file edits during exploration should be incremental — update the plan as decisions resolve, do not rewrite large sections per round.

Tested with: synthesis-kb design session (this conversation), where 9 decisions were resolved across 10 rounds.
