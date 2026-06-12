---
name: feedback-tests-vs-evals
description: For the KB work, Brandon distinguishes UNIT TESTS (code correctness, pre-deployment, first) from EVALS (does the deployed KG produce helpful info for the user, measured after deployment, later). "Evals" loosely said often means "unit tests first". research-kb is the maturity target to mirror.
metadata:
  node_type: memory
  type: feedback
---

When Brandon says "evals" for the KB he often means **two distinct things in sequence**, and wants them kept separate:

- **Unit tests** = is the CODE correct? Deterministic checks of the pipeline (resolution cascade picks the right source, anchor join derives the right primaries, bridge computation groups correctly, parsers parse). **These come FIRST, pre-deployment.**
- **Evals** = is the KG actually USEFUL? Measured on the **deployed** system — does querying it surface genuinely helpful, non-obvious information for the user. **These come AFTER deployment.**

**Why:** he wants confidence the machinery is correct before judging the *output's* usefulness — and usefulness can only be judged once it's live and queried for real. Conflating them (calling correctness-checks "evals") muddies the sequencing.

**How to apply:**
1. Build **real unit tests** for code correctness first (no stubs — see cross-project rule). Do NOT skip to "is it useful" judgments while the code is unverified.
2. **Mirror `~/Claude/research-kb/*`** — it is the mature sibling this repo (synthesis-kb) should age into; copy its test framework/layout/DB-fixture/mocking conventions rather than inventing.
3. **Don't enable/register the MCP tool** (make the KG queryable in-session) until unit tests are in place — testing gates deployment.
4. Defer eval design until post-deployment; when building evals, ground in the KG-eval material in `~/Claude/*` (he flagged there's a lot) rather than designing from scratch.

**research-kb test conventions to mirror (surveyed 2026-05-29):** pytest + `pytest-asyncio` with `asyncio_mode="auto"` (no per-test decorator); markers `unit`/`integration`/`slow`/`e2e` + `--strict-markers`; per-package `packages/*/tests/` (NOT top-level) + a top-level `tests/{e2e,integration,quality}/` for cross-package; **real test DB `research_kb_test`** (env-overridable) with schema applied once per session + **per-test transaction rollback** for isolation (not mocks, not testcontainers); pure-logic tests split into `test_*_unit.py` with NO db fixture; embedder/LLM/socket mocked via `monkeypatch` fixtures (fake 1024-dim `[[0.1]*1024]`); class-grouped Arrange-Act-Assert, `@pytest.mark.parametrize`, create deps through Store API. **Apply to synthesis-kb:** unit tests for the pure logic FIRST (parse_arxiv/parse_doi/norm_url, resolve_one cascade, canon(), load_envelope, build_cache_to_url, ordered(), bridge grouping), then DB-backed round-trip tests (ingest→resolve→anchor→bridge) against a `synthesis_kb_test` DB with rollback.

Relates to [[synthesis-kb-planned]], [[verify-against-source-of-truth]] (verify before claiming done — several "complete" claims this session were premature), and the cross-project "real tests only, no stubs" baseline.
