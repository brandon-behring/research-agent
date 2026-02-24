# Research Agent: Best Practices Upgrade

**Created**: 2026-02-24
**Status**: implemented
**Completed**: 2026-02-24
**Estimated LOC**: ~2,000+

## Phases

1. **Type Foundation** — Pydantic v2 models + structured LLM output
2. **Resilience & Observability** — Exception hierarchy, retries, timeouts, LangSmith
3. **Evals Framework** — Golden cases, LLM-as-judge, metrics dashboard
4. **Testing & Tooling** — mypy strict, comprehensive tests, uv, pre-commit, CI

## Decisions Made

- Evals: Full suite + LLM-as-judge + metrics dashboard
- Deps: uv for dependency management + lock file
- MCP parsers: Harden (defensive checks + parameterized tests), keep regex
- Observability: LangSmith only (graceful no-op without key)
- Execution: Phase commits (4 commits, each independently testable)
- Config: BaseSettings with explicit overrides (tests pass values directly)
- LangGraph v1.0.9 fully supports Pydantic BaseModel state

## Baseline

- 33 tests passing
- ~1,550 LOC
- No type checker, no lock file, no evals
