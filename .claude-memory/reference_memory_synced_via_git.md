---
name: reference-memory-synced-via-git
description: Claude auto-memory for this project lives at ~/Claude/research-agent/.claude-memory/ and is symlinked from ~/.claude/projects/.../memory. It is tracked in git for cross-machine sync.
metadata:
  type: reference
---

**Layout (since 2026-05-27)**

- Canonical location: `~/Claude/research-agent/.claude-memory/` (tracked in git)
- Auto-memory read path: `~/.claude/projects/-home-brandon-behring-Claude-research-agent/memory` → symlink → canonical location

**Implications**

- Memory writes (new files, edits) land in the repo working tree. They will appear in `git status` as modifications/additions. This is expected, not a bug.
- After substantive memory changes, commit + push to sync to the other machine.
- On a fresh clone (other machine), recreate the symlink before starting Claude:
  ```bash
  mkdir -p ~/.claude/projects/-home-brandon-behring-Claude-research-agent
  ln -s ~/Claude/research-agent/.claude-memory \
        ~/.claude/projects/-home-brandon-behring-Claude-research-agent/memory
  ```
- The surrounding `~/.claude/projects/-home-brandon-behring-Claude-research-agent/*.jsonl` session transcripts are **not** synced; only the curated memory.

**Why this exists**

User requested cross-machine continuity for project + Claude's curated notes. Audit confirmed no secrets/PII in memory beyond what's already in git. See commit `dc1f481` (chore: track Claude auto-memory under .claude-memory/).
