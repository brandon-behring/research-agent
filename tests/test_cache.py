"""Tests for the SQLite report cache.

Tests query normalization, cache key computation, put/get lifecycle,
TTL expiration, graceful degradation, and CLI integration.
"""

from __future__ import annotations

import json
import time
from io import StringIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from research_agent.cache import ReportCache, _normalize_query, compute_cache_key


class TestNormalizeQuery:
    """Tests for query normalization."""

    def test_lowercase(self) -> None:
        """Mixed case is lowercased."""
        assert _normalize_query("What Is DML?") == "what is dml?"

    def test_strip(self) -> None:
        """Leading and trailing whitespace removed."""
        assert _normalize_query("  what is dml?  ") == "what is dml?"

    def test_collapse_whitespace(self) -> None:
        """Multiple spaces collapsed to single space."""
        assert _normalize_query("what   is   dml?") == "what is dml?"

    def test_tabs_and_newlines(self) -> None:
        """Tabs and newlines treated as whitespace."""
        assert _normalize_query("what\tis\ndml?") == "what is dml?"


class TestComputeCacheKey:
    """Tests for cache key computation."""

    def test_deterministic(self) -> None:
        """Same inputs produce same key."""
        key1 = compute_cache_key("What is DML?", 10, 15, 20, "sonnet")
        key2 = compute_cache_key("What is DML?", 10, 15, 20, "sonnet")
        assert key1 == key2

    def test_normalization_invariance(self) -> None:
        """Equivalent queries after normalization produce same key."""
        key1 = compute_cache_key("What is DML?", 10, 15, 20, "sonnet")
        key2 = compute_cache_key("  what  is  dml?  ", 10, 15, 20, "sonnet")
        assert key1 == key2

    def test_different_query_different_key(self) -> None:
        """Different queries produce different keys."""
        key1 = compute_cache_key("What is DML?", 10, 15, 20, "sonnet")
        key2 = compute_cache_key("What is IV?", 10, 15, 20, "sonnet")
        assert key1 != key2

    def test_different_config_different_key(self) -> None:
        """Different config params produce different keys."""
        key1 = compute_cache_key("What is DML?", 10, 15, 20, "sonnet")
        key2 = compute_cache_key("What is DML?", 5, 15, 20, "sonnet")
        assert key1 != key2

    def test_different_model_different_key(self) -> None:
        """Different synthesis model produces different key."""
        key1 = compute_cache_key("What is DML?", 10, 15, 20, "sonnet")
        key2 = compute_cache_key("What is DML?", 10, 15, 20, "opus")
        assert key1 != key2

    def test_hex_format(self) -> None:
        """Key is a valid 64-char hex string (SHA-256)."""
        key = compute_cache_key("test", 10, 15, 20, "sonnet")
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)


class TestReportCache:
    """Tests for ReportCache put/get lifecycle."""

    def test_put_get_roundtrip(self, tmp_path: Path) -> None:
        """Put then get returns the cached entry."""
        cache = ReportCache(tmp_path / "cache.db")
        metadata = json.dumps({"source_count": 5})
        config = json.dumps({"max_search_results": 10})
        cache.put("key1", "What is DML?", "# Report", metadata, config)

        entry = cache.get("key1")
        assert entry is not None
        assert entry.report == "# Report"
        assert entry.metadata["source_count"] == 5
        assert "cached_at" in entry.metadata
        cache.close()

    def test_miss_returns_none(self, tmp_path: Path) -> None:
        """Non-existent key returns None."""
        cache = ReportCache(tmp_path / "cache.db")
        assert cache.get("nonexistent") is None
        cache.close()

    def test_ttl_expiry(self, tmp_path: Path) -> None:
        """Expired entries return None."""
        cache = ReportCache(tmp_path / "cache.db", ttl_hours=0.001)
        cache.put("key1", "q", "report", "{}", "{}")
        # Backdate entry to simulate expiry
        cache._conn.execute(  # type: ignore[union-attr]
            "UPDATE report_cache SET created_at = ? WHERE cache_key = ?",
            (time.time() - 100, "key1"),
        )
        cache._conn.commit()  # type: ignore[union-attr]
        assert cache.get("key1") is None
        cache.close()

    def test_expired_entry_deleted_on_get(self, tmp_path: Path) -> None:
        """Expired entries are cleaned up on access."""
        cache = ReportCache(tmp_path / "cache.db", ttl_hours=0.001)
        cache.put("key1", "q", "report", "{}", "{}")
        cache._conn.execute(  # type: ignore[union-attr]
            "UPDATE report_cache SET created_at = ? WHERE cache_key = ?",
            (time.time() - 100, "key1"),
        )
        cache._conn.commit()  # type: ignore[union-attr]
        cache.get("key1")  # Triggers deletion

        cursor = cache._conn.execute(  # type: ignore[union-attr]
            "SELECT COUNT(*) FROM report_cache"
        )
        assert cursor.fetchone()[0] == 0
        cache.close()

    def test_overwrite_existing_key(self, tmp_path: Path) -> None:
        """INSERT OR REPLACE overwrites existing entries."""
        cache = ReportCache(tmp_path / "cache.db")
        cache.put("key1", "q", "old report", "{}", "{}")
        cache.put("key1", "q", "new report", "{}", "{}")

        entry = cache.get("key1")
        assert entry is not None
        assert entry.report == "new report"
        cache.close()

    def test_clear_returns_count(self, tmp_path: Path) -> None:
        """clear() returns the number of deleted entries."""
        cache = ReportCache(tmp_path / "cache.db")
        cache.put("key1", "q1", "r1", "{}", "{}")
        cache.put("key2", "q2", "r2", "{}", "{}")

        count = cache.clear()
        assert count == 2
        assert cache.get("key1") is None
        assert cache.get("key2") is None
        cache.close()

    def test_evict_expired(self, tmp_path: Path) -> None:
        """evict_expired() removes only expired entries."""
        cache = ReportCache(tmp_path / "cache.db", ttl_hours=1.0)
        cache.put("old", "q", "r", "{}", "{}")
        cache.put("new", "q", "r", "{}", "{}")

        # Backdate one entry past TTL
        cache._conn.execute(  # type: ignore[union-attr]
            "UPDATE report_cache SET created_at = ? WHERE cache_key = ?",
            (time.time() - 7200, "old"),
        )
        cache._conn.commit()  # type: ignore[union-attr]

        evicted = cache.evict_expired()
        assert evicted == 1
        assert cache.get("new") is not None
        assert cache.get("old") is None
        cache.close()

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """Parent directories are created automatically (mkdir -p)."""
        deep_path = tmp_path / "a" / "b" / "c" / "cache.db"
        cache = ReportCache(deep_path)
        assert deep_path.parent.exists()
        cache.put("key1", "q", "r", "{}", "{}")
        assert cache.get("key1") is not None
        cache.close()

    def test_context_manager_closes(self, tmp_path: Path) -> None:
        """Context manager closes connection on exit."""
        with ReportCache(tmp_path / "cache.db") as cache:
            cache.put("key1", "q", "r", "{}", "{}")
            assert cache._conn is not None
        assert cache._conn is None

    def test_enabled_false_is_noop(self, tmp_path: Path) -> None:
        """enabled=False makes all methods no-op without touching disk."""
        with ReportCache(tmp_path / "cache.db", enabled=False) as cache:
            assert cache._conn is None
            assert cache.get("key1") is None
            cache.put("key1", "q", "r", "{}", "{}")
            assert cache.clear() == 0
            assert cache.evict_expired() == 0


class TestReportCacheGracefulDegradation:
    """Tests for graceful degradation on errors."""

    def test_unwritable_path(self) -> None:
        """Cache degrades gracefully when path is unwritable."""
        cache = ReportCache(Path("/proc/nonexistent/cache.db"))
        assert cache.get("key") is None
        cache.put("key", "q", "r", "{}", "{}")  # No-op, no exception
        assert cache.clear() == 0
        cache.close()

    def test_corrupted_db(self, tmp_path: Path) -> None:
        """Cache handles corrupted database gracefully."""
        db_path = tmp_path / "cache.db"
        db_path.write_text("not a sqlite database")
        cache = ReportCache(db_path)
        # Operations should not raise regardless of init outcome
        cache.get("key")
        cache.put("key", "q", "r", "{}", "{}")
        cache.close()

    def test_closed_connection(self, tmp_path: Path) -> None:
        """Operations on closed cache return None/no-op."""
        cache = ReportCache(tmp_path / "cache.db")
        cache.close()
        assert cache.get("key") is None
        cache.put("key", "q", "r", "{}", "{}")  # No-op
        assert cache.clear() == 0

    def test_corrupted_metadata_json_deleted(self, tmp_path: Path) -> None:
        """Corrupt metadata_json entry returns None and is deleted from DB."""
        cache = ReportCache(tmp_path / "cache.db")
        cache.put("key1", "q", "report", "{}", "{}")

        # Corrupt the metadata_json column directly
        cache._conn.execute(  # type: ignore[union-attr]
            "UPDATE report_cache SET metadata_json = ? WHERE cache_key = ?",
            ("not-valid-json{{{", "key1"),
        )
        cache._conn.commit()  # type: ignore[union-attr]

        # get() should return None (graceful degradation)
        assert cache.get("key1") is None

        # Corrupt entry should have been deleted
        cursor = cache._conn.execute(  # type: ignore[union-attr]
            "SELECT COUNT(*) FROM report_cache WHERE cache_key = ?",
            ("key1",),
        )
        assert cursor.fetchone()[0] == 0
        cache.close()


class TestExtractMetadata:
    """Tests for _extract_metadata() from cli.py."""

    def test_with_pydantic_like_objects(self) -> None:
        """Extracts names from objects with .name / .method_name attrs."""
        from research_agent.cli import _extract_metadata

        concept = MagicMock()
        concept.name = "double_machine_learning"
        audit = MagicMock()
        audit.method_name = "propensity_score"

        result = _extract_metadata(
            {
                "search_results": [{"id": 1}, {"id": 2}],
                "concepts": [concept],
                "assumption_audits": [audit],
            }
        )
        assert result["source_count"] == 2
        assert result["concept_names"] == ["double_machine_learning"]
        assert result["methods_audited"] == ["propensity_score"]

    def test_with_empty_result(self) -> None:
        """Empty result dict yields zero-value metadata."""
        from research_agent.cli import _extract_metadata

        result = _extract_metadata({})
        assert result == {
            "source_count": 0,
            "concept_names": [],
            "methods_audited": [],
        }


class TestCLICacheIntegration:
    """Tests for cache integration in CLI main()."""

    def test_cache_hit_skips_pipeline(self, tmp_path: Path) -> None:
        """Cache hit returns report without running the pipeline."""
        from research_agent.cli import main

        # Pre-populate cache
        cache = ReportCache(tmp_path / "cache.db")
        key = compute_cache_key("What is DML?", 10, 15, 20, "claude-sonnet-4-6")
        cache.put(key, "What is DML?", "# Cached Report", "{}", "{}")
        cache.close()

        mock_config = MagicMock(
            cache_enabled=True,
            cache_db_path=str(tmp_path / "cache.db"),
            cache_ttl_hours=24.0,
            max_search_results=10,
            max_concepts=15,
            max_citations=20,
            models=MagicMock(synthesis="claude-sonnet-4-6"),
        )
        with (
            patch("sys.argv", ["cli", "What is DML?"]),
            patch("research_agent.cli.AgentConfig", return_value=mock_config),
            patch("research_agent.cli.run_research", new_callable=AsyncMock) as mock_run,
            patch("sys.stdout", new_callable=StringIO) as mock_stdout,
            patch("sys.stderr", new_callable=StringIO) as mock_stderr,
        ):
            main()
            mock_run.assert_not_called()
            assert "Cached Report" in mock_stdout.getvalue()
            assert "[cached]" in mock_stderr.getvalue()

    def test_no_cache_flag_bypasses(self, tmp_path: Path) -> None:
        """--no-cache bypasses cache even when entry exists."""
        from research_agent.cli import main

        # Pre-populate cache
        cache = ReportCache(tmp_path / "cache.db")
        key = compute_cache_key("What is DML?", 10, 15, 20, "claude-sonnet-4-6")
        cache.put(key, "What is DML?", "# Cached", "{}", "{}")
        cache.close()

        mock_config = MagicMock(
            cache_enabled=True,
            cache_db_path=str(tmp_path / "cache.db"),
            cache_ttl_hours=24.0,
            max_search_results=10,
            max_concepts=15,
            max_citations=20,
            models=MagicMock(synthesis="claude-sonnet-4-6"),
        )
        mock_result = {"report": "# Fresh Report"}
        with (
            patch("sys.argv", ["cli", "--no-cache", "What is DML?"]),
            patch("research_agent.cli.AgentConfig", return_value=mock_config),
            patch("research_agent.cli.run_research", new_callable=AsyncMock) as mock_run,
            patch("sys.stdout", new_callable=StringIO) as mock_stdout,
        ):
            mock_run.return_value = mock_result
            main()
            mock_run.assert_called_once()
            assert "Fresh Report" in mock_stdout.getvalue()

    def test_cache_disabled_config(self) -> None:
        """cache_enabled=False skips all cache operations."""
        from research_agent.cli import main

        mock_config = MagicMock(
            cache_enabled=False,
            max_search_results=10,
            max_concepts=15,
            max_citations=20,
            models=MagicMock(synthesis="claude-sonnet-4-6"),
        )
        mock_result = {"report": "# Report"}
        with (
            patch("sys.argv", ["cli", "What is DML?"]),
            patch("research_agent.cli.AgentConfig", return_value=mock_config),
            patch("research_agent.cli.run_research", new_callable=AsyncMock) as mock_run,
            patch("sys.stdout", new_callable=StringIO),
        ):
            mock_run.return_value = mock_result
            main()
            mock_run.assert_called_once()

    def test_clear_cache_exits_early(self, tmp_path: Path) -> None:
        """--clear-cache deletes all entries and exits without running pipeline."""
        from research_agent.cli import main

        # Pre-populate cache
        cache = ReportCache(tmp_path / "cache.db")
        cache.put("key1", "q", "r", "{}", "{}")
        cache.close()

        mock_config = MagicMock(
            cache_enabled=True,
            cache_db_path=str(tmp_path / "cache.db"),
            cache_ttl_hours=24.0,
        )
        with (
            patch("sys.argv", ["cli", "--clear-cache"]),
            patch("research_agent.cli.AgentConfig", return_value=mock_config),
            patch("research_agent.cli.run_research", new_callable=AsyncMock) as mock_run,
            patch("sys.stderr", new_callable=StringIO) as mock_stderr,
        ):
            main()
            mock_run.assert_not_called()
            assert "Cleared" in mock_stderr.getvalue()
