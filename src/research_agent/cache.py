"""SQLite-backed report cache for the research agent.

Caches pipeline reports keyed on query hash + config parameters.
Repeated queries skip the full pipeline and return cached reports.

Design:
    - stdlib only (sqlite3, hashlib, json, re, time, pathlib, dataclasses)
    - Graceful degradation: all public methods catch sqlite3.Error/OSError
    - TTL-based invalidation with opportunistic cleanup
    - Cache key includes output-affecting config (model, limits) but
      excludes transport/timeout/planning model
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _normalize_query(query: str) -> str:
    """Normalize query for consistent cache keys.

    Lowercase, strip whitespace, collapse internal whitespace.
    ``"What is DML?"`` == ``"what  is  dml?  "``

    Args:
        query: Raw user query.

    Returns:
        Normalized query string.
    """
    return re.sub(r"\s+", " ", query.strip().lower())


def compute_cache_key(
    query: str,
    max_search_results: int,
    max_concepts: int,
    max_citations: int,
    synthesis_model: str,
) -> str:
    """Compute SHA-256 cache key from query + output-affecting config.

    Only includes parameters that change the report content.
    Excludes: transport, timeout, planning model.

    Args:
        query: Research question.
        max_search_results: MAX_SEARCH_RESULTS config value.
        max_concepts: MAX_CONCEPTS config value.
        max_citations: MAX_CITATIONS config value.
        synthesis_model: SYNTHESIS_MODEL config value.

    Returns:
        Hex digest of the SHA-256 hash.
    """
    normalized = _normalize_query(query)
    key_input = json.dumps(
        {
            "query": normalized,
            "max_search_results": max_search_results,
            "max_concepts": max_concepts,
            "max_citations": max_citations,
            "synthesis_model": synthesis_model,
        },
        sort_keys=True,
    )
    return hashlib.sha256(key_input.encode()).hexdigest()


@dataclass(frozen=True)
class CacheEntry:
    """A cached report with metadata.

    Attributes:
        report: The cached report text.
        metadata: Summary metadata (source_count, concept_names, methods_audited, cached_at).
    """

    report: str
    metadata: dict[str, Any]


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS report_cache (
    cache_key     TEXT PRIMARY KEY,
    query         TEXT NOT NULL,
    report        TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at    REAL NOT NULL,
    config_json   TEXT NOT NULL
)
"""

_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_cache_created ON report_cache (created_at)
"""


class ReportCache:
    """SQLite-backed report cache with TTL expiration.

    Graceful degradation: every public method catches sqlite3.Error and OSError,
    logs a warning, and returns None or no-op. Callers never handle cache exceptions.

    Args:
        db_path: Path to SQLite database file. Parent directories created automatically.
        ttl_hours: Hours before cached reports expire.

    Example::

        cache = ReportCache(Path("~/.cache/research-agent/cache.db").expanduser())
        entry = cache.get(cache_key)
        if entry is not None:
            print(entry.report)
        cache.close()
    """

    def __init__(self, db_path: Path, ttl_hours: float = 24.0) -> None:
        self._db_path = db_path
        self._ttl_seconds = ttl_hours * 3600.0
        self._conn: sqlite3.Connection | None = None
        try:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(db_path))
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(_CREATE_TABLE_SQL)
            self._conn.execute(_CREATE_INDEX_SQL)
            self._conn.commit()
        except (sqlite3.Error, OSError) as e:
            logger.warning("Cache initialization failed: %s", e)
            self._conn = None

    def get(self, cache_key: str) -> CacheEntry | None:
        """Retrieve a cached report by key.

        Returns None on miss, expiry, or error. Expired entries are deleted on access.

        Args:
            cache_key: SHA-256 hex digest from compute_cache_key().

        Returns:
            CacheEntry on hit, None on miss/expiry/error.
        """
        if self._conn is None:
            return None
        try:
            cursor = self._conn.execute(
                "SELECT report, metadata_json, created_at FROM report_cache WHERE cache_key = ?",
                (cache_key,),
            )
            row = cursor.fetchone()
            if row is None:
                return None

            report, metadata_json, created_at = row
            age = time.time() - created_at
            if age > self._ttl_seconds:
                self._conn.execute(
                    "DELETE FROM report_cache WHERE cache_key = ?",
                    (cache_key,),
                )
                self._conn.commit()
                return None

            metadata = json.loads(metadata_json)
            metadata["cached_at"] = created_at
            return CacheEntry(report=report, metadata=metadata)
        except (sqlite3.Error, json.JSONDecodeError) as e:
            logger.warning("Cache get failed: %s", e)
            return None

    def put(
        self,
        cache_key: str,
        query: str,
        report: str,
        metadata_json: str,
        config_json: str,
    ) -> None:
        """Store a report in the cache.

        Uses INSERT OR REPLACE to handle key collisions.

        Args:
            cache_key: SHA-256 hex digest.
            query: Original query text.
            report: Report text to cache.
            metadata_json: JSON-encoded metadata dict.
            config_json: JSON-encoded config summary.
        """
        if self._conn is None:
            return
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO report_cache "
                "(cache_key, query, report, metadata_json, created_at, config_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (cache_key, query, report, metadata_json, time.time(), config_json),
            )
            self._conn.commit()
        except sqlite3.Error as e:
            logger.warning("Cache put failed: %s", e)

    def clear(self) -> int:
        """Delete all cached entries.

        Returns:
            Number of entries deleted.
        """
        if self._conn is None:
            return 0
        try:
            cursor = self._conn.execute("DELETE FROM report_cache")
            self._conn.commit()
            return cursor.rowcount
        except sqlite3.Error as e:
            logger.warning("Cache clear failed: %s", e)
            return 0

    def evict_expired(self) -> int:
        """Bulk delete expired entries.

        Called opportunistically after cache writes.

        Returns:
            Number of entries evicted.
        """
        if self._conn is None:
            return 0
        try:
            cutoff = time.time() - self._ttl_seconds
            cursor = self._conn.execute(
                "DELETE FROM report_cache WHERE created_at < ?",
                (cutoff,),
            )
            self._conn.commit()
            return cursor.rowcount
        except sqlite3.Error as e:
            logger.warning("Cache evict failed: %s", e)
            return 0

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            with contextlib.suppress(sqlite3.Error):
                self._conn.close()
            self._conn = None
