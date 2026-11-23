"""Deduplication and freshness store.

Provides abstract DeduplicationStore and two backends:
- RedisDedup: SETNX for cross-node distributed dedup
- SqliteDedup: Local SQLite table fallback for single-node execution
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import sqlite3
import structlog

from src.config import Settings, get_settings

logger = structlog.get_logger("storage.seen")


class DeduplicationStore(ABC):
    """Abstract interface for checking and recording seen URLs and content hashes."""

    @abstractmethod
    async def is_new(
        self,
        url_hash: str,
        content_hash: str,
        source: str,
        record_type: str,
    ) -> bool:
        """Return True if url_hash is unseen OR if content_hash has changed."""
        pass

    @abstractmethod
    async def mark_seen(
        self,
        url_hash: str,
        content_hash: str,
        source: str,
        record_type: str,
    ) -> None:
        """Record the item as seen, updating last_seen and content_hash."""
        pass

    @abstractmethod
    async def get_first_seen(self, url_hash: str) -> Optional[datetime]:
        """Return the UTC timestamp when the URL was first ingested."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Release underlying connections."""
        pass

    async def __aenter__(self) -> DeduplicationStore:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()


class SqliteDedup(DeduplicationStore):
    """SQLite-backed deduplication store for local development and single-node runs."""

    def __init__(self, db_path: str = "data/pipeline.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        with self._conn:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS seen_items (
                    url_hash TEXT PRIMARY KEY,
                    content_hash TEXT,
                    source TEXT,
                    record_type TEXT,
                    first_seen TEXT,
                    last_seen TEXT,
                    status TEXT DEFAULT 'active'
                )
            """)
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_seen_source ON seen_items(source)")

    async def is_new(
        self,
        url_hash: str,
        content_hash: str,
        source: str,
        record_type: str,
    ) -> bool:
        async with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                "SELECT content_hash FROM seen_items WHERE url_hash = ?",
                (url_hash,),
            )
            row = cursor.fetchone()
            if row is None:
                return True
            # New if content has changed since last crawled
            return row["content_hash"] != content_hash

    async def mark_seen(
        self,
        url_hash: str,
        content_hash: str,
        source: str,
        record_type: str,
    ) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        async with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO seen_items (url_hash, content_hash, source, record_type, first_seen, last_seen, status)
                    VALUES (?, ?, ?, ?, ?, ?, 'active')
                    ON CONFLICT(url_hash) DO UPDATE SET
                        content_hash = excluded.content_hash,
                        last_seen = excluded.last_seen,
                        status = 'active'
                    """,
                    (url_hash, content_hash, source, record_type, now_iso, now_iso),
                )

    async def get_first_seen(self, url_hash: str) -> Optional[datetime]:
        async with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                "SELECT first_seen FROM seen_items WHERE url_hash = ?",
                (url_hash,),
            )
            row = cursor.fetchone()
            if row and row["first_seen"]:
                try:
                    dt = datetime.fromisoformat(row["first_seen"])
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return dt.astimezone(timezone.utc)
                except Exception:
                    return None
            return None

    async def close(self) -> None:
        async with self._lock:
            self._conn.close()


class RedisDedup(DeduplicationStore):
    """Redis-backed deduplication store for distributed multi-worker pipelines."""

    def __init__(self, redis_url: str) -> None:
        import redis.asyncio as aioredis
        self.redis = aioredis.from_url(redis_url, decode_responses=True)

    async def is_new(
        self,
        url_hash: str,
        content_hash: str,
        source: str,
        record_type: str,
    ) -> bool:
        key = f"seen:{url_hash}"
        existing_content = await self.redis.hget(key, "content_hash")
        if existing_content is None:
            return True
        return existing_content != content_hash

    async def mark_seen(
        self,
        url_hash: str,
        content_hash: str,
        source: str,
        record_type: str,
    ) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        key = f"seen:{url_hash}"
        pipe = self.redis.pipeline()
        pipe.hset(key, mapping={
            "url_hash": url_hash,
            "content_hash": content_hash,
            "source": source,
            "record_type": record_type,
            "last_seen": now_iso,
        })
        # Set first_seen only if not exists
        pipe.hsetnx(key, "first_seen", now_iso)
        await pipe.execute()

    async def get_first_seen(self, url_hash: str) -> Optional[datetime]:
        key = f"seen:{url_hash}"
        first_seen_str = await self.redis.hget(key, "first_seen")
        if first_seen_str:
            try:
                dt = datetime.fromisoformat(first_seen_str)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except Exception:
                return None
        return None

    async def close(self) -> None:
        await self.redis.close()


def create_dedup_store(settings: Optional[Settings] = None) -> DeduplicationStore:
    """Factory creating Redis dedup if configured and reachable, falling back to SQLite."""
    cfg = settings or get_settings()
    if cfg.REDIS_URL:
        try:
            import redis
            # Quick connectivity check to verify server is reachable
            client = redis.Redis.from_url(cfg.REDIS_URL, socket_timeout=1.0, socket_connect_timeout=1.0)
            client.ping()
            client.close()
            logger.info("Initializing Redis deduplication store", url=cfg.REDIS_URL)
            return RedisDedup(cfg.REDIS_URL)
        except Exception as e:
            logger.warning("Failed to connect to Redis, falling back to SQLite", error=str(e))
    return SqliteDedup(cfg.SQLITE_DB_PATH)
