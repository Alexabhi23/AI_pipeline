"""SQLite state store for the GraphOne pipeline.

Implements all 6 state schema tables:
1. seen_items: url_hash PK, content_hash, source, record_type, first_seen, last_seen, status
2. raw_pages: url_hash PK, url, http_status, fetched_at, html_path
3. records: id PK, record_type, url_hash, payload_json, date_source, date_confidence, collected_at, UNIQUE(url_hash, record_type)
4. entity_map: id PK, raw_name, canonical_name, method, score, created_at
5. llm_calls: id PK, tier, status_code, retries, tokens_in, latency_ms, ts, success
6. seed_entities: canonical PK, aliases_json
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Union

import structlog

from src.config import Settings, get_settings
from src.models import BaseRecord, EntityMapping, LLMCallLog, RecordType

logger = structlog.get_logger("storage.database")


class Database:
    """Manages SQLite tables for the pipeline run."""

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        cfg = settings or get_settings()
        self.db_path = Path(db_path or cfg.SQLITE_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        with self.conn:
            # 1. seen_items
            self.conn.execute("""
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
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_seen_status ON seen_items(status)")

            # 2. raw_pages
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS raw_pages (
                    url_hash TEXT PRIMARY KEY,
                    url TEXT NOT NULL,
                    http_status INTEGER NOT NULL,
                    fetched_at TEXT NOT NULL,
                    html_path TEXT
                )
            """)

            # 3. records
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    record_type TEXT NOT NULL,
                    url_hash TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    date_source TEXT,
                    date_confidence TEXT,
                    collected_at TEXT NOT NULL,
                    UNIQUE(url_hash, record_type)
                )
            """)
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_records_type ON records(record_type)")

            # 4. entity_map
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS entity_map (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    raw_name TEXT NOT NULL,
                    canonical_name TEXT NOT NULL,
                    method TEXT NOT NULL,
                    score REAL NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)

            # 5. llm_calls
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS llm_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tier TEXT NOT NULL,
                    status_code INTEGER,
                    retries INTEGER DEFAULT 0,
                    tokens_in INTEGER,
                    latency_ms REAL,
                    ts TEXT NOT NULL,
                    success INTEGER DEFAULT 1
                )
            """)

            # 6. seed_entities
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS seed_entities (
                    canonical TEXT PRIMARY KEY,
                    aliases_json TEXT NOT NULL
                )
            """)

    def save_raw_page(
        self,
        url_hash: str,
        url: str,
        http_status: int,
        fetched_at: Union[str, datetime],
        html_path: Optional[str] = None,
    ) -> None:
        """Store fetched page trace."""
        ts_str = fetched_at.isoformat() if isinstance(fetched_at, datetime) else str(fetched_at)
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO raw_pages (url_hash, url, http_status, fetched_at, html_path)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(url_hash) DO UPDATE SET
                    http_status = excluded.http_status,
                    fetched_at = excluded.fetched_at,
                    html_path = excluded.html_path
                """,
                (url_hash, url, http_status, ts_str, html_path),
            )

    def save_record(self, record: BaseRecord) -> int:
        """Store validated domain record. Returns the record ID."""
        row = record.to_db_row()
        with self.conn:
            cursor = self.conn.execute(
                """
                INSERT INTO records (record_type, url_hash, payload_json, date_source, date_confidence, collected_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(url_hash, record_type) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    date_source = excluded.date_source,
                    date_confidence = excluded.date_confidence,
                    collected_at = excluded.collected_at
                """,
                (
                    row["record_type"],
                    row["url_hash"],
                    row["payload_json"],
                    row["date_source"],
                    row["date_confidence"],
                    row["collected_at"],
                ),
            )
            return cursor.lastrowid or 1

    def save_entity_mapping(self, mapping: EntityMapping) -> int:
        """Record entity resolution decision."""
        with self.conn:
            cursor = self.conn.execute(
                """
                INSERT INTO entity_map (raw_name, canonical_name, method, score, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    mapping.rawName,
                    mapping.canonicalName,
                    mapping.method,
                    mapping.score,
                    mapping.createdAt.isoformat(),
                ),
            )
            return cursor.lastrowid or 1

    def save_llm_call(self, log: LLMCallLog) -> int:
        """Record LLM call telemetry."""
        with self.conn:
            cursor = self.conn.execute(
                """
                INSERT INTO llm_calls (tier, status_code, retries, tokens_in, latency_ms, ts, success)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    log.tier,
                    log.statusCode,
                    log.retries,
                    log.tokensIn,
                    log.latencyMs,
                    log.timestamp.isoformat(),
                    1 if log.success else 0,
                ),
            )
            return cursor.lastrowid or 1

    def seed_initial_entities(self, entities: dict[str, list[str]]) -> None:
        """Seed canonical entities and their aliases."""
        with self.conn:
            for canonical, aliases in entities.items():
                self.conn.execute(
                    """
                    INSERT INTO seed_entities (canonical, aliases_json)
                    VALUES (?, ?)
                    ON CONFLICT(canonical) DO UPDATE SET
                        aliases_json = excluded.aliases_json
                    """,
                    (canonical, json.dumps(aliases)),
                )

    def get_records(self, record_type: Optional[RecordType] = None) -> List[Dict[str, Any]]:
        """Retrieve stored records with parsed payload JSON."""
        cursor = self.conn.cursor()
        if record_type:
            val = record_type.value if hasattr(record_type, "value") else str(record_type)
            cursor.execute(
                "SELECT * FROM records WHERE record_type = ? ORDER BY id ASC",
                (val,),
            )
        else:
            cursor.execute("SELECT * FROM records ORDER BY id ASC")

        results = []
        for row in cursor.fetchall():
            item = dict(row)
            try:
                item["payload"] = json.loads(item["payload_json"])
            except Exception:
                item["payload"] = {}
            results.append(item)
        return results

    def get_entity_mappings(self) -> List[Dict[str, Any]]:
        """Return all logged entity resolution mappings."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM entity_map ORDER BY id ASC")
        return [dict(row) for row in cursor.fetchall()]

    def get_llm_calls(self) -> List[Dict[str, Any]]:
        """Return all LLM telemetry records."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM llm_calls ORDER BY id ASC")
        return [dict(row) for row in cursor.fetchall()]

    def get_stats(self) -> Dict[str, Any]:
        """Aggregate summary counts across tables."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT record_type, COUNT(*) as cnt FROM records GROUP BY record_type")
        records_by_type = {row["record_type"]: row["cnt"] for row in cursor.fetchall()}
        total_records = sum(records_by_type.values())

        cursor.execute("SELECT COUNT(*) as cnt FROM raw_pages")
        raw_pages_count = cursor.fetchone()["cnt"]

        cursor.execute("SELECT COUNT(*) as cnt FROM entity_map")
        entity_maps_count = cursor.fetchone()["cnt"]

        cursor.execute("SELECT COUNT(*) as cnt FROM llm_calls")
        llm_calls_count = cursor.fetchone()["cnt"]

        cursor.execute("SELECT COUNT(*) as cnt FROM llm_calls WHERE success = 1")
        row_succ = cursor.fetchone()
        successful_llm_calls = row_succ["cnt"] if row_succ else 0

        cursor.execute("SELECT COUNT(*) as cnt FROM seen_items WHERE status = 'dropped'")
        row_drop = cursor.fetchone()
        drops_count = row_drop["cnt"] if row_drop else 0

        return {
            "records": records_by_type,
            "records_by_type": records_by_type,
            "total_records": total_records,
            "raw_pages": raw_pages_count,
            "total_raw_pages": raw_pages_count,
            "entity_mappings": entity_maps_count,
            "total_entity_mappings": entity_maps_count,
            "llm_calls": llm_calls_count,
            "total_llm_calls": llm_calls_count,
            "successful_llm_calls": successful_llm_calls,
            "drops": drops_count,
            "total_drops": drops_count,
        }

    def close(self) -> None:
        """Close SQLite database connection."""
        self.conn.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
