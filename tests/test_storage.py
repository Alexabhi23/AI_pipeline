"""Unit tests for storage subsystem (Database, SqliteDedup, and store factory)."""

import asyncio
from datetime import datetime, timezone
import os
import tempfile
import pytest

from src.models import (
    DateConfidence,
    DateSource,
    EntityMapping,
    Job,
    LLMCallLog,
    News,
    PricingModel,
    Product,
    RecordType,
    ResearchPaper,
    Startup,
)
from src.storage.database import Database
from src.storage.seen import SqliteDedup, create_dedup_store


def test_database_crud():
    with tempfile.TemporaryDirectory() as td:
        temp_db_path = os.path.join(td, "test_pipeline.db")
        with Database(db_path=temp_db_path) as db:
            # 1. raw_pages
            db.save_raw_page(
                url_hash="hash_p1",
                url="https://example.com/p1",
                http_status=200,
                fetched_at=datetime.now(timezone.utc),
                html_path="/data/raw/p1.html",
            )

            # 2. records (Startup, Product, Job, Paper, News)
            s = Startup(
                name="CognitiveAI",
                sourceUrl="https://example.com/startups/cognitive",
                urlHash="hash_st1",
                contentHash="ch_st1",
                pricingModel=PricingModel.FREEMIUM,
                categories=["NLP", "Agents"],
                dateSource=DateSource.META,
                dateConfidence=DateConfidence.HIGH,
            )
            rec_id = db.save_record(s)
            assert rec_id > 0

            p = Product(
                name="SuperSearch",
                company="CognitiveAI",
                sourceUrl="https://example.com/products/supersearch",
                urlHash="hash_prod1",
                contentHash="ch_prod1",
                upvotes=140,
            )
            p_id = db.save_record(p)
            assert p_id > 0

            # Retrieve records
            startups = db.get_records(RecordType.STARTUP)
            assert len(startups) == 1
            assert startups[0]["record_type"] == "STARTUP"

            all_records = db.get_records()
            assert len(all_records) == 2

            # 3. entity_map
            mapping = EntityMapping(
                rawName="Cognitive AI Inc",
                canonicalName="CognitiveAI",
                method="exact",
                score=1.0,
            )
            m_id = db.save_entity_mapping(mapping)
            assert m_id > 0
            mappings = db.get_entity_mappings()
            assert len(mappings) == 1
            assert mappings[0]["raw_name"] == "Cognitive AI Inc"

            # 4. llm_calls
            call_log = LLMCallLog(
                tier="Groq Llama 3",
                statusCode=200,
                retries=0,
                tokensIn=350,
                latencyMs=142.8,
                success=True,
            )
            c_id = db.save_llm_call(call_log)
            assert c_id > 0
            calls = db.get_llm_calls()
            assert len(calls) == 1
            assert calls[0]["tier"] == "Groq Llama 3"

            # 5. seed_entities
            db.seed_initial_entities({"OpenAI": ["Open AI", "OAI"], "Anthropic": ["Anthropic PBC"]})

            # 6. get_stats
            stats = db.get_stats()
            assert stats["total_records"] == 2
            assert stats["records_by_type"]["STARTUP"] == 1
            assert stats["records_by_type"]["PRODUCT"] == 1
            assert stats["total_raw_pages"] == 1
            assert stats["total_llm_calls"] == 1
            assert stats["successful_llm_calls"] == 1
            assert stats["total_entity_mappings"] == 1


def test_sqlite_dedup():
    with tempfile.TemporaryDirectory() as td:
        temp_db_path = os.path.join(td, "test_pipeline.db")

        async def _run():
            store = SqliteDedup(db_path=temp_db_path)
            try:
                # Initially new
                is_new = await store.is_new("url_h1", "content_h1", "techcrunch", "STARTUP")
                assert is_new is True

                # Mark seen
                await store.mark_seen("url_h1", "content_h1", "techcrunch", "STARTUP")

                # Now should not be new
                is_new_again = await store.is_new("url_h1", "content_h1", "techcrunch", "STARTUP")
                assert is_new_again is False

                # If content hash changes, it should be detected as new/updated
                is_updated = await store.is_new("url_h1", "content_h2_modified", "techcrunch", "STARTUP")
                assert is_updated is True

                # Check first seen timestamp
                first_seen = await store.get_first_seen("url_h1")
                assert first_seen is not None
                assert first_seen.tzinfo is not None
            finally:
                await store.close()

        asyncio.run(_run())


def test_create_dedup_store_fallback():
    with tempfile.TemporaryDirectory() as td:
        temp_db_path = os.path.join(td, "test_pipeline.db")
        from src.config import Settings

        custom_settings = Settings(
            SQLITE_DB_PATH=temp_db_path,
            REDIS_URL="redis://127.0.0.1:9999/0",  # Unreachable port
        )
        store = create_dedup_store(custom_settings)
        try:
            assert isinstance(store, SqliteDedup)
        finally:
            asyncio.run(store.close())
