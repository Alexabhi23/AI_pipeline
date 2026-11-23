"""GraphOne / FrontierAtlas - AI Engineer Demo Pipeline.

Main entrypoint and pipeline orchestrator.
Executes the production async ingestion pipeline:
  discover -> fetch -> extract -> ground-check -> resolve -> dedup -> store -> export
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import structlog

from src.config import Settings, get_settings
from src.crawler.dates import compute_content_hash, compute_url_hash, extract_published, is_fresh
from src.crawler.http import HttpClient
from src.crawler.sources import GitHubStarsEnricher, get_adapter
from src.crawler.sources.base import SourceConfig, load_sources
from src.export import SheetsExporter, export_csv
from src.llm import LLMOrchestrator
from src.models import (
    BaseRecord,
    DateConfidence,
    DateSource,
    Job,
    News,
    PricingModel,
    Product,
    RecordType,
    ResearchPaper,
    Startup,
)
from src.resolver import EntityResolver
from src.storage import Database, DeduplicationStore, create_dedup_store

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger("pipeline.main")


class PipelineRunner:
    """Async pipeline orchestrator executing the ingestion flow."""

    def __init__(self, settings: Settings, phase: str = "all", export_format: str = "both") -> None:
        self.settings = settings
        self.phase = phase.lower()
        self.export_format = export_format.lower()
        self.http_client: Optional[HttpClient] = None
        self.dedup_store: Optional[DeduplicationStore] = None
        self.database: Optional[Database] = None
        self.orchestrator: Optional[LLMOrchestrator] = None
        self.resolver: Optional[EntityResolver] = None
        self.stars_enricher: Optional[GitHubStarsEnricher] = None
        self.shutdown_event = asyncio.Event()

        # Telemetry & stats tracking
        self.stats = {
            "start_time": datetime.now(timezone.utc).isoformat(),
            "sources_processed": 0,
            "items_discovered": 0,
            "items_fetched": 0,
            "items_deduped": 0,
            "items_dropped": 0,
            "drop_reasons": {},
            "records_stored": {
                RecordType.STARTUP.value: 0,
                RecordType.PRODUCT.value: 0,
                RecordType.PAPER.value: 0,
                RecordType.JOB.value: 0,
                RecordType.NEWS.value: 0,
            },
            "llm_fallbacks_used": 0,
            "errors": [],
        }

    async def initialize(self) -> None:
        """Initialize all subsystems and connections."""
        logger.info("Initializing pipeline subsystems", phase=self.phase)
        self.http_client = HttpClient(self.settings)
        self.dedup_store = create_dedup_store(self.settings)
        self.database = Database(self.settings.SQLITE_DB_PATH)
        self.orchestrator = LLMOrchestrator(self.settings)
        self.resolver = EntityResolver()
        self.stars_enricher = GitHubStarsEnricher(self.http_client)

    async def shutdown(self) -> None:
        """Gracefully release all network connections and database handles."""
        logger.info("Shutting down pipeline subsystems...")
        if self.http_client:
            await self.http_client.close()
        if self.dedup_store:
            await self.dedup_store.close()
        if self.database:
            self.database.close()
        logger.info("All subsystems closed gracefully.")

    def record_drop(self, reason: str) -> None:
        """Increment drop metrics with structured reason."""
        self.stats["items_dropped"] += 1
        self.stats["drop_reasons"][reason] = self.stats["drop_reasons"].get(reason, 0) + 1

    def matches_phase(self, source_name: str, adapter_name: str) -> bool:
        """Filter sources based on the requested phase CLI option."""
        if self.phase == "all":
            return True
        mapping = {
            "papers": ["arxiv", "hf_papers"],
            "startups": ["yc"],
            "products": ["producthunt"],
            "news": ["rss_news"],
            "jobs": ["jobs"],
        }
        allowed_adapters = mapping.get(self.phase, [])
        return adapter_name in allowed_adapters or self.phase in source_name

    async def process_raw_item(self, raw_item: Dict[str, Any], source_cfg: SourceConfig) -> None:
        """Process a single raw crawler item through extraction, validation, and storage."""
        url = raw_item.get("url") or raw_item.get("link")
        if not url:
            self.record_drop("missing_source_url")
            return

        # 1. URL and Content hashing
        url_hash = compute_url_hash(url)
        content_text = raw_item.get("content") or raw_item.get("text") or raw_item.get("description") or str(raw_item)
        content_hash = compute_content_hash(content_text)

        # 2. Cross-node / Cross-run Dedup
        is_new = await self.dedup_store.is_new(
            url_hash=url_hash,
            content_hash=content_hash,
            source=source_cfg.name,
            record_type=source_cfg.adapter,
        )
        if not is_new:
            self.stats["items_deduped"] += 1
            return

        # 3. Date extraction and provenance
        published_at = raw_item.get("published_at")
        date_src = raw_item.get("date_source")
        date_conf = DateConfidence.HIGH

        if not published_at:
            extracted_dt, ds = extract_published(
                html=raw_item.get("html", ""),
                headers=raw_item.get("headers", {}),
                url=url,
            )
            published_at = extracted_dt
            date_src = ds

        if not published_at:
            # Heuristic fallback: first seen
            published_at = datetime.now(timezone.utc)
            date_src = DateSource.HEURISTIC
            date_conf = DateConfidence.LOW

        # 4. Freshness filter for News and Jobs (Strict 24h constraint)
        if source_cfg.adapter in ("rss_news", "jobs"):
            if not is_fresh(published_at, hours=self.settings.FRESHNESS_HOURS):
                self.record_drop("stale_over_24h")
                return

        # 5. Extract structured record via LLM Orchestrator
        target_type = self._determine_record_type(source_cfg.adapter)
        extracted, llm_log = await self.orchestrator.extract(
            raw_data=raw_item,
            record_type=target_type,
            source_text=content_text,
        )

        if llm_log:
            self.database.save_llm_call(llm_log)

        if not extracted:
            self.record_drop("extraction_or_grounding_failed")
            return

        # 6. Entity Resolution
        if target_type == RecordType.STARTUP:
            canonical, method, score = self.resolver.resolve(extracted.get("name", ""))
            extracted["canonicalName"] = canonical
            extracted["entityResolutionMethod"] = method
            extracted["entityResolutionScore"] = score
        elif target_type == RecordType.PRODUCT:
            company_raw = extracted.get("company")
            if company_raw:
                canonical, method, score = self.resolver.resolve(company_raw)
                extracted["canonicalCompany"] = canonical

        # 7. GitHub Stars live enrichment for Papers
        if target_type == RecordType.PAPER:
            gh_url = extracted.get("githubUrl") or raw_item.get("github_url")
            if gh_url:
                stars = await self.stars_enricher.get_stars(gh_url)
                if stars is not None:
                    extracted["githubStars"] = stars
                    extracted["githubUrl"] = gh_url

        # 8. Instantiate Pydantic record model & validate
        record = self._build_record_model(
            target_type=target_type,
            data=extracted,
            url=url,
            url_hash=url_hash,
            content_hash=content_hash,
            date_source=date_src,
            date_confidence=date_conf,
            published_at=published_at,
        )

        if not record:
            self.record_drop("schema_validation_failed")
            return

        # 9. Store in SQLite DB and mark seen
        self.database.save_record(record)
        await self.dedup_store.mark_seen(
            url_hash=url_hash,
            content_hash=content_hash,
            source=source_cfg.name,
            record_type=target_type.value,
        )
        self.stats["records_stored"][target_type.value] += 1

    def _determine_record_type(self, adapter_name: str) -> RecordType:
        if adapter_name in ("arxiv", "hf_papers"):
            return RecordType.PAPER
        elif adapter_name == "yc":
            return RecordType.STARTUP
        elif adapter_name == "producthunt":
            return RecordType.PRODUCT
        elif adapter_name == "rss_news":
            return RecordType.NEWS
        elif adapter_name == "jobs":
            return RecordType.JOB
        return RecordType.STARTUP

    def _build_record_model(
        self,
        target_type: RecordType,
        data: Dict[str, Any],
        url: str,
        url_hash: str,
        content_hash: str,
        date_source: Any,
        date_confidence: Any,
        published_at: Optional[datetime],
    ) -> Optional[BaseRecord]:
        try:
            base_kwargs = {
                "sourceUrl": url,
                "urlHash": url_hash,
                "contentHash": content_hash,
                "dateSource": date_source,
                "dateConfidence": date_confidence,
            }
            if target_type == RecordType.STARTUP:
                return Startup(**base_kwargs, **data)
            elif target_type == RecordType.PRODUCT:
                return Product(**base_kwargs, **data)
            elif target_type == RecordType.PAPER:
                if published_at and not data.get("publishedAt"):
                    data["publishedAt"] = published_at
                return ResearchPaper(**base_kwargs, **data)
            elif target_type == RecordType.JOB:
                if published_at and not data.get("postedAt"):
                    data["postedAt"] = published_at
                return Job(**base_kwargs, **data)
            elif target_type == RecordType.NEWS:
                if published_at and not data.get("publishedAt"):
                    data["publishedAt"] = published_at
                return News(**base_kwargs, **data)
        except Exception as e:
            logger.warning("Pydantic validation failure", error=str(e), type=target_type.value)
            return None
        return None

    async def run(self) -> None:
        """Run the end-to-end extraction pipeline."""
        start_ts = time.time()
        logger.info("Starting GraphOne pipeline run", phase=self.phase)
        await self.initialize()

        sources = load_sources(self.settings.SOURCES_YAML_PATH)
        active_sources = [s for s in sources if s.enabled and self.matches_phase(s.name, s.adapter)]
        logger.info(f"Loaded {len(active_sources)} active sources for phase '{self.phase}'")

        # Process each source
        for source_cfg in active_sources:
            if self.shutdown_event.is_set():
                break

            logger.info("Processing source", source=source_cfg.name, adapter=source_cfg.adapter)
            try:
                adapter_cls = get_adapter(source_cfg.adapter)
                adapter = adapter_cls(source_cfg, self.http_client)
                raw_items = await adapter.run()
                self.stats["items_discovered"] += len(raw_items)
                logger.info(f"Source {source_cfg.name} discovered {len(raw_items)} items")

                for item in raw_items:
                    if self.shutdown_event.is_set():
                        break
                    await self.process_raw_item(item, source_cfg)

                self.stats["sources_processed"] += 1
            except Exception as e:
                logger.error("Error running source", source=source_cfg.name, error=str(e))
                self.stats["errors"].append({"source": source_cfg.name, "error": str(e)})

        # Save Entity Resolver mapping logs to DB
        for mapping in self.resolver.mapping_log:
            self.database.save_entity_mapping(mapping)

        # Export phase
        if self.export_format in ("csv", "both"):
            csv_paths = export_csv(self.database)
            logger.info("CSV export completed", paths=csv_paths)

        if self.export_format in ("sheets", "both"):
            sheets_exporter = SheetsExporter(self.settings)
            sheet_url = await sheets_exporter.export(self.database)
            if sheet_url:
                logger.info("Google Sheets export available at:", url=sheet_url)

        self.stats["end_time"] = datetime.now(timezone.utc).isoformat()
        self.stats["duration_seconds"] = round(time.time() - start_ts, 2)
        self.print_summary()
        await self.shutdown()

    def print_summary(self) -> None:
        """Print formatted terminal report of pipeline execution metrics."""
        print("\n" + "=" * 60)
        print("   GraphOne / FrontierAtlas Ingestion Run Report")
        print("=" * 60)
        print(f"Phase:                 {self.phase}")
        print(f"Duration:              {self.stats.get('duration_seconds')}s")
        print(f"Sources Processed:     {self.stats['sources_processed']}")
        print(f"Items Discovered:      {self.stats['items_discovered']}")
        print(f"Items Deduped:         {self.stats['items_deduped']}")
        print(f"Items Dropped:         {self.stats['items_dropped']}")
        print("-" * 60)
        print("Stored Records by Type:")
        for rtype, count in self.stats["records_stored"].items():
            print(f"  - {rtype.ljust(12)}: {count}")
        print("-" * 60)
        if self.stats["drop_reasons"]:
            print("Drop Reasons Breakdown:")
            for reason, count in self.stats["drop_reasons"].items():
                print(f"  - {reason.ljust(30)}: {count}")
        print("=" * 60 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GraphOne AI Ingestion Pipeline Runner")
    parser.add_argument(
        "--phase",
        choices=["all", "papers", "startups", "products", "news", "jobs"],
        default="all",
        help="Run full pipeline or a specific source phase",
    )
    parser.add_argument(
        "--export",
        choices=["csv", "sheets", "both", "none"],
        default="both",
        help="Export destination for collected data",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    settings = get_settings()
    runner = PipelineRunner(settings=settings, phase=args.phase, export_format=args.export)

    # Register graceful shutdown signals (handling Windows lack of SIGTERM/SIGQUIT)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT,):
        try:
            loop.add_signal_handler(sig, lambda: runner.shutdown_event.set())
        except (NotImplementedError, AttributeError):
            pass

    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
