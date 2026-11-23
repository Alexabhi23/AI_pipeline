"""CSV Exporter for GraphOne / FrontierAtlas Intelligence Pipeline.

Dumps records from the pipeline database into formatted CSV files:
- startups.csv
- products.csv
- research_papers.csv
- jobs.csv
- news.csv
- entity_mapping_log.csv
"""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Any, List, Optional

import structlog

from src.models import RecordType

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

HEADERS_STARTUPS = [
    "name",
    "canonical_name",
    "description",
    "founders",
    "founded_date",
    "headquarters",
    "employee_count",
    "funding",
    "pricing_model",
    "website",
    "categories",
    "entity_resolution_method",
    "entity_resolution_score",
    "source_url",
    "date_source",
    "date_confidence",
    "collected_at",
]

HEADERS_PRODUCTS = [
    "name",
    "company",
    "canonical_company",
    "description",
    "launch_date",
    "pricing_model",
    "website",
    "categories",
    "upvotes",
    "source_url",
    "date_source",
    "date_confidence",
    "collected_at",
]

HEADERS_PAPERS = [
    "title",
    "authors",
    "abstract",
    "published_at",
    "arxiv_id",
    "github_url",
    "github_stars",
    "categories",
    "paper_url",
    "source_url",
    "collected_at",
]

HEADERS_JOBS = [
    "title",
    "company",
    "location",
    "remote",
    "salary_min",
    "salary_max",
    "currency",
    "posted_at",
    "apply_url",
    "categories",
    "source",
    "source_url",
    "collected_at",
]

HEADERS_NEWS = [
    "title",
    "summary",
    "author",
    "published_at",
    "source",
    "categories",
    "sentiment",
    "source_url",
    "collected_at",
]

HEADERS_MAPPINGS = [
    "raw_name",
    "canonical_name",
    "method",
    "score",
    "created_at",
]


def _format_csv_value(val: Any) -> str:
    """Format a value for clean CSV output."""
    if val is None:
        return ""
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (list, tuple)):
        return ", ".join(str(x) for x in val if x is not None)
    return str(val)


def _to_dict(obj: Any) -> dict[str, Any]:
    """Convert model or row to dict."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict") and callable(obj.dict):
        return obj.dict()
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return {}


def _get_records_for_csv(db: Any, record_type: str) -> list[dict[str, Any]]:
    """Query records by type from db object or sqlite database."""
    if hasattr(db, "get_records_by_type"):
        rows = db.get_records_by_type(record_type)
        return [_to_dict(r) for r in rows]
    if hasattr(db, "get_records"):
        rows = db.get_records(record_type)
        return [_to_dict(r) for r in rows]

    conn: Optional[sqlite3.Connection] = None
    close_after = False
    if hasattr(db, "conn") and isinstance(db.conn, sqlite3.Connection):
        conn = db.conn
    elif hasattr(db, "connection") and isinstance(db.connection, sqlite3.Connection):
        conn = db.connection
    elif hasattr(db, "db_path") and Path(str(db.db_path)).exists():
        conn = sqlite3.connect(str(db.db_path))
        close_after = True
    elif isinstance(db, (str, Path)) and Path(str(db)).exists():
        conn = sqlite3.connect(str(db))
        close_after = True

    if not conn:
        return []

    conn.row_factory = sqlite3.Row
    results: list[dict[str, Any]] = []
    try:
        cur = conn.cursor()
        table_exists = cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='records'"
        ).fetchone()
        if table_exists:
            for row in cur.execute("SELECT * FROM records WHERE record_type = ?", (record_type,)):
                row_dict = dict(row)
                payload_json = row_dict.get("payload_json")
                if payload_json:
                    try:
                        payload = json.loads(payload_json)
                        if isinstance(payload, dict):
                            row_dict.update(payload)
                    except Exception:
                        pass
                results.append(row_dict)
    except Exception as e:
        logger.warning("csv_export.read_records_error", record_type=record_type, error=str(e))
    finally:
        if close_after:
            conn.close()

    return results


def _get_mappings_for_csv(db: Any) -> list[dict[str, Any]]:
    """Query entity mappings from db object or sqlite database."""
    if hasattr(db, "get_entity_mappings"):
        rows = db.get_entity_mappings()
        return [_to_dict(r) for r in rows]
    if hasattr(db, "mapping_log"):
        return [_to_dict(r) for r in db.mapping_log]

    conn: Optional[sqlite3.Connection] = None
    close_after = False
    if hasattr(db, "conn") and isinstance(db.conn, sqlite3.Connection):
        conn = db.conn
    elif hasattr(db, "connection") and isinstance(db.connection, sqlite3.Connection):
        conn = db.connection
    elif hasattr(db, "db_path") and Path(str(db.db_path)).exists():
        conn = sqlite3.connect(str(db.db_path))
        close_after = True
    elif isinstance(db, (str, Path)) and Path(str(db)).exists():
        conn = sqlite3.connect(str(db))
        close_after = True

    if not conn:
        return []

    conn.row_factory = sqlite3.Row
    results: list[dict[str, Any]] = []
    try:
        cur = conn.cursor()
        table_exists = cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='entity_mappings'"
        ).fetchone()
        if table_exists:
            for row in cur.execute("SELECT * FROM entity_mappings ORDER BY id ASC"):
                results.append(dict(row))
    except Exception as e:
        logger.warning("csv_export.read_mappings_error", error=str(e))
    finally:
        if close_after:
            conn.close()

    return results


def export_csv(db: Any, output_dir: str = "data/exports") -> dict[str, str]:
    """Query all records from database by type and write formatted CSV files.

    Outputs:
    - data/exports/startups.csv
    - data/exports/products.csv
    - data/exports/research_papers.csv
    - data/exports/jobs.csv
    - data/exports/news.csv
    - data/exports/entity_mapping_log.csv

    Returns dictionary mapping table name to absolute or relative file path.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    result_files: dict[str, str] = {}

    # 1. Startups
    startups = _get_records_for_csv(db, RecordType.STARTUP.value)
    startups_file = out_path / "startups.csv"
    with startups_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADERS_STARTUPS)
        for r in startups:
            writer.writerow(
                [
                    _format_csv_value(r.get("name")),
                    _format_csv_value(r.get("canonicalName") or r.get("canonical_name")),
                    _format_csv_value(r.get("description")),
                    _format_csv_value(r.get("founders")),
                    _format_csv_value(r.get("foundedDate") or r.get("founded_date")),
                    _format_csv_value(r.get("headquarters")),
                    _format_csv_value(r.get("employeeCount") or r.get("employee_count")),
                    _format_csv_value(r.get("funding")),
                    _format_csv_value(r.get("pricingModel") or r.get("pricing_model")),
                    _format_csv_value(r.get("website")),
                    _format_csv_value(r.get("categories")),
                    _format_csv_value(r.get("entityResolutionMethod") or r.get("entity_resolution_method")),
                    _format_csv_value(r.get("entityResolutionScore") or r.get("entity_resolution_score")),
                    _format_csv_value(r.get("sourceUrl") or r.get("source_url")),
                    _format_csv_value(r.get("dateSource") or r.get("date_source")),
                    _format_csv_value(r.get("dateConfidence") or r.get("date_confidence")),
                    _format_csv_value(r.get("collectedAt") or r.get("collected_at")),
                ]
            )
    result_files["startups"] = str(startups_file)
    logger.info("csv_export.startups_written", path=str(startups_file), count=len(startups))

    # 2. Products
    products = _get_records_for_csv(db, RecordType.PRODUCT.value)
    products_file = out_path / "products.csv"
    with products_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADERS_PRODUCTS)
        for r in products:
            writer.writerow(
                [
                    _format_csv_value(r.get("name")),
                    _format_csv_value(r.get("company")),
                    _format_csv_value(r.get("canonicalCompany") or r.get("canonical_company")),
                    _format_csv_value(r.get("description")),
                    _format_csv_value(r.get("launchDate") or r.get("launch_date")),
                    _format_csv_value(r.get("pricingModel") or r.get("pricing_model")),
                    _format_csv_value(r.get("website")),
                    _format_csv_value(r.get("categories")),
                    _format_csv_value(r.get("upvotes")),
                    _format_csv_value(r.get("sourceUrl") or r.get("source_url")),
                    _format_csv_value(r.get("dateSource") or r.get("date_source")),
                    _format_csv_value(r.get("dateConfidence") or r.get("date_confidence")),
                    _format_csv_value(r.get("collectedAt") or r.get("collected_at")),
                ]
            )
    result_files["products"] = str(products_file)
    logger.info("csv_export.products_written", path=str(products_file), count=len(products))

    # 3. Research Papers
    papers = _get_records_for_csv(db, RecordType.PAPER.value)
    papers_file = out_path / "research_papers.csv"
    with papers_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADERS_PAPERS)
        for r in papers:
            writer.writerow(
                [
                    _format_csv_value(r.get("title")),
                    _format_csv_value(r.get("authors")),
                    _format_csv_value(r.get("abstract")),
                    _format_csv_value(r.get("publishedAt") or r.get("published_at")),
                    _format_csv_value(r.get("arxivId") or r.get("arxiv_id")),
                    _format_csv_value(r.get("githubUrl") or r.get("github_url")),
                    _format_csv_value(r.get("githubStars") or r.get("github_stars")),
                    _format_csv_value(r.get("categories")),
                    _format_csv_value(r.get("paperUrl") or r.get("paper_url")),
                    _format_csv_value(r.get("sourceUrl") or r.get("source_url")),
                    _format_csv_value(r.get("collectedAt") or r.get("collected_at")),
                ]
            )
    result_files["research_papers"] = str(papers_file)
    logger.info("csv_export.research_papers_written", path=str(papers_file), count=len(papers))

    # 4. Jobs
    jobs = _get_records_for_csv(db, RecordType.JOB.value)
    jobs_file = out_path / "jobs.csv"
    with jobs_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADERS_JOBS)
        for r in jobs:
            writer.writerow(
                [
                    _format_csv_value(r.get("title")),
                    _format_csv_value(r.get("company")),
                    _format_csv_value(r.get("location")),
                    _format_csv_value(r.get("remote")),
                    _format_csv_value(r.get("salaryMin") or r.get("salary_min")),
                    _format_csv_value(r.get("salaryMax") or r.get("salary_max")),
                    _format_csv_value(r.get("currency")),
                    _format_csv_value(r.get("postedAt") or r.get("posted_at")),
                    _format_csv_value(r.get("applyUrl") or r.get("apply_url")),
                    _format_csv_value(r.get("categories")),
                    _format_csv_value(r.get("source")),
                    _format_csv_value(r.get("sourceUrl") or r.get("source_url")),
                    _format_csv_value(r.get("collectedAt") or r.get("collected_at")),
                ]
            )
    result_files["jobs"] = str(jobs_file)
    logger.info("csv_export.jobs_written", path=str(jobs_file), count=len(jobs))

    # 5. News
    news = _get_records_for_csv(db, RecordType.NEWS.value)
    news_file = out_path / "news.csv"
    with news_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADERS_NEWS)
        for r in news:
            writer.writerow(
                [
                    _format_csv_value(r.get("title")),
                    _format_csv_value(r.get("summary")),
                    _format_csv_value(r.get("author")),
                    _format_csv_value(r.get("publishedAt") or r.get("published_at")),
                    _format_csv_value(r.get("source")),
                    _format_csv_value(r.get("categories")),
                    _format_csv_value(r.get("sentiment")),
                    _format_csv_value(r.get("sourceUrl") or r.get("source_url")),
                    _format_csv_value(r.get("collectedAt") or r.get("collected_at")),
                ]
            )
    result_files["news"] = str(news_file)
    logger.info("csv_export.news_written", path=str(news_file), count=len(news))

    # 6. Entity Mapping Log
    mappings = _get_mappings_for_csv(db)
    mappings_file = out_path / "entity_mapping_log.csv"
    with mappings_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADERS_MAPPINGS)
        for m in mappings:
            writer.writerow(
                [
                    _format_csv_value(m.get("rawName") or m.get("raw_name")),
                    _format_csv_value(m.get("canonicalName") or m.get("canonical_name")),
                    _format_csv_value(m.get("method")),
                    _format_csv_value(m.get("score")),
                    _format_csv_value(m.get("createdAt") or m.get("created_at")),
                ]
            )
    result_files["entity_mapping_log"] = str(mappings_file)
    logger.info("csv_export.entity_mapping_log_written", path=str(mappings_file), count=len(mappings))

    return result_files
