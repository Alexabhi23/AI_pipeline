"""Google Sheets automated export for GraphOne pipeline.

Creates/updates a 6-tab spreadsheet using gspread and google-auth:
- Startups
- Products
- Research Papers
- Jobs
- News
- Entity Mapping Log

Sets public read-only access and logs the shareable URL.
Handles missing credentials with a graceful warning and fallback instructions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from src.config import Settings
from src.models import RecordType
from src.storage.database import Database

logger = structlog.get_logger("export.sheets")

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


class SheetsExporter:
    """Exports structured SQLite intelligence directly into Google Sheets."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.creds_path = settings.GOOGLE_SHEETS_CREDS_PATH
        self.spreadsheet_id = settings.SPREADSHEET_ID

    async def export(self, db: Database) -> Optional[str]:
        """Export all data to Google Sheets tabs and return shareable link."""
        if not self.creds_path or not Path(self.creds_path).exists():
            logger.warning(
                "Google Sheets export skipped: credentials file not found or not specified. "
                "To enable Sheets export, set GOOGLE_SHEETS_CREDS_PATH in your .env file."
            )
            return None

        try:
            import gspread
            from google.oauth2.service_account import Credentials

            creds = Credentials.from_service_account_file(self.creds_path, scopes=SCOPES)
            client = gspread.authorize(creds)

            # Open existing or create new spreadsheet
            if self.spreadsheet_id:
                sheet = client.open_by_key(self.spreadsheet_id)
            else:
                sheet = client.create("GraphOne FrontierAtlas Intelligence")

            # Try to make public view
            try:
                sheet.share("", perm_type="anyone", role="reader")
            except Exception as share_err:
                logger.debug("Could not set public sharing permissions", error=str(share_err))

            # Export tabs
            self._export_startups(sheet, db)
            self._export_products(sheet, db)
            self._export_papers(sheet, db)
            self._export_jobs(sheet, db)
            self._export_news(sheet, db)
            self._export_mappings(sheet, db)

            url = sheet.url
            logger.info("Google Sheets export successful", url=url)
            print(f"\n[Google Sheets Export Live URL]: {url}\n")
            return url

        except Exception as e:
            logger.error("Failed to export data to Google Sheets", error=str(e))
            return None

    def _get_or_create_worksheet(self, sheet: Any, title: str) -> Any:
        try:
            ws = sheet.worksheet(title)
            ws.clear()
            return ws
        except Exception:
            return sheet.add_worksheet(title=title, rows=1000, cols=20)

    def _export_startups(self, sheet: Any, db: Database) -> None:
        ws = self._get_or_create_worksheet(sheet, "Startups")
        headers = [
            "Canonical Name",
            "Raw Name",
            "Website",
            "Description",
            "Founders",
            "Headquarters",
            "Employees",
            "Pricing Model",
            "Funding",
            "Categories",
            "Resolution Method",
            "Confidence Score",
            "Source URL",
            "Collected At",
        ]
        records = db.get_records(RecordType.STARTUP)
        rows = [
            [
                r["payload"].get("canonicalName", ""),
                r["payload"].get("name", ""),
                r["payload"].get("website", ""),
                r["payload"].get("description", ""),
                ", ".join(r["payload"].get("founders") or []),
                r["payload"].get("headquarters", ""),
                str(r["payload"].get("employeeCount", "")),
                r["payload"].get("pricingModel", ""),
                r["payload"].get("funding", ""),
                ", ".join(r["payload"].get("categories") or []),
                r["payload"].get("entityResolutionMethod", ""),
                str(r["payload"].get("entityResolutionScore", "")),
                r.get("sourceUrl", r.get("url", "")),
                r.get("collected_at", ""),
            ]
            for r in records
        ]
        ws.update([headers] + rows)
        ws.freeze(rows=1)

    def _export_products(self, sheet: Any, db: Database) -> None:
        ws = self._get_or_create_worksheet(sheet, "Products")
        headers = [
            "Product Name",
            "Canonical Company",
            "Company",
            "Website",
            "Description",
            "Pricing Model",
            "Upvotes",
            "Categories",
            "Source URL",
            "Collected At",
        ]
        records = db.get_records(RecordType.PRODUCT)
        rows = [
            [
                r["payload"].get("name", ""),
                r["payload"].get("canonicalCompany", ""),
                r["payload"].get("company", ""),
                r["payload"].get("website", ""),
                r["payload"].get("description", ""),
                r["payload"].get("pricingModel", ""),
                str(r["payload"].get("upvotes", "")),
                ", ".join(r["payload"].get("categories") or []),
                r.get("sourceUrl", r.get("url", "")),
                r.get("collected_at", ""),
            ]
            for r in records
        ]
        ws.update([headers] + rows)
        ws.freeze(rows=1)

    def _export_papers(self, sheet: Any, db: Database) -> None:
        ws = self._get_or_create_worksheet(sheet, "Research Papers")
        headers = [
            "Title",
            "Authors",
            "ArXiv ID",
            "GitHub URL",
            "GitHub Stars",
            "Published At",
            "Categories",
            "Abstract",
            "Source URL",
            "Collected At",
        ]
        records = db.get_records(RecordType.PAPER)
        rows = [
            [
                r["payload"].get("title", ""),
                ", ".join(r["payload"].get("authors") or []),
                r["payload"].get("arxivId", ""),
                r["payload"].get("githubUrl", ""),
                str(r["payload"].get("githubStars", "")),
                r["payload"].get("publishedAt", ""),
                ", ".join(r["payload"].get("categories") or []),
                r["payload"].get("abstract", "")[:300] if r["payload"].get("abstract") else "",
                r.get("sourceUrl", r.get("url", "")),
                r.get("collected_at", ""),
            ]
            for r in records
        ]
        ws.update([headers] + rows)
        ws.freeze(rows=1)

    def _export_jobs(self, sheet: Any, db: Database) -> None:
        ws = self._get_or_create_worksheet(sheet, "Jobs")
        headers = [
            "Job Title",
            "Company",
            "Location",
            "Remote",
            "Salary Min",
            "Salary Max",
            "Currency",
            "Posted At",
            "Apply URL",
            "Categories",
            "Source URL",
            "Collected At",
        ]
        records = db.get_records(RecordType.JOB)
        rows = [
            [
                r["payload"].get("title", ""),
                r["payload"].get("company", ""),
                r["payload"].get("location", ""),
                "Yes" if r["payload"].get("remote") else "No",
                str(r["payload"].get("salaryMin", "")),
                str(r["payload"].get("salaryMax", "")),
                r["payload"].get("currency", "USD"),
                r["payload"].get("postedAt", ""),
                r["payload"].get("applyUrl", ""),
                ", ".join(r["payload"].get("categories") or []),
                r.get("sourceUrl", r.get("url", "")),
                r.get("collected_at", ""),
            ]
            for r in records
        ]
        ws.update([headers] + rows)
        ws.freeze(rows=1)

    def _export_news(self, sheet: Any, db: Database) -> None:
        ws = self._get_or_create_worksheet(sheet, "News")
        headers = [
            "Headline",
            "Source",
            "Author",
            "Published At",
            "Summary",
            "Categories",
            "Sentiment",
            "Source URL",
            "Collected At",
        ]
        records = db.get_records(RecordType.NEWS)
        rows = [
            [
                r["payload"].get("title", ""),
                r["payload"].get("source", ""),
                r["payload"].get("author", ""),
                r["payload"].get("publishedAt", ""),
                r["payload"].get("summary", ""),
                ", ".join(r["payload"].get("categories") or []),
                r["payload"].get("sentiment", ""),
                r.get("sourceUrl", r.get("url", "")),
                r.get("collected_at", ""),
            ]
            for r in records
        ]
        ws.update([headers] + rows)
        ws.freeze(rows=1)

    def _export_mappings(self, sheet: Any, db: Database) -> None:
        ws = self._get_or_create_worksheet(sheet, "Entity Mapping Log")
        headers = ["Raw Name", "Canonical Name", "Method", "Score", "Created At"]
        mappings = db.get_entity_mappings()
        rows = [
            [
                m.get("raw_name", ""),
                m.get("canonical_name", ""),
                m.get("method", ""),
                str(m.get("score", "")),
                m.get("created_at", ""),
            ]
            for m in mappings
        ]
        ws.update([headers] + rows)
        ws.freeze(rows=1)
