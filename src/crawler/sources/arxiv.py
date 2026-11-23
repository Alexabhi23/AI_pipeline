"""arXiv source adapter for the GraphOne AI data pipeline.

Queries the official arXiv Atom API for AI research papers (cs.AI, cs.LG, cs.CL, cs.CV),
parses entries, extracts paper metadata and linked GitHub code repositories.
"""

from __future__ import annotations

import re
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any

import structlog

from src.crawler.sources.base import BaseSource, HttpClient, SourceConfig

try:
    from src.crawler.dates import extract_published, is_fresh
except ImportError:
    import dateparser

    def extract_published(
        html: str = "",
        headers: dict | None = None,
        url: str = "",
        date_str: str = "",
    ) -> tuple[datetime | None, str | None]:
        target = date_str or html
        if not target:
            return None, None
        parsed = dateparser.parse(target, settings={"TIMEZONE": "UTC", "RETURN_AS_TIMEZONE_AWARE": True})
        return parsed, "api" if date_str else "meta"

    def is_fresh(dt: datetime | str | None, hours: int = 24) -> bool:
        if dt is None:
            return False
        if isinstance(dt, str):
            parsed, _ = extract_published(date_str=dt)
            if not parsed:
                return False
            dt = parsed
        if getattr(dt, "tzinfo", None) is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age_seconds = (datetime.now(timezone.utc) - dt).total_seconds()
        return age_seconds <= hours * 3600


logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# Atom and arXiv XML namespaces
ATOM_NS = "http://www.w3.org/2005/Atom"
ARXIV_NS = "http://arxiv.org/schemas/atom"
NAMESPACES = {
    "atom": ATOM_NS,
    "arxiv": ARXIV_NS,
}

# Regex to detect GitHub repositories from abstract and comments
GITHUB_REPO_REGEX = re.compile(
    r"https?://(?:www\.)?github\.com/([a-zA-Z0-9_\-\.]+)/([a-zA-Z0-9_\-\.]+)",
    re.IGNORECASE,
)
ARXIV_ID_REGEX = re.compile(r"(\d{4}\.\d{4,5}(?:v\d+)?|[a-zA-Z\-]+/\d{7}(?:v\d+)?)")

# Non-repository GitHub paths to filter out
GITHUB_IGNORE_PATHS = {
    "about", "features", "topics", "collections", "trending", "events",
    "sponsors", "pricing", "login", "signup", "settings", "explore",
}


def _find_elem(node: ET.Element, *tags: str, namespaces: dict[str, str] | None = None) -> ET.Element | None:
    """Safely find the first matching child element without relying on truthiness of Element."""
    for tag in tags:
        elem = node.find(tag, namespaces) if namespaces else node.find(tag)
        if elem is not None:
            return elem
    return None


def _find_all(node: ET.Element, *tags: str, namespaces: dict[str, str] | None = None) -> list[ET.Element]:
    """Safely find all matching child elements."""
    for tag in tags:
        elems = node.findall(tag, namespaces) if namespaces else node.findall(tag)
        if elems:
            return elems
    return []


class ArxivSource(BaseSource):
    """Adapter for arXiv Atom API queries."""

    def __init__(self, source_config: SourceConfig, http_client: HttpClient) -> None:
        super().__init__(source_config, http_client)
        self._log = logger.bind(
            source=source_config.name,
            adapter=source_config.adapter,
        )

    async def discover(self) -> list[str]:
        """Construct the arXiv query URL based on configured AI categories and limit.

        Returns:
            list[str]: One or more arXiv query endpoint URLs.
        """
        cfg = self.adapter_config
        base_url = cfg.get("base_url", "http://export.arxiv.org/api/query")
        categories = cfg.get("categories", ["cs.AI", "cs.LG", "cs.CL", "cs.CV"])
        max_results = int(cfg.get("max_results", 500))
        sort_by = cfg.get("sort_by", "lastUpdatedDate")
        sort_order = cfg.get("sort_order", "descending")

        # Build category query: cat:cs.AI OR cat:cs.LG OR ...
        cat_query = " OR ".join(f"cat:{cat.strip()}" for cat in categories if cat.strip())
        if not cat_query:
            cat_query = "cat:cs.AI"

        params = {
            "search_query": cat_query,
            "sortBy": sort_by,
            "sortOrder": sort_order,
            "start": 0,
            "max_results": max_results,
        }

        query_url = f"{base_url}?{urllib.parse.urlencode(params, safe=':+')}"
        self._log.info(
            "arxiv_query_discovered",
            categories=categories,
            max_results=max_results,
            sort_by=sort_by,
            url=query_url,
        )
        return [query_url]

    async def fetch_and_parse(self, url: str) -> list[dict[str, Any]]:
        """Fetch XML feed from arXiv API and parse into structured paper records.

        Args:
            url: arXiv API query endpoint URL.

        Returns:
            list[dict[str, Any]]: Raw paper items ready for downstream pipeline.
        """
        self._log.info("arxiv_fetching_feed", url=url)
        xml_text = await self.http_client.get_text(url)
        if not xml_text or not xml_text.strip():
            self._log.warning("arxiv_empty_response", url=url)
            return []

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            self._log.exception("arxiv_xml_parse_error", url=url)
            return []

        entries = (
            root.findall("atom:entry", NAMESPACES)
            or root.findall("{http://www.w3.org/2005/Atom}entry")
            or root.findall("entry")
        )

        records: list[dict[str, Any]] = []

        for entry in entries:
            try:
                record = self._parse_entry(entry)
                if record:
                    records.append(record)
            except Exception as exc:
                self._log.warning("arxiv_entry_parse_failed", error=str(exc))
                continue

        self._log.info("arxiv_feed_parsed", entries_count=len(entries), valid_records=len(records))
        return records

    def _parse_entry(self, entry: ET.Element) -> dict[str, Any] | None:
        """Parse an individual <entry> element into a raw ResearchPaper dict."""
        # Entry ID / ArXiv ID
        id_elem = _find_elem(entry, "atom:id", "{http://www.w3.org/2005/Atom}id", "id", namespaces=NAMESPACES)
        raw_id_url = id_elem.text.strip() if id_elem is not None and id_elem.text else ""

        match = ARXIV_ID_REGEX.search(raw_id_url)
        if not match:
            return None
        arxiv_id = match.group(1)

        # Title
        title_elem = _find_elem(entry, "atom:title", "{http://www.w3.org/2005/Atom}title", "title", namespaces=NAMESPACES)
        raw_title = title_elem.text if title_elem is not None and title_elem.text else ""
        title = re.sub(r"\s+", " ", raw_title).strip()
        if not title:
            return None

        # Abstract / Summary
        summary_elem = _find_elem(entry, "atom:summary", "{http://www.w3.org/2005/Atom}summary", "summary", namespaces=NAMESPACES)
        raw_summary = summary_elem.text if summary_elem is not None and summary_elem.text else ""
        abstract = re.sub(r"\s+", " ", raw_summary).strip()

        # Authors
        authors: list[str] = []
        author_elems = _find_all(entry, "atom:author", "{http://www.w3.org/2005/Atom}author", "author", namespaces=NAMESPACES)
        for auth in author_elems:
            name_elem = _find_elem(auth, "atom:name", "{http://www.w3.org/2005/Atom}name", "name", namespaces=NAMESPACES)
            if name_elem is not None and name_elem.text:
                cleaned_name = name_elem.text.strip()
                if cleaned_name:
                    authors.append(cleaned_name)

        # Publication date
        published_elem = _find_elem(entry, "atom:published", "{http://www.w3.org/2005/Atom}published", "published", namespaces=NAMESPACES)
        published_str = published_elem.text.strip() if published_elem is not None and published_elem.text else ""

        updated_elem = _find_elem(entry, "atom:updated", "{http://www.w3.org/2005/Atom}updated", "updated", namespaces=NAMESPACES)
        updated_str = updated_elem.text.strip() if updated_elem is not None and updated_elem.text else ""

        published_iso = published_str or updated_str
        published_dt = None
        if published_iso:
            try:
                published_dt, _ = extract_published(date_str=published_iso)
            except Exception:
                published_dt = None

        if published_dt:
            published_iso = published_dt.isoformat()

        # Categories
        categories: list[str] = []
        cat_elems = _find_all(entry, "atom:category", "{http://www.w3.org/2005/Atom}category", "category", namespaces=NAMESPACES)
        for cat in cat_elems:
            term = cat.attrib.get("term")
            if term and term not in categories:
                categories.append(term)

        # Links (PDF, landing page)
        paper_url = f"https://arxiv.org/abs/{arxiv_id}"
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        link_elems = _find_all(entry, "atom:link", "{http://www.w3.org/2005/Atom}link", "link", namespaces=NAMESPACES)
        for link in link_elems:
            attrib = link.attrib
            href = attrib.get("href", "")
            if attrib.get("title") == "pdf" or attrib.get("type") == "application/pdf":
                pdf_url = href
            elif attrib.get("rel") == "alternate":
                paper_url = href

        # Extract GitHub repository link from abstract and comments
        comments_elem = _find_elem(entry, "arxiv:comment", "{http://arxiv.org/schemas/atom}comment", namespaces=NAMESPACES)
        comment_text = comments_elem.text if comments_elem is not None and comments_elem.text else ""
        combined_text = f"{abstract} {comment_text}"
        github_url = self._extract_github_url(combined_text)

        return {
            "id": arxiv_id,
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "published_at": published_iso,
            "publishedAt": published_iso,
            "arxiv_id": arxiv_id,
            "arxivId": arxiv_id,
            "pdf_url": pdf_url,
            "paperUrl": paper_url,
            "sourceUrl": paper_url,
            "url": paper_url,
            "github_url": github_url,
            "githubUrl": github_url,
            "categories": categories,
            "recordType": "PAPER",
            "dateSource": "api",
            "dateConfidence": "high",
            "source": self.name,
        }

    def _extract_github_url(self, text: str) -> str | None:
        """Scan text for the first valid GitHub repository URL."""
        if not text:
            return None

        matches = GITHUB_REPO_REGEX.findall(text)
        for owner, repo in matches:
            owner = owner.strip().rstrip("/")
            repo = repo.strip().rstrip("/").removesuffix(".git")
            if owner.lower() in GITHUB_IGNORE_PATHS or repo.lower() in GITHUB_IGNORE_PATHS:
                continue
            if len(owner) >= 1 and len(repo) >= 1:
                return f"https://github.com/{owner}/{repo}"

        return None
