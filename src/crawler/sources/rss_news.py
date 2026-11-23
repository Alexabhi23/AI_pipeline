"""Generic RSS / Atom news source adapter for the GraphOne AI data pipeline.

Handles syndicated feeds (TechCrunch, VentureBeat, The Verge, MIT Tech Review,
Ars Technica), parses item metadata, normalizes publication timestamps to ISO-8601 UTC,
and enforces a strict 24-hour freshness filter.
"""

from __future__ import annotations

import html
import re
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
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age_seconds = (datetime.now(timezone.utc) - dt).total_seconds()
        return age_seconds <= hours * 3600


logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# Common XML namespaces across RSS/Atom feeds
NS_MAP = {
    "atom": "http://www.w3.org/2005/Atom",
    "dc": "http://purl.org/dc/elements/1.1/",
    "content": "http://purl.org/rss/1.0/modules/content/",
}

STRIP_HTML_TAGS = re.compile(r"<[^>]+>")


class RSSNewsSource(BaseSource):
    """Universal RSS 2.0 and Atom adapter for AI news feeds."""

    def __init__(self, source_config: SourceConfig, http_client: HttpClient) -> None:
        super().__init__(source_config, http_client)
        self._log = logger.bind(
            source=source_config.name,
            adapter=source_config.adapter,
        )

    async def discover(self) -> list[str]:
        """Discover the feed URL from source configuration.

        Returns:
            list[str]: Feed endpoint URL.
        """
        cfg = self.adapter_config
        feed_url = cfg.get("feed_url") or cfg.get("url")
        if not feed_url:
            self._log.error("rss_missing_feed_url")
            return []

        self._log.info("rss_feed_discovered", feed_url=feed_url)
        return [feed_url]

    async def fetch_and_parse(self, url: str) -> list[dict[str, Any]]:
        """Fetch and parse RSS or Atom XML feed into News items.

        Applies strict 24-hour freshness constraint: older articles are discarded.

        Args:
            url: RSS or Atom feed URL.

        Returns:
            list[dict[str, Any]]: Fresh News item dictionaries.
        """
        self._log.info("rss_fetching_feed", url=url)
        xml_text = await self.http_client.get_text(url)
        if not xml_text or not xml_text.strip():
            self._log.warning("rss_empty_feed", url=url)
            return []

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            self._log.exception("rss_xml_parse_error", url=url)
            return []

        # Determine feed title as default source name
        channel_title = self.name
        title_node = root.find("channel/title") or root.find("title") or root.find("{http://www.w3.org/2005/Atom}title")
        if title_node is not None and title_node.text:
            channel_title = title_node.text.strip()

        # Find items (RSS: channel/item, Atom: entry)
        items = root.findall(".//item")
        is_atom = False
        if not items:
            items = root.findall(".//{http://www.w3.org/2005/Atom}entry") or root.findall(".//entry")
            is_atom = True

        self._log.info("rss_items_found", total=len(items), is_atom=is_atom)

        records: list[dict[str, Any]] = []
        stale_count = 0

        for item in items:
            try:
                record, item_dt = self._parse_item(item, is_atom=is_atom, default_source=channel_title)
                if not record:
                    continue

                # Check strict 24-hour freshness filter
                if not is_fresh(item_dt, hours=24):
                    stale_count += 1
                    continue

                records.append(record)
            except Exception as exc:
                self._log.warning("rss_item_parse_failed", error=str(exc))
                continue

        self._log.info(
            "rss_parse_completed",
            total_items=len(items),
            fresh_records=len(records),
            stale_skipped=stale_count,
        )
        return records

    def _parse_item(self, item: ET.Element, is_atom: bool, default_source: str) -> tuple[dict[str, Any] | None, datetime | None]:
        """Parse single RSS <item> or Atom <entry>."""
        # 1. Title
        title_elem = self._find_first(item, ["title", "{http://www.w3.org/2005/Atom}title"])
        raw_title = title_elem.text if title_elem is not None and title_elem.text else ""
        title = html.unescape(raw_title).strip()
        if not title:
            return None, None

        # 2. Link
        link = ""
        if is_atom:
            link_elem = self._find_first(item, ["{http://www.w3.org/2005/Atom}link", "link"])
            if link_elem is not None:
                link = link_elem.attrib.get("href", "") or (link_elem.text or "").strip()
        else:
            link_elem = self._find_first(item, ["link", "{http://www.w3.org/2005/Atom}link"])
            if link_elem is not None:
                link = (link_elem.text or link_elem.attrib.get("href", "")).strip()

        if not link:
            # Fallback to guid/id if looks like URL
            guid_elem = self._find_first(item, ["guid", "id", "{http://www.w3.org/2005/Atom}id"])
            if guid_elem is not None and guid_elem.text and guid_elem.text.startswith("http"):
                link = guid_elem.text.strip()

        if not link:
            return None, None

        # 3. Description / Summary
        summary = ""
        desc_elem = self._find_first(
            item,
            [
                "{http://purl.org/rss/1.0/modules/content/}encoded",
                "description",
                "summary",
                "{http://www.w3.org/2005/Atom}summary",
                "{http://www.w3.org/2005/Atom}content",
            ],
        )
        if desc_elem is not None and desc_elem.text:
            cleaned_text = STRIP_HTML_TAGS.sub(" ", desc_elem.text)
            summary = re.sub(r"\s+", " ", html.unescape(cleaned_text)).strip()

        # 4. Publication date
        date_elem = self._find_first(
            item,
            [
                "pubDate",
                "published",
                "{http://www.w3.org/2005/Atom}published",
                "updated",
                "{http://www.w3.org/2005/Atom}updated",
                "{http://purl.org/dc/elements/1.1/}date",
            ],
        )
        date_str = date_elem.text.strip() if date_elem is not None and date_elem.text else ""

        published_dt = None
        if date_str:
            try:
                published_dt, _ = extract_published(date_str=date_str)
            except Exception:
                published_dt = None

        if published_dt is None:
            # Fallback to current time if no date tag was found
            published_dt = datetime.now(timezone.utc)

        published_iso = published_dt.isoformat()

        # 5. Author
        author = None
        author_elem = self._find_first(
            item,
            [
                "{http://purl.org/dc/elements/1.1/}creator",
                "author",
                "{http://www.w3.org/2005/Atom}author/{http://www.w3.org/2005/Atom}name",
                "{http://www.w3.org/2005/Atom}author",
            ],
        )
        if author_elem is not None and author_elem.text:
            author = html.unescape(author_elem.text).strip()

        # 6. Categories
        categories: list[str] = []
        for cat in item.findall("category") + item.findall("{http://www.w3.org/2005/Atom}category"):
            c_text = cat.attrib.get("term") or cat.text
            if c_text:
                c_clean = html.unescape(c_text).strip()
                if c_clean and c_clean not in categories:
                    categories.append(c_clean)

        record = {
            "id": link,
            "title": title,
            "summary": summary,
            "publishedAt": published_iso,
            "published_at": published_iso,
            "author": author,
            "source": default_source,
            "categories": categories,
            "sourceUrl": link,
            "url": link,
            "recordType": "NEWS",
            "dateSource": "meta",
            "dateConfidence": "high",
        }
        return record, published_dt

    def _find_first(self, elem: ET.Element, tags: list[str]) -> ET.Element | None:
        """Find the first matching child element among multiple tag names."""
        for tag in tags:
            found = elem.find(tag)
            if found is not None:
                return found
        return None
