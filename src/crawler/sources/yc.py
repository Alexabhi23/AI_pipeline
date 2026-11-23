"""Y Combinator Startup Directory adapter for the GraphOne AI data pipeline.

Crawls and parses Y Combinator's company directory for AI / Machine Learning
startups, extracting company names, descriptions, batches, headquarters,
team size, and websites.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

import structlog
try:
    from selectolax.lexbor import LexborHTMLParser as HTMLParser, LexborNode as Node
except ImportError:
    from selectolax.parser import HTMLParser, Node  # type: ignore[no-redef]

from src.crawler.sources.base import BaseSource, HttpClient, SourceConfig

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

YC_BASE_URL = "https://www.ycombinator.com"
COMPANY_SLUG_REGEX = re.compile(r"/companies/([a-zA-Z0-9_\-]+)")
BATCH_REGEX = re.compile(r"\b([WS]\d{2})\b", re.IGNORECASE)


class YCDirectorySource(BaseSource):
    """Adapter for scraping AI startups from Y Combinator directory."""

    def __init__(self, source_config: SourceConfig, http_client: HttpClient) -> None:
        super().__init__(source_config, http_client)
        self._log = logger.bind(
            source=source_config.name,
            adapter=source_config.adapter,
        )

    async def discover(self) -> list[str]:
        """Discover YC AI companies directory URLs.

        Returns:
            list[str]: Filtered directory URLs for AI companies.
        """
        cfg = self.adapter_config
        base_url = cfg.get("url", "https://www.ycombinator.com/companies")

        # Add AI industry filter if not already present
        if "industry=" not in base_url and "?" not in base_url:
            target_url = f"{base_url}?industry=Artificial%20Intelligence"
        else:
            target_url = base_url

        self._log.info("yc_directory_discovered", url=target_url)
        return [target_url]

    async def fetch_and_parse(self, url: str) -> list[dict[str, Any]]:
        """Fetch and parse YC company directory page.

        Supports extracting structured JSON from Next.js SSR hydration data
        (__NEXT_DATA__) as well as selectolax DOM fallback.

        Args:
            url: The directory URL.

        Returns:
            list[dict[str, Any]]: Parsed Startup raw dicts.
        """
        self._log.info("yc_directory_fetching", url=url)
        html_text = await self.http_client.get_text(url)
        if not html_text or not html_text.strip():
            self._log.warning("yc_directory_empty_response", url=url)
            return []

        tree = HTMLParser(html_text)

        # 1. Try Next.js __NEXT_DATA__ extraction (most complete & reliable)
        records = self._parse_from_next_data(tree)
        if records:
            self._log.info("yc_parsed_from_next_data", count=len(records))
            return records

        # 2. Fall back to selectolax DOM tree parsing
        records = self._parse_from_dom(tree)
        self._log.info("yc_parsed_from_dom", count=len(records))
        return records

    def _parse_from_next_data(self, tree: HTMLParser) -> list[dict[str, Any]]:
        """Extract structured company records from Next.js __NEXT_DATA__ script."""
        script_node = tree.css_first('script#__NEXT_DATA__[type="application/json"]')
        if script_node is None or not script_node.text():
            return []

        try:
            payload = json.loads(script_node.text())
        except Exception as exc:
            self._log.debug("yc_next_data_json_parse_failed", error=str(exc))
            return []

        page_props = payload.get("props", {}).get("pageProps", {})
        companies_raw = (
            page_props.get("companies")
            or page_props.get("initialCompanies")
            or page_props.get("results")
            or []
        )

        if not isinstance(companies_raw, list):
            return []

        records: list[dict[str, Any]] = []
        seen_slugs: set[str] = set()

        for c in companies_raw:
            if not isinstance(c, dict):
                continue

            slug = c.get("slug") or c.get("id")
            name = c.get("name")
            if not slug or not name or slug in seen_slugs:
                continue
            seen_slugs.add(slug)

            one_liner = c.get("one_liner") or c.get("oneLiner") or ""
            long_desc = c.get("long_description") or c.get("longDescription") or ""
            description = one_liner or long_desc

            batch = c.get("batch_name") or c.get("batch") or ""
            location = c.get("location") or c.get("all_locations") or ""
            website = c.get("website") or ""

            team_size_raw = c.get("team_size") or c.get("teamSize")
            employee_count = None
            if team_size_raw is not None:
                try:
                    employee_count = int(team_size_raw)
                except (ValueError, TypeError):
                    employee_count = None

            tags = c.get("tags") or c.get("industries") or []
            categories = ["AI", "Artificial Intelligence"]
            for tag in tags:
                if isinstance(tag, str) and tag not in categories:
                    categories.append(tag)

            yc_profile_url = f"{YC_BASE_URL}/companies/{slug}"
            source_url = website or yc_profile_url

            records.append({
                "id": str(slug),
                "name": str(name).strip(),
                "description": description.strip(),
                "founders": None,
                "foundedDate": batch if batch else None,
                "headquarters": location.strip() if location else None,
                "employeeCount": employee_count,
                "funding": f"YC {batch}" if batch else "YC",
                "pricingModel": None,
                "website": website if website else yc_profile_url,
                "categories": categories,
                "sourceUrl": yc_profile_url,
                "url": yc_profile_url,
                "recordType": "STARTUP",
                "dateSource": "meta",
                "dateConfidence": "high",
                "source": self.name,
            })

        return records

    def _parse_from_dom(self, tree: HTMLParser) -> list[dict[str, Any]]:
        """Fallback DOM parser for YC directory company cards."""
        records: list[dict[str, Any]] = []
        seen_slugs: set[str] = set()

        # Find all company anchor links
        links = tree.css('a[href*="/companies/"]')
        for link in links:
            href = link.attributes.get("href", "")
            match = COMPANY_SLUG_REGEX.search(href)
            if not match:
                continue

            slug = match.group(1).strip()
            # Ignore directory filter URLs like /companies?industry=...
            if slug in ("founders", "industry", "batch", "regions") or slug in seen_slugs:
                continue
            seen_slugs.add(slug)

            # Locate card
            card: Node = link
            for _ in range(4):
                parent = card.parent
                if parent is None:
                    break
                card = parent
                classes = card.attributes.get("class", "")
                if "company" in classes or "card" in classes:
                    break

            # Company name
            name_node = card.css_first("span[class*='name'], div[class*='name'], h3, h4")
            name = name_node.text(strip=True) if name_node and name_node.text() else link.text(strip=True)
            if not name or len(name) < 2:
                name = slug.replace("-", " ").title()

            # Description
            desc_node = card.css_first("span[class*='description'], div[class*='description'], p")
            description = desc_node.text(strip=True) if desc_node and desc_node.text() else ""

            # Batch
            batch = ""
            batch_node = card.css_first("span[class*='batch'], a[href*='/batch/']")
            if batch_node and batch_node.text():
                b_match = BATCH_REGEX.search(batch_node.text(strip=True))
                if b_match:
                    batch = b_match.group(1).upper()
            if not batch:
                full_text = card.text() or ""
                b_match = BATCH_REGEX.search(full_text)
                if b_match:
                    batch = b_match.group(1).upper()

            # Location
            location = ""
            loc_node = card.css_first("span[class*='location']")
            if loc_node and loc_node.text():
                location = loc_node.text(strip=True)

            yc_url = f"{YC_BASE_URL}/companies/{slug}"

            records.append({
                "id": slug,
                "name": name,
                "description": description,
                "founders": None,
                "foundedDate": batch if batch else None,
                "headquarters": location if location else None,
                "employeeCount": None,
                "funding": f"YC {batch}" if batch else "YC",
                "pricingModel": None,
                "website": yc_url,
                "categories": ["AI", "Artificial Intelligence", "Startups"],
                "sourceUrl": yc_url,
                "url": yc_url,
                "recordType": "STARTUP",
                "dateSource": "meta",
                "dateConfidence": "high",
                "source": self.name,
            })

        return records
