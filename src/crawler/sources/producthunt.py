"""Product Hunt AI Products adapter for the GraphOne AI data pipeline.

Crawls and parses Product Hunt's Artificial Intelligence topic page, extracting
product titles, taglines, makers, vote counts, and external links.
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

PH_BASE_URL = "https://www.producthunt.com"
POST_SLUG_REGEX = re.compile(r"/posts/([a-zA-Z0-9_\-]+)")
UPVOTE_DIGITS_REGEX = re.compile(r"(\d+)")


class ProductHuntSource(BaseSource):
    """Adapter for scraping AI products from Product Hunt topic feeds."""

    def __init__(self, source_config: SourceConfig, http_client: HttpClient) -> None:
        super().__init__(source_config, http_client)
        self._log = logger.bind(
            source=source_config.name,
            adapter=source_config.adapter,
        )

    async def discover(self) -> list[str]:
        """Discover Product Hunt AI topic URLs.

        Returns:
            list[str]: URLs for the Product Hunt AI topic pages.
        """
        cfg = self.adapter_config
        target_url = cfg.get("url", "https://www.producthunt.com/topics/artificial-intelligence")
        self._log.info("producthunt_discovered", url=target_url)
        return [target_url]

    async def fetch_and_parse(self, url: str) -> list[dict[str, Any]]:
        """Fetch and parse Product Hunt topic page.

        Extracts products via Next.js Apollo state if available, falling back
        to selectolax DOM tree parsing.

        Args:
            url: The topic page URL.

        Returns:
            list[dict[str, Any]]: Parsed Product raw dicts.
        """
        self._log.info("producthunt_fetching", url=url)
        html_text = await self.http_client.get_text(url)
        if not html_text or not html_text.strip():
            self._log.warning("producthunt_empty_response", url=url)
            return []

        tree = HTMLParser(html_text)

        # 1. Try Next.js __NEXT_DATA__ Apollo state extraction
        records = self._parse_from_next_data(tree)
        if records:
            self._log.info("producthunt_parsed_from_next_data", count=len(records))
            return records

        # 2. Fall back to selectolax DOM parsing
        records = self._parse_from_dom(tree)
        self._log.info("producthunt_parsed_from_dom", count=len(records))
        return records

    def _parse_from_next_data(self, tree: HTMLParser) -> list[dict[str, Any]]:
        """Extract structured products from Next.js Apollo cache."""
        script_node = tree.css_first('script#__NEXT_DATA__[type="application/json"]')
        if script_node is None or not script_node.text():
            return []

        try:
            payload = json.loads(script_node.text())
        except Exception as exc:
            self._log.debug("producthunt_next_data_json_parse_failed", error=str(exc))
            return []

        props = payload.get("props", {}).get("pageProps", {})
        apollo_state = props.get("apolloState", {}) or props.get("__APOLLO_STATE__", {})

        records: list[dict[str, Any]] = []
        seen_slugs: set[str] = set()

        for key, entity in apollo_state.items():
            if not isinstance(entity, dict) or not key.startswith("Post:"):
                continue

            slug = entity.get("slug")
            name = entity.get("name")
            if not slug or not name or slug in seen_slugs:
                continue
            seen_slugs.add(slug)

            tagline = entity.get("tagline") or ""
            description = entity.get("description") or tagline
            votes_count = entity.get("votesCount") or entity.get("upvotes") or 0
            website = entity.get("website") or entity.get("url")
            created_at = entity.get("createdAt")

            # Extract maker or company if available
            maker = None
            user_ref = entity.get("user")
            if isinstance(user_ref, dict):
                user_id = user_ref.get("id") or user_ref.get("__ref")
                if user_id and user_id in apollo_state:
                    maker = apollo_state[user_id].get("name")

            ph_post_url = f"{PH_BASE_URL}/posts/{slug}"

            records.append({
                "id": str(slug),
                "name": str(name).strip(),
                "description": description.strip(),
                "company": maker or name,
                "launchDate": created_at,
                "pricingModel": None,
                "website": website or ph_post_url,
                "categories": ["AI", "Artificial Intelligence", "SaaS"],
                "upvotes": int(votes_count) if votes_count else 0,
                "sourceUrl": ph_post_url,
                "url": ph_post_url,
                "recordType": "PRODUCT",
                "dateSource": "heuristic",
                "dateConfidence": "high",
                "source": self.name,
            })

        return records

    def _parse_from_dom(self, tree: HTMLParser) -> list[dict[str, Any]]:
        """Fallback DOM parser for Product Hunt product cards."""
        records: list[dict[str, Any]] = []
        seen_slugs: set[str] = set()

        links = tree.css('a[href*="/posts/"]')
        for link in links:
            href = link.attributes.get("href", "")
            match = POST_SLUG_REGEX.search(href)
            if not match:
                continue

            slug = match.group(1).strip()
            if slug in ("reviews", "activity", "discussion") or slug in seen_slugs:
                continue
            seen_slugs.add(slug)

            # Locate product card container
            card: Node = link
            for _ in range(5):
                parent = card.parent
                if parent is None:
                    break
                card = parent
                classes = card.attributes.get("class", "")
                if "item" in classes or "post" in classes or "flex" in classes:
                    break

            # Product name
            name_node = card.css_first("h3, h4, [data-test*='post-name']")
            name = name_node.text(strip=True) if name_node and name_node.text() else link.text(strip=True)
            if not name or len(name) < 2:
                name = slug.replace("-", " ").title()

            # Tagline / Description
            tagline = ""
            tagline_node = card.css_first("p, [data-test*='post-tagline'], span[class*='tagline']")
            if tagline_node and tagline_node.text():
                tagline = tagline_node.text(strip=True)

            # Upvotes
            upvotes = 0
            vote_node = card.css_first("[data-test*='vote-button'], button")
            if vote_node and vote_node.text():
                digit_match = UPVOTE_DIGITS_REGEX.search(vote_node.text(strip=True))
                if digit_match:
                    try:
                        upvotes = int(digit_match.group(1))
                    except ValueError:
                        pass

            ph_post_url = f"{PH_BASE_URL}/posts/{slug}"

            records.append({
                "id": slug,
                "name": name,
                "description": tagline or name,
                "company": name,
                "launchDate": None,
                "pricingModel": None,
                "website": ph_post_url,
                "categories": ["AI", "Artificial Intelligence", "Product Hunt"],
                "upvotes": upvotes,
                "sourceUrl": ph_post_url,
                "url": ph_post_url,
                "recordType": "PRODUCT",
                "dateSource": "heuristic",
                "dateConfidence": "high",
                "source": self.name,
            })

        return records
