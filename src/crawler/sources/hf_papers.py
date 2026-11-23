"""Hugging Face Papers source adapter for the GraphOne AI data pipeline.

Crawls and parses daily curated research papers from https://huggingface.co/papers,
extracting paper titles, arXiv IDs, community upvote counts, thumbnails, and author lists.
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

try:
    from src.crawler.dates import extract_published, is_fresh
except ImportError:
    def extract_published(
        html: str = "",
        headers: dict | None = None,
        url: str = "",
        date_str: str = "",
    ) -> tuple[datetime | None, str | None]:
        return datetime.now(timezone.utc), "meta"

    def is_fresh(dt: datetime | str | None, hours: int = 24) -> bool:
        return True


logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

ARXIV_ID_PATTERN = re.compile(r"/papers/(\d{4}\.\d{4,5}(?:v\d+)?)")
UPVOTE_PATTERN = re.compile(r"^\s*(\d+)\s*$")


class HFPapersSource(BaseSource):
    """Adapter for scraping Hugging Face Daily Papers."""

    def __init__(self, source_config: SourceConfig, http_client: HttpClient) -> None:
        super().__init__(source_config, http_client)
        self._log = logger.bind(
            source=source_config.name,
            adapter=source_config.adapter,
        )

    async def discover(self) -> list[str]:
        """Discover target URLs for Hugging Face papers.

        Returns:
            list[str]: Hugging Face papers landing page URL.
        """
        cfg = self.adapter_config
        target_url = cfg.get("url", "https://huggingface.co/papers")
        self._log.info("hf_papers_discovered", url=target_url)
        return [target_url]

    async def fetch_and_parse(self, url: str) -> list[dict[str, Any]]:
        """Fetch and parse the Hugging Face papers page using selectolax.

        Args:
            url: The page URL to fetch.

        Returns:
            list[dict[str, Any]]: Parsed ResearchPaper raw dicts.
        """
        self._log.info("hf_papers_fetching", url=url)
        html_text = await self.http_client.get_text(url)
        if not html_text or not html_text.strip():
            self._log.warning("hf_papers_empty_response", url=url)
            return []

        tree = HTMLParser(html_text)

        # 1. Try extracting structured JSON data from data-props if present
        records = self._parse_from_props(tree)
        if records:
            self._log.info("hf_papers_parsed_from_props", count=len(records))
            return records

        # 2. Fall back to selectolax DOM parsing
        records = self._parse_from_dom(tree)
        self._log.info("hf_papers_parsed_from_dom", count=len(records))
        return records

    def _parse_from_props(self, tree: HTMLParser) -> list[dict[str, Any]]:
        """Attempt to extract paper entries from embedded data-props or script JSON."""
        records: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        for node in tree.css("div[data-props], main[data-props]"):
            raw_props = node.attributes.get("data-props")
            if not raw_props or "dailyPapers" not in raw_props:
                continue

            try:
                data = json.loads(raw_props)
            except Exception:
                continue

            papers_list = data.get("dailyPapers") or data.get("papers") or []
            if not isinstance(papers_list, list):
                continue

            for item in papers_list:
                paper_dict = item.get("paper") if isinstance(item, dict) and "paper" in item else item
                if not isinstance(paper_dict, dict):
                    continue

                arxiv_id = paper_dict.get("id") or paper_dict.get("arxivId")
                if not arxiv_id or arxiv_id in seen_ids:
                    continue
                seen_ids.add(arxiv_id)

                title = paper_dict.get("title", "").strip()
                summary = paper_dict.get("summary", "").strip()
                upvotes = paper_dict.get("upvotes") or paper_dict.get("numUpvotes", 0)
                thumbnail = paper_dict.get("thumbnail") or paper_dict.get("mediaUrl")

                authors_raw = paper_dict.get("authors") or []
                authors: list[str] = []
                for auth in authors_raw:
                    if isinstance(auth, dict):
                        name = auth.get("name") or auth.get("user")
                        if name:
                            authors.append(str(name).strip())
                    elif isinstance(auth, str):
                        authors.append(auth.strip())

                published_at = paper_dict.get("publishedAt") or datetime.now(timezone.utc).isoformat()
                hf_url = f"https://huggingface.co/papers/{arxiv_id}"
                arxiv_url = f"https://arxiv.org/abs/{arxiv_id}"

                records.append({
                    "id": arxiv_id,
                    "title": title or f"Paper {arxiv_id}",
                    "authors": authors,
                    "abstract": summary,
                    "published_at": published_at,
                    "publishedAt": published_at,
                    "arxiv_id": arxiv_id,
                    "arxivId": arxiv_id,
                    "upvotes": int(upvotes) if upvotes else 0,
                    "thumbnail": thumbnail,
                    "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                    "paperUrl": arxiv_url,
                    "sourceUrl": hf_url,
                    "url": hf_url,
                    "github_url": None,
                    "githubUrl": None,
                    "categories": ["cs.AI"],
                    "recordType": "PAPER",
                    "dateSource": "meta",
                    "dateConfidence": "high",
                    "source": self.name,
                })

        return records

    def _parse_from_dom(self, tree: HTMLParser) -> list[dict[str, Any]]:
        """Parse paper cards directly from the HTML DOM tree using selectolax."""
        records: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        # Find all link tags pointing to /papers/ID
        paper_links = tree.css('a[href*="/papers/"]')
        for link in paper_links:
            href = link.attributes.get("href", "")
            match = ARXIV_ID_PATTERN.search(href)
            if not match:
                continue

            arxiv_id = match.group(1)
            if arxiv_id in seen_ids:
                continue
            seen_ids.add(arxiv_id)

            # Locate card container (traverse up to article or parent div)
            card: Node = link
            for _ in range(5):
                parent = card.parent
                if parent is None:
                    break
                card = parent
                if card.tag in ("article", "div") and ("border" in card.attributes.get("class", "") or "card" in card.attributes.get("class", "")):
                    break

            # Title
            title = ""
            # Often title is in h3, h2, or the link's text
            h_node = card.css_first("h3, h2, h1")
            if h_node and h_node.text():
                title = h_node.text().strip()
            elif link.text():
                title = link.text().strip()

            title = re.sub(r"\s+", " ", title).strip()
            if not title or title.lower() in ("read paper", "view paper", "comments", "upvote"):
                title = f"Paper {arxiv_id}"

            # Upvotes
            upvotes = 0
            # Look for button or span with upvote counts
            for btn in card.css("button, div, span"):
                classes = btn.attributes.get("class", "")
                if "upvote" in classes or "leading-none" in classes:
                    text_match = UPVOTE_PATTERN.search(btn.text(strip=True))
                    if text_match:
                        try:
                            upvotes = int(text_match.group(1))
                            break
                        except ValueError:
                            pass

            # Thumbnail
            thumbnail = None
            img_node = card.css_first("img")
            if img_node:
                src = img_node.attributes.get("src")
                if src and not src.startswith("data:") and "avatar" not in src:
                    thumbnail = src

            # Authors
            authors: list[str] = []
            for author_node in card.css("span.author, a[href*='/author/'], img[alt]"):
                if author_node.tag == "img":
                    alt = author_node.attributes.get("alt", "").strip()
                    if alt and "avatar" not in alt.lower() and alt not in authors:
                        authors.append(alt)
                else:
                    name = author_node.text(strip=True)
                    if name and name not in authors and len(name) < 50:
                        authors.append(name)

            hf_url = f"https://huggingface.co/papers/{arxiv_id}"
            arxiv_url = f"https://arxiv.org/abs/{arxiv_id}"
            now_iso = datetime.now(timezone.utc).isoformat()

            records.append({
                "id": arxiv_id,
                "title": title,
                "authors": authors,
                "abstract": None,
                "published_at": now_iso,
                "publishedAt": now_iso,
                "arxiv_id": arxiv_id,
                "arxivId": arxiv_id,
                "upvotes": upvotes,
                "thumbnail": thumbnail,
                "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                "paperUrl": arxiv_url,
                "sourceUrl": hf_url,
                "url": hf_url,
                "github_url": None,
                "githubUrl": None,
                "categories": ["cs.AI"],
                "recordType": "PAPER",
                "dateSource": "heuristic",
                "dateConfidence": "high",
                "source": self.name,
            })

        return records
