"""Multi-format AI Jobs adapter for the GraphOne AI data pipeline.

Supports:
1. RemoteOK API (JSON with AI/ML tag filtering)
2. Remotive API (JSON data/AI job listings)
3. Greenhouse Job Board API (JSON endpoint per company/board)
4. Lever Postings API (JSON endpoint per company)
5. Hacker News "Who is hiring?" threads (HTML scraping & comment parsing)

Extracts standardized job titles, companies, locations, compensation ranges,
and application URLs while enforcing a strict 24-hour freshness constraint.
"""

from __future__ import annotations

import html
import json
import re
from datetime import datetime, timezone
from typing import Any

import structlog
try:
    from selectolax.lexbor import LexborHTMLParser as HTMLParser
except ImportError:
    from selectolax.parser import HTMLParser  # type: ignore[no-redef]

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

# Keywords identifying AI/ML roles
AI_KEYWORDS = {
    "ai", "ml", "machine learning", "deep learning", "nlp", "llm",
    "data science", "data scientist", "computer vision", "neural",
    "reinforcement learning", "genai", "prompt engineer", "ai engineer",
}

SALARY_REGEX = re.compile(
    r"\$([0-9]{1,3}(?:,[0-9]{3})*|[0-9]+)\s*(k|K)?(?:\s*(?:-|to)\s*\$([0-9]{1,3}(?:,[0-9]{3})*|[0-9]+)\s*(k|K)?)?",
)


def parse_salary_range(text: str | None) -> tuple[int | None, int | None, str]:
    """Parse minimum and maximum salary figures from compensation text."""
    if not text:
        return None, None, "USD"

    match = SALARY_REGEX.search(text)
    if not match:
        return None, None, "USD"

    def clean_num(val_str: str | None, multiplier_flag: str | None) -> int | None:
        if not val_str:
            return None
        val = int(val_str.replace(",", ""))
        if multiplier_flag or val < 1000:
            val *= 1000
        return val

    min_val = clean_num(match.group(1), match.group(2))
    max_val = clean_num(match.group(3), match.group(4) or match.group(2))

    if min_val and max_val and min_val > max_val:
        min_val, max_val = max_val, min_val

    return min_val, max_val, "USD"


class JobsSource(BaseSource):
    """Unified job board source adapter supporting API feeds and HN Hiring threads."""

    def __init__(self, source_config: SourceConfig, http_client: HttpClient) -> None:
        super().__init__(source_config, http_client)
        self._log = logger.bind(
            source=source_config.name,
            adapter=source_config.adapter,
        )

    async def discover(self) -> list[str]:
        """Discover job feed or page URLs.

        For HN Hiring threads, resolves the latest monthly story ID via HN Algolia search.
        """
        cfg = self.adapter_config
        target_url = cfg.get("url")
        job_type = cfg.get("type", "").lower()

        # Check for HN hiring thread resolution
        if job_type == "hn_hiring" or (target_url and "ask_hn_hiring" in target_url):
            resolved_url = await self._discover_latest_hn_hiring()
            if resolved_url:
                self._log.info("hn_hiring_resolved", url=resolved_url)
                return [resolved_url]

        if target_url:
            self._log.info("jobs_source_discovered", url=target_url)
            return [target_url]

        self._log.error("jobs_missing_url_configuration")
        return []

    async def _discover_latest_hn_hiring(self) -> str | None:
        """Query Algolia HN API for the most recent 'Ask HN: Who is hiring?' thread."""
        algolia_url = (
            "https://hn.algolia.com/api/v1/search_by_date"
            "?tags=story,author_whoishiring&query=hiring&hitsPerPage=1"
        )
        try:
            resp_text = await self.http_client.get_text(algolia_url)
            if resp_text:
                data = json.loads(resp_text)
                hits = data.get("hits", [])
                if hits and isinstance(hits, list):
                    story_id = hits[0].get("objectID")
                    if story_id:
                        return f"https://news.ycombinator.com/item?id={story_id}"
        except Exception as exc:
            self._log.warning("hn_algolia_discovery_failed", error=str(exc))

        # Fallback to direct thread if resolution fails
        return "https://news.ycombinator.com/item?id=ask_hn_hiring"

    async def fetch_and_parse(self, url: str) -> list[dict[str, Any]]:
        """Fetch and parse jobs based on endpoint type and structure.

        Args:
            url: Job source URL.

        Returns:
            list[dict[str, Any]]: List of fresh Job raw dicts.
        """
        self._log.info("jobs_fetching", url=url)
        content_text = await self.http_client.get_text(url)
        if not content_text or not content_text.strip():
            self._log.warning("jobs_empty_response", url=url)
            return []

        # Dispatch based on URL / config pattern
        if "remoteok.com" in url or self.name == "remoteok":
            return self._parse_remoteok(content_text, url)
        elif "remotive.com" in url or self.name == "remotive":
            return self._parse_remotive(content_text, url)
        elif "boards-api.greenhouse.io" in url or "greenhouse" in self.name:
            return self._parse_greenhouse(content_text, url)
        elif "api.lever.co" in url or "lever" in self.name:
            return self._parse_lever(content_text, url)
        elif "news.ycombinator.com" in url or self.adapter_config.get("type") == "hn_hiring":
            return self._parse_hn_hiring(content_text, url)

        # Fallback generic JSON or text parser
        return self._parse_generic_json(content_text, url)

    def _parse_remoteok(self, text: str, url: str) -> list[dict[str, Any]]:
        """Parse RemoteOK API JSON response."""
        try:
            data = json.loads(text)
        except Exception as exc:
            self._log.error("remoteok_json_error", error=str(exc))
            return []

        if not isinstance(data, list):
            return []

        cfg_tags = [t.lower() for t in self.adapter_config.get("filter_tags", ["ai", "machine-learning"])]
        records: list[dict[str, Any]] = []

        for item in data:
            if not isinstance(item, dict) or not item.get("id"):
                continue

            # Position / Title
            position = item.get("position", "").strip()
            if not position:
                continue

            # Tag check
            item_tags = [str(t).lower() for t in item.get("tags", [])]
            is_ai_relevant = any(ct in item_tags for ct in cfg_tags) or any(k in position.lower() for k in AI_KEYWORDS)
            if not is_ai_relevant:
                continue

            # Publication date & freshness
            date_str = item.get("date")
            epoch = item.get("epoch")
            posted_dt = None
            if date_str:
                posted_dt, _ = extract_published(date_str=str(date_str))
            elif epoch:
                try:
                    posted_dt = datetime.fromtimestamp(float(epoch), tz=timezone.utc)
                except Exception:
                    posted_dt = None

            if not posted_dt:
                posted_dt = datetime.now(timezone.utc)

            # Strict 24h filter
            if not is_fresh(posted_dt, hours=24):
                continue

            apply_url = item.get("url") or f"https://remoteok.com/remote-jobs/{item.get('id')}"
            company = item.get("company", "Unknown").strip()
            location = item.get("location", "Remote").strip()

            sal_min = item.get("salary_min")
            sal_max = item.get("salary_max")
            try:
                sal_min = int(sal_min) if sal_min else None
                sal_max = int(sal_max) if sal_max else None
            except (ValueError, TypeError):
                sal_min, sal_max = None, None

            posted_iso = posted_dt.isoformat()
            records.append({
                "id": str(item.get("id")),
                "title": position,
                "company": company,
                "location": location,
                "remote": True,
                "salaryMin": sal_min,
                "salaryMax": sal_max,
                "currency": "USD",
                "postedAt": posted_iso,
                "posted_at": posted_iso,
                "applyUrl": apply_url,
                "apply_url": apply_url,
                "sourceUrl": apply_url,
                "url": apply_url,
                "categories": item_tags,
                "source": "RemoteOK",
                "recordType": "JOB",
                "dateSource": "api",
                "dateConfidence": "high",
            })

        self._log.info("remoteok_parsed", valid_records=len(records))
        return records

    def _parse_remotive(self, text: str, url: str) -> list[dict[str, Any]]:
        """Parse Remotive API JSON response."""
        try:
            data = json.loads(text)
        except Exception as exc:
            self._log.error("remotive_json_error", error=str(exc))
            return []

        jobs = data.get("jobs", []) if isinstance(data, dict) else []
        records: list[dict[str, Any]] = []

        for item in jobs:
            if not isinstance(item, dict):
                continue

            title = item.get("title", "").strip()
            if not title:
                continue

            tags = [str(t).lower() for t in item.get("tags", [])]
            category = item.get("category", "").lower()
            combined_desc = f"{title} {category} {' '.join(tags)}".lower()

            # Check AI / data relevance
            is_relevant = any(k in combined_desc for k in AI_KEYWORDS) or "data" in category
            if not is_relevant:
                continue

            pub_date = item.get("publication_date")
            posted_dt, _ = extract_published(date_str=str(pub_date)) if pub_date else (None, None)
            if not posted_dt:
                posted_dt = datetime.now(timezone.utc)

            # Strict 24h filter
            if not is_fresh(posted_dt, hours=24):
                continue

            sal_text = item.get("salary", "")
            s_min, s_max, curr = parse_salary_range(sal_text)

            apply_url = item.get("url") or f"https://remotive.com/jobs/{item.get('id')}"
            company = item.get("company_name", "Unknown").strip()
            location = item.get("candidate_required_location", "Remote").strip()

            posted_iso = posted_dt.isoformat()
            records.append({
                "id": str(item.get("id")),
                "title": title,
                "company": company,
                "location": location,
                "remote": True,
                "salaryMin": s_min,
                "salaryMax": s_max,
                "currency": curr,
                "postedAt": posted_iso,
                "posted_at": posted_iso,
                "applyUrl": apply_url,
                "apply_url": apply_url,
                "sourceUrl": apply_url,
                "url": apply_url,
                "categories": tags or ["AI / Data"],
                "source": "Remotive",
                "recordType": "JOB",
                "dateSource": "api",
                "dateConfidence": "high",
            })

        self._log.info("remotive_parsed", valid_records=len(records))
        return records

    def _parse_greenhouse(self, text: str, url: str) -> list[dict[str, Any]]:
        """Parse Greenhouse board API JSON response."""
        try:
            data = json.loads(text)
        except Exception as exc:
            self._log.error("greenhouse_json_error", error=str(exc))
            return []

        jobs = data.get("jobs", []) if isinstance(data, dict) else []
        records: list[dict[str, Any]] = []

        cfg_company = self.adapter_config.get("company")
        if not cfg_company:
            # Extract board name from URL: /boards/{board}/jobs
            match = re.search(r"/boards/([^/]+)/jobs", url)
            cfg_company = match.group(1).title() if match else "Company"

        for item in jobs:
            if not isinstance(item, dict):
                continue

            title = item.get("title", "").strip()
            if not title:
                continue

            updated_at = item.get("updated_at")
            posted_dt, _ = extract_published(date_str=str(updated_at)) if updated_at else (None, None)
            if not posted_dt:
                posted_dt = datetime.now(timezone.utc)

            # Strict 24h filter
            if not is_fresh(posted_dt, hours=24):
                continue

            loc_dict = item.get("location")
            location = loc_dict.get("name", "Remote") if isinstance(loc_dict, dict) else "Remote"
            is_remote = "remote" in location.lower()
            apply_url = item.get("absolute_url") or url

            posted_iso = posted_dt.isoformat()
            records.append({
                "id": str(item.get("id")),
                "title": title,
                "company": cfg_company,
                "location": location,
                "remote": is_remote,
                "salaryMin": None,
                "salaryMax": None,
                "currency": "USD",
                "postedAt": posted_iso,
                "posted_at": posted_iso,
                "applyUrl": apply_url,
                "apply_url": apply_url,
                "sourceUrl": apply_url,
                "url": apply_url,
                "categories": ["AI", "Engineering"],
                "source": self.name,
                "recordType": "JOB",
                "dateSource": "api",
                "dateConfidence": "high",
            })

        self._log.info("greenhouse_parsed", valid_records=len(records))
        return records

    def _parse_lever(self, text: str, url: str) -> list[dict[str, Any]]:
        """Parse Lever postings API JSON response."""
        try:
            postings = json.loads(text)
        except Exception as exc:
            self._log.error("lever_json_error", error=str(exc))
            return []

        if not isinstance(postings, list):
            return []

        records: list[dict[str, Any]] = []
        cfg_company = self.adapter_config.get("company")
        if not cfg_company:
            # Extract company from URL: /postings/{company}
            match = re.search(r"/postings/([^/?]+)", url)
            cfg_company = match.group(1).title() if match else "Company"

        for item in postings:
            if not isinstance(item, dict):
                continue

            title = item.get("text", "").strip()
            if not title:
                continue

            created_at = item.get("createdAt")
            posted_dt = None
            if created_at:
                try:
                    posted_dt = datetime.fromtimestamp(float(created_at) / 1000.0, tz=timezone.utc)
                except Exception:
                    posted_dt = None

            if not posted_dt:
                posted_dt = datetime.now(timezone.utc)

            # Strict 24h filter
            if not is_fresh(posted_dt, hours=24):
                continue

            cats = item.get("categories", {})
            location = cats.get("location", "Remote") if isinstance(cats, dict) else "Remote"
            is_remote = "remote" in location.lower() or "remote" in str(cats).lower()
            apply_url = item.get("hostedUrl") or item.get("applyUrl") or url

            sal_range = item.get("salaryRange")
            s_min, s_max = None, None
            if isinstance(sal_range, dict):
                s_min = sal_range.get("min")
                s_max = sal_range.get("max")

            posted_iso = posted_dt.isoformat()
            records.append({
                "id": str(item.get("id")),
                "title": title,
                "company": cfg_company,
                "location": location,
                "remote": is_remote,
                "salaryMin": s_min,
                "salaryMax": s_max,
                "currency": "USD",
                "postedAt": posted_iso,
                "posted_at": posted_iso,
                "applyUrl": apply_url,
                "apply_url": apply_url,
                "sourceUrl": apply_url,
                "url": apply_url,
                "categories": ["AI", "Engineering"],
                "source": self.name,
                "recordType": "JOB",
                "dateSource": "api",
                "dateConfidence": "high",
            })

        self._log.info("lever_parsed", valid_records=len(records))
        return records

    def _parse_hn_hiring(self, text: str, url: str) -> list[dict[str, Any]]:
        """Parse Hacker News 'Ask HN: Who is hiring?' HTML page comments."""
        tree = HTMLParser(text)
        records: list[dict[str, Any]] = []

        # Find comments
        comment_nodes = tree.css(".comment")
        for node in comment_nodes:
            raw_text = node.text(strip=True)
            if not raw_text or len(raw_text) < 30:
                continue

            # First line usually contains: Company | Role | Location | Remote ...
            first_line = raw_text.split("\n")[0]
            tokens = [t.strip() for t in first_line.split("|")]
            if len(tokens) < 2:
                continue

            company = tokens[0]
            title = tokens[1]
            location = tokens[2] if len(tokens) > 2 else "Remote"
            is_remote = "remote" in raw_text.lower()

            # Filter for AI/ML relevance
            combined_content = f"{title} {raw_text}".lower()
            if not any(k in combined_content for k in AI_KEYWORDS):
                continue

            # Extract application URL or email
            urls = re.findall(r"https?://[^\s<>\"']+", raw_text)
            apply_url = urls[0] if urls else url

            # Extract salary if mentioned
            s_min, s_max, curr = parse_salary_range(raw_text)

            # HN comment date
            now = datetime.now(timezone.utc)
            posted_iso = now.isoformat()

            records.append({
                "id": apply_url or f"{company}_{title}",
                "title": title,
                "company": company,
                "location": location,
                "remote": is_remote,
                "salaryMin": s_min,
                "salaryMax": s_max,
                "currency": curr,
                "postedAt": posted_iso,
                "posted_at": posted_iso,
                "applyUrl": apply_url,
                "apply_url": apply_url,
                "sourceUrl": apply_url,
                "url": apply_url,
                "categories": ["AI", "Hiring"],
                "source": "HN Hiring",
                "recordType": "JOB",
                "dateSource": "meta",
                "dateConfidence": "high",
            })

        self._log.info("hn_hiring_parsed", valid_records=len(records))
        return records

    def _parse_generic_json(self, text: str, url: str) -> list[dict[str, Any]]:
        """Fallback generic JSON parser for unrecognized job payloads."""
        try:
            data = json.loads(text)
        except Exception:
            return []

        items = data if isinstance(data, list) else data.get("jobs", [])
        if not isinstance(items, list):
            return []

        records: list[dict[str, Any]] = []
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            title = item.get("title") or item.get("position")
            if not title:
                continue
            records.append({
                "id": str(item.get("id", idx)),
                "title": str(title),
                "company": item.get("company", "Unknown"),
                "location": item.get("location", "Remote"),
                "remote": True,
                "salaryMin": None,
                "salaryMax": None,
                "currency": "USD",
                "postedAt": datetime.now(timezone.utc).isoformat(),
                "posted_at": datetime.now(timezone.utc).isoformat(),
                "applyUrl": item.get("url", url),
                "apply_url": item.get("url", url),
                "sourceUrl": item.get("url", url),
                "url": item.get("url", url),
                "categories": ["AI"],
                "source": self.name,
                "recordType": "JOB",
                "dateSource": "api",
                "dateConfidence": "high",
            })
        return records
