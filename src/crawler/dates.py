"""Date extraction, freshness checking, and hashing utilities for crawling.

Provides robust multi-strategy publication date extraction:
1. HTML meta tags (article:published_time, og:published_time, date, DC.date, parsely-pub-date)
2. JSON-LD structured data (datePublished, uploadDate)
3. HTML <time> tags (datetime, pubdate, or text content)
4. RSS/Atom XML tags (<pubDate>, <updated>)
5. Sitemap XML tags (<lastmod>)
6. HTTP Last-Modified headers
7. Relative text parsing via dateparser
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import re
from typing import Any, Optional
import urllib.parse

import structlog

from src.models import DateConfidence, DateSource

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# Tracking parameters stripped during URL canonicalization
_TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "fbclid",
    "gclid",
    "ref",
    "source",
    "mc_cid",
    "mc_eid",
    "yclid",
    "_ga",
    "_gl",
}

# Regex for relative dates in plain text (e.g. "3 hours ago", "yesterday")
_RELATIVE_DATE_RE = re.compile(
    r"\b(\d+\s+(?:minute|hour|day|week|month|year)s?\s+ago|yesterday|today|just now)\b",
    re.IGNORECASE,
)

# RSS / Atom / XML tag patterns
_RSS_PUB_DATE_RE = re.compile(r"<pubDate>(.*?)</pubDate>", re.IGNORECASE | re.DOTALL)
_ATOM_UPDATED_RE = re.compile(r"<(?:updated|published)>(.*?)</(?:updated|published)>", re.IGNORECASE | re.DOTALL)
_SITEMAP_LASTMOD_RE = re.compile(r"<lastmod>(.*?)</lastmod>", re.IGNORECASE | re.DOTALL)

_META_TAG_KEYS = {
    "article:published_time",
    "og:published_time",
    "date",
    "dc.date",
    "dc.date.issued",
    "parsely-pub-date",
    "sailthru.date",
    "pubdate",
    "publishdate",
    "release_date",
    "article:modified_time",
    "og:updated_time",
}


def _normalize_utc(dt: Any) -> Optional[datetime]:
    """Ensure datetime is UTC-aware and normalized."""
    if dt is None:
        return None
    if isinstance(dt, str):
        parsed = _parse_date_str(dt)
        if parsed is None:
            return None
        dt = parsed
    if getattr(dt, "tzinfo", None) is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_date_str(date_str: str) -> Optional[datetime]:
    """Parse an arbitrary date string into a UTC datetime."""
    if not date_str or not isinstance(date_str, str):
        return None

    cleaned = date_str.strip()
    if not cleaned:
        return None

    # Fast ISO-8601 parsing
    try:
        iso_cand = cleaned.replace("Z", "+00:00")
        dt = datetime.fromisoformat(iso_cand)
        return _normalize_utc(dt)
    except (ValueError, TypeError):
        pass

    # Fallback to dateparser
    try:
        import dateparser

        dt = dateparser.parse(
            cleaned,
            settings={
                "TIMEZONE": "UTC",
                "RETURN_AS_TIMEZONE_AWARE": True,
                "PREFER_DATES_FROM": "past",
            },
        )
        return _normalize_utc(dt)
    except Exception:
        return None


def _find_date_in_json_ld(obj: Any) -> Optional[str]:
    """Recursively search a JSON-LD data structure for publication dates."""
    if isinstance(obj, dict):
        for key in ("datePublished", "uploadDate", "dateCreated", "dateModified"):
            val = obj.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()

        # Check @graph if present
        if "@graph" in obj:
            found = _find_date_in_json_ld(obj["@graph"])
            if found:
                return found

        for v in obj.values():
            found = _find_date_in_json_ld(v)
            if found:
                return found

    elif isinstance(obj, list):
        for item in obj:
            found = _find_date_in_json_ld(item)
            if found:
                return found

    return None


def extract_published(
    html: str = "",
    headers: Optional[dict[str, Any]] = None,
    url: str = "",
    date_str: Optional[str] = None,
) -> tuple[Optional[datetime], DateSource]:
    """Extract publication datetime from HTML, headers, date_str, or feeds.

    Follows prioritized hierarchy:
    0. Explicit date string if provided
    1. HTML meta tags (article:published_time, og:published_time, date, DC.date, parsely-pub-date)
    2. JSON-LD script tags (datePublished, uploadDate)
    3. HTML <time> tags (datetime or pubdate attribute)
    4. RSS/Atom XML tags (<pubDate>, <updated>)
    5. Sitemap XML tags (<lastmod>)
    6. HTTP Last-Modified header
    7. Relative date parsing in body text
    """
    if date_str:
        dt = _parse_date_str(date_str)
        if dt:
            return dt, DateSource.API

    headers = headers or {}

    if html:
        # 1-3: Try selectolax LexborHTMLParser
        try:
            from selectolax.lexbor import LexborHTMLParser

            parser = LexborHTMLParser(html)

            # 1. Meta tags
            for meta_node in parser.css("meta"):
                attr_name = (
                    meta_node.attributes.get("property")
                    or meta_node.attributes.get("name")
                    or meta_node.attributes.get("itemprop")
                )
                if attr_name and attr_name.lower() in _META_TAG_KEYS:
                    content = meta_node.attributes.get("content")
                    if content:
                        dt = _parse_date_str(content)
                        if dt:
                            return dt, DateSource.META

            # 2. JSON-LD scripts
            for script_node in parser.css('script[type="application/ld+json"]'):
                raw_text = script_node.text()
                if not raw_text:
                    continue
                try:
                    data = json.loads(raw_text.strip())
                    date_val = _find_date_in_json_ld(data)
                    if date_val:
                        dt = _parse_date_str(date_val)
                        if dt:
                            return dt, DateSource.META
                except (json.JSONDecodeError, TypeError):
                    continue

            # 3. HTML <time> tags
            for time_node in parser.css("time"):
                datetime_attr = time_node.attributes.get("datetime") or time_node.attributes.get("pubdate")
                if datetime_attr:
                    dt = _parse_date_str(datetime_attr)
                    if dt:
                        return dt, DateSource.META

                text_content = time_node.text()
                if text_content:
                    dt = _parse_date_str(text_content.strip())
                    if dt:
                        return dt, DateSource.META

        except Exception as exc:
            logger.debug("lexbor_parser_exception", url=url, error=str(exc))

        # Fallback for XML feeds and sitemaps
        # 4. RSS / Atom feed tags
        m = _RSS_PUB_DATE_RE.search(html)
        if m:
            dt = _parse_date_str(m.group(1).strip())
            if dt:
                return dt, DateSource.API

        m = _ATOM_UPDATED_RE.search(html)
        if m:
            dt = _parse_date_str(m.group(1).strip())
            if dt:
                return dt, DateSource.API

        # 5. Sitemap XML tags
        m = _SITEMAP_LASTMOD_RE.search(html)
        if m:
            dt = _parse_date_str(m.group(1).strip())
            if dt:
                return dt, DateSource.API

    # 6. HTTP Last-Modified header
    last_mod_header = headers.get("Last-Modified") or headers.get("last-modified")
    if last_mod_header and isinstance(last_mod_header, str):
        dt = _parse_date_str(last_mod_header)
        if dt:
            return dt, DateSource.META

    # 7. Relative date parsing in text
    if html:
        rel_match = _RELATIVE_DATE_RE.search(html)
        if rel_match:
            dt = _parse_date_str(rel_match.group(1))
            if dt:
                return dt, DateSource.RELATIVE

    # No date detected
    return None, DateSource.HEURISTIC


def is_fresh(dt: Optional[datetime], hours: int = 24) -> bool:
    """Check if *dt* falls within the last *hours* relative to current UTC time."""
    if dt is None:
        return False
    dt_utc = _normalize_utc(dt)
    if dt_utc is None:
        return False
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    return dt_utc >= cutoff


def compute_content_hash(text: str) -> str:
    """Compute 64-bit xxhash or sha256 hex digest of string content."""
    data = text.encode("utf-8", errors="replace")
    try:
        import xxhash

        return xxhash.xxh64_hexdigest(data)
    except ImportError:
        import hashlib

        return hashlib.sha256(data).hexdigest()


def normalize_url(url: str) -> str:
    """Canonicalize a URL: lowercase domain, strip tracking params, sort query."""
    if not url:
        return ""
    try:
        parsed = urllib.parse.urlsplit(url.strip())
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()

        # Remove default ports
        if (scheme == "http" and netloc.endswith(":80")) or (
            scheme == "https" and netloc.endswith(":443")
        ):
            netloc = netloc.rsplit(":", 1)[0]

        path = parsed.path
        if path != "/" and path.endswith("/"):
            path = path.rstrip("/")
        if not path:
            path = "/"

        # Filter out tracking query parameters
        query_params = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        filtered = [
            (k, v) for k, v in query_params if k.lower() not in _TRACKING_PARAMS
        ]
        filtered.sort(key=lambda pair: pair[0])
        clean_query = urllib.parse.urlencode(filtered)

        # Omit fragment
        return urllib.parse.urlunsplit((scheme, netloc, path, clean_query, ""))
    except Exception:
        return url.strip()


def compute_url_hash(url: str) -> str:
    """Compute hash of the canonicalized URL."""
    return compute_content_hash(normalize_url(url))
