"""High-concurrency, polite HTTP client for the GraphOne crawler pipeline.

Implements bounded concurrency with a global semaphore, per-domain rate limiting,
robots.txt parsing and caching, conditional ETag/Last-Modified caching,
exponential backoff retry on 5xx/429 with full jitter, and optional delegation
to Playwright for JavaScript-heavy pages.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import random
import time
from typing import Any, Optional
import urllib.parse
import urllib.robotparser

import aiohttp
import structlog

from src.config import Settings, get_settings
from src.crawler.browser import PlaywrightFetcher

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data transfer objects
# ---------------------------------------------------------------------------


@dataclass
class HttpResponse:
    """Standardized HTTP response container for all crawled resources."""

    url: str
    status_code: int
    headers: dict[str, str] = field(default_factory=dict)
    text: str = ""
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    from_cache: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_retry_after(header_val: Optional[str]) -> float:
    """Parse HTTP Retry-After header into seconds to wait."""
    if not header_val:
        return 2.0
    val_str = header_val.strip()
    try:
        val = float(val_str)
        return max(0.5, val)
    except ValueError:
        pass
    try:
        dt = parsedate_to_datetime(val_str)
        diff = (dt - datetime.now(timezone.utc)).total_seconds()
        return max(0.5, diff)
    except Exception:
        return 2.0


# ---------------------------------------------------------------------------
# HttpClient implementation
# ---------------------------------------------------------------------------


class HttpClient:
    """Asynchronous HTTP client tailored for polite, robust web crawling.

    Features:
    - Bounded global concurrency via an asyncio.Semaphore
    - Per-domain rate limiting with asyncio.Lock + last_request_time
    - robots.txt compliance checking with memory caching
    - HTTP conditional requests (ETag / If-None-Match, Last-Modified / If-Modified-Since)
    - Full-jitter backoff retry on HTTP 429 and 5xx errors
    - Seamless fallback or delegation to PlaywrightFetcher for dynamic rendering
    - Context-manager lifecycle management (__aenter__ / __aexit__)
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        max_concurrency: Optional[int] = None,
        per_domain_delay: Optional[float] = None,
        user_agent: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        cfg = settings or get_settings()
        self.settings = cfg
        self.max_concurrency: int = max_concurrency or cfg.MAX_CONCURRENCY
        self.per_domain_delay: float = (
            per_domain_delay if per_domain_delay is not None else cfg.PER_DOMAIN_DELAY
        )
        self.user_agent: str = user_agent or cfg.USER_AGENT
        self.timeout = aiohttp.ClientTimeout(total=timeout)

        # Concurrency & rate-limiting primitives
        self._semaphore = asyncio.Semaphore(self.max_concurrency)
        self._domain_locks: dict[str, asyncio.Lock] = {}
        self._last_request_time: dict[str, float] = {}

        # Caches
        self._robots_cache: dict[str, urllib.robotparser.RobotFileParser] = {}
        self._etag_cache: dict[str, tuple[Optional[str], Optional[str]]] = {}
        self._response_cache: dict[str, HttpResponse] = {}

        # Sub-components
        self._session: Optional[aiohttp.ClientSession] = None
        self._playwright_fetcher: Optional[PlaywrightFetcher] = None
        self._closed: bool = False

    async def _get_session(self) -> aiohttp.ClientSession:
        """Return or lazily construct the underlying aiohttp.ClientSession."""
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(
                limit=self.max_concurrency,
                limit_per_host=5,
                enable_cleanup_closed=True,
                ssl=False,
            )
            self._session = aiohttp.ClientSession(
                connector=connector,
                headers={"User-Agent": self.user_agent},
                timeout=self.timeout,
            )
        return self._session

    def _get_domain_lock(self, domain: str) -> asyncio.Lock:
        """Retrieve or create a concurrency lock for the given domain."""
        if domain not in self._domain_locks:
            self._domain_locks[domain] = asyncio.Lock()
        return self._domain_locks[domain]

    async def is_allowed(self, url: str) -> bool:
        """Verify whether url crawling is permitted by domain robots.txt."""
        try:
            parsed = urllib.parse.urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return True

            domain = parsed.netloc.lower()
            if domain not in self._robots_cache:
                robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
                session = await self._get_session()
                rp = urllib.robotparser.RobotFileParser()
                rp.set_url(robots_url)
                try:
                    async with session.get(
                        robots_url,
                        timeout=aiohttp.ClientTimeout(total=5.0),
                    ) as r_resp:
                        if r_resp.status == 200:
                            content = await r_resp.text(errors="replace")
                            rp.parse(content.splitlines())
                        else:
                            # 404/403 or other non-200 -> allow crawling
                            setattr(rp, "_graphone_allow_all", True)
                except Exception as err:
                    logger.debug("robots_txt_fetch_failed", domain=domain, error=str(err))
                    setattr(rp, "_graphone_allow_all", True)

                self._robots_cache[domain] = rp

            parser = self._robots_cache[domain]
            if getattr(parser, "_graphone_allow_all", False):
                return True
            return parser.can_fetch(self.user_agent, url)

        except Exception as exc:
            logger.warning("robots_txt_eval_error", url=url, error=str(exc))
            return True

    async def fetch(self, url: str, use_browser: bool = False) -> HttpResponse:
        """Fetch *url* respecting concurrency, rate limits, robots.txt, and caches."""
        if use_browser:
            allowed = await self.is_allowed(url)
            if not allowed:
                logger.warning("crawl_disallowed_by_robots", url=url)
                return HttpResponse(
                    url=url,
                    status_code=403,
                    headers={},
                    text="Disallowed by robots.txt",
                    fetched_at=datetime.now(timezone.utc),
                    from_cache=False,
                )
            if self._playwright_fetcher is None:
                self._playwright_fetcher = PlaywrightFetcher(
                    user_agent=self.user_agent,
                    settings=self.settings,
                )
            return await self._playwright_fetcher.fetch(url)

        # 1. robots.txt check
        allowed = await self.is_allowed(url)
        if not allowed:
            logger.warning("crawl_disallowed_by_robots", url=url)
            return HttpResponse(
                url=url,
                status_code=403,
                headers={},
                text="Disallowed by robots.txt",
                fetched_at=datetime.now(timezone.utc),
                from_cache=False,
            )

        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc.lower()

        # 2. Acquire global semaphore for concurrency bounding
        async with self._semaphore:
            # 3. Acquire per-domain lock for polite sequential domain crawling
            async with self._get_domain_lock(domain):
                # Enforce per-domain minimum delay
                now = time.monotonic()
                last_time = self._last_request_time.get(domain, 0.0)
                elapsed = now - last_time
                if elapsed < self.per_domain_delay:
                    await asyncio.sleep(self.per_domain_delay - elapsed)

                session = await self._get_session()

                # Prepare conditional headers if cached
                req_headers: dict[str, str] = {}
                if url in self._etag_cache:
                    cached_etag, cached_last_mod = self._etag_cache[url]
                    if cached_etag:
                        req_headers["If-None-Match"] = cached_etag
                    if cached_last_mod:
                        req_headers["If-Modified-Since"] = cached_last_mod

                # Retry loop for 5xx and 429
                max_retries = 3
                last_error: Optional[Exception] = None

                for attempt in range(max_retries + 1):
                    try:
                        async with session.get(url, headers=req_headers) as resp:
                            resp_headers = dict(resp.headers)
                            self._last_request_time[domain] = time.monotonic()

                            # 304 Not Modified: return cached payload
                            if resp.status == 304:
                                cached_resp = self._response_cache.get(url)
                                if cached_resp:
                                    logger.debug("http_cache_hit_304", url=url)
                                    return HttpResponse(
                                        url=url,
                                        status_code=304,
                                        headers={**cached_resp.headers, **resp_headers},
                                        text=cached_resp.text,
                                        fetched_at=datetime.now(timezone.utc),
                                        from_cache=True,
                                    )
                                return HttpResponse(
                                    url=url,
                                    status_code=304,
                                    headers=resp_headers,
                                    text="",
                                    fetched_at=datetime.now(timezone.utc),
                                    from_cache=True,
                                )

                            # 429 Rate limited: honor Retry-After + full jitter
                            if resp.status == 429 and attempt < max_retries:
                                retry_after_raw = resp_headers.get("Retry-After") or resp_headers.get("retry-after")
                                wait_base = _parse_retry_after(retry_after_raw)
                                jitter_delay = random.uniform(0.5, max(1.0, wait_base + 1.0))
                                logger.warning(
                                    "http_rate_limited_429",
                                    url=url,
                                    attempt=attempt + 1,
                                    retry_after_sec=wait_base,
                                    jitter_delay_sec=round(jitter_delay, 2),
                                )
                                await asyncio.sleep(jitter_delay)
                                continue

                            # 5xx Server Error: exponential backoff + jitter
                            if 500 <= resp.status <= 599 and attempt < max_retries:
                                backoff_delay = (2.0 ** attempt) + random.uniform(0.1, 0.5)
                                logger.warning(
                                    "http_server_error_5xx",
                                    url=url,
                                    status=resp.status,
                                    attempt=attempt + 1,
                                    backoff_sec=round(backoff_delay, 2),
                                )
                                await asyncio.sleep(backoff_delay)
                                continue

                            # Read response body
                            body_text = await resp.text(errors="replace")

                            # Store conditional cache tokens on 200 OK
                            if resp.status == 200:
                                etag = resp_headers.get("ETag") or resp_headers.get("etag")
                                last_mod = resp_headers.get("Last-Modified") or resp_headers.get("last-modified")
                                if etag or last_mod:
                                    self._etag_cache[url] = (etag, last_mod)

                            http_resp = HttpResponse(
                                url=url,
                                status_code=resp.status,
                                headers=resp_headers,
                                text=body_text,
                                fetched_at=datetime.now(timezone.utc),
                                from_cache=False,
                            )

                            if resp.status == 200:
                                self._response_cache[url] = http_resp

                            return http_resp

                    except (aiohttp.ClientError, asyncio.TimeoutError) as err:
                        last_error = err
                        self._last_request_time[domain] = time.monotonic()
                        if attempt < max_retries:
                            backoff_delay = (2.0 ** attempt) + random.uniform(0.1, 0.5)
                            logger.warning(
                                "http_connection_error_retry",
                                url=url,
                                attempt=attempt + 1,
                                error=str(err),
                                backoff_sec=round(backoff_delay, 2),
                            )
                            await asyncio.sleep(backoff_delay)
                            continue
                        logger.error("http_fetch_exhausted_retries", url=url, error=str(err))
                        break

                return HttpResponse(
                    url=url,
                    status_code=500,
                    headers={},
                    text=f"HTTP Request failed after retries: {last_error}",
                    fetched_at=datetime.now(timezone.utc),
                    from_cache=False,
                )

    # -----------------------------------------------------------------------
    # Compatibility with BaseSource interface
    # -----------------------------------------------------------------------

    async def get(
        self,
        url: str,
        *,
        headers: Optional[dict[str, str]] = None,
        params: Optional[dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> bytes:
        """Fetch raw response bytes."""
        final_url = url
        if params:
            query = urllib.parse.urlencode(params)
            sep = "&" if "?" in final_url else "?"
            final_url = f"{final_url}{sep}{query}"
        resp = await self.fetch(final_url)
        return resp.text.encode("utf-8")

    async def get_text(
        self,
        url: str,
        *,
        headers: Optional[dict[str, str]] = None,
        params: Optional[dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> str:
        """Fetch response decoded as text."""
        final_url = url
        if params:
            query = urllib.parse.urlencode(params)
            sep = "&" if "?" in final_url else "?"
            final_url = f"{final_url}{sep}{query}"
        resp = await self.fetch(final_url)
        return resp.text

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    async def close(self) -> None:
        """Gracefully release HTTP sessions and browser instances."""
        if self._closed:
            return
        self._closed = True

        if self._session is not None and not self._session.closed:
            await self._session.close()
            # Allow underlying SSL connections to terminate
            await asyncio.sleep(0.05)
            self._session = None

        if self._playwright_fetcher is not None:
            await self._playwright_fetcher.close()
            self._playwright_fetcher = None

        logger.debug("http_client_closed")

    async def __aenter__(self) -> HttpClient:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()
