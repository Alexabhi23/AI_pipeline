"""GitHub repository stars enricher with ETag caching and rate-limit handling.

Provides the `GitHubStarsEnricher` class to query the GitHub REST API for
repository stargazers counts while respecting API rate limits and minimizing
quota usage with HTTP ETag caching.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import structlog

from src.crawler.sources.base import HttpClient

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# Regex to parse GitHub repository owner and repo name from arbitrary URL
GITHUB_URL_REGEX = re.compile(
    r"(?:https?://)?(?:www\.)?github\.com/([a-zA-Z0-9_\-\.]+)/([a-zA-Z0-9_\-\.]+)",
    re.IGNORECASE,
)

IGNORED_REPOS = {
    "features", "topics", "collections", "trending", "events", "sponsors",
    "pricing", "login", "signup", "settings", "explore", "about", "site",
}


class GitHubStarsEnricher:
    """Enriches paper records with live GitHub repository stargazers counts."""

    def __init__(self, http_client: HttpClient, token: str | None = None) -> None:
        """Initialize enricher with HTTP client and optional GitHub API token.

        Args:
            http_client: HttpClient instance for performing HTTP calls.
            token: Optional GitHub personal access token (falls back to GITHUB_TOKEN env var).
        """
        self.http_client = http_client
        self.token = token or os.getenv("GITHUB_TOKEN")
        self._log = logger.bind(enricher="github_stars")

        # Cache: repo_key ("owner/repo") -> (etag, stargazers_count)
        self._etag_cache: dict[str, tuple[str, int]] = {}
        # Direct lookup cache: repo_key -> stargazers_count
        self._stars_cache: dict[str, int] = {}

        # Rate limit tracking
        self._rate_limit_remaining: int | None = None
        self._rate_limit_reset: float | None = None

    def extract_repo_key(self, repo_url: str) -> tuple[str, str] | None:
        """Extract (owner, repo) from a GitHub URL, or None if invalid."""
        if not repo_url:
            return None

        match = GITHUB_URL_REGEX.search(repo_url)
        if not match:
            return None

        owner = match.group(1).strip()
        repo = match.group(2).strip().removesuffix(".git").rstrip("/")

        if owner.lower() in IGNORED_REPOS or repo.lower() in IGNORED_REPOS:
            return None

        if not owner or not repo:
            return None

        return owner, repo

    async def get_stars(self, repo_url: str) -> int | None:
        """Fetch stargazers count for a given GitHub repository URL.

        Handles:
        - URL normalization and validation.
        - In-memory duplicate lookup caching.
        - ETag caching (`If-None-Match`).
        - Rate limit tracking (`X-RateLimit-Remaining`, `X-RateLimit-Reset`).

        Args:
            repo_url: URL to a GitHub repository (e.g. "https://github.com/vllm-project/vllm").

        Returns:
            int | None: Stargazers count, or None if unavailable / rate-limited.
        """
        parsed = self.extract_repo_key(repo_url)
        if not parsed:
            return None

        owner, repo = parsed
        repo_key = f"{owner}/{repo}".lower()

        # 1. Check in-memory stars cache
        if repo_key in self._stars_cache:
            return self._stars_cache[repo_key]

        # 2. Check rate limit status
        if self._rate_limit_remaining is not None and self._rate_limit_remaining <= 0:
            now = time.time()
            if self._rate_limit_reset and now < self._rate_limit_reset:
                wait_sec = int(self._rate_limit_reset - now)
                self._log.warning(
                    "github_rate_limited",
                    repo=repo_key,
                    reset_in_seconds=wait_sec,
                )
                if repo_key in self._etag_cache:
                    return self._etag_cache[repo_key][1]
                return None

        api_url = f"https://api.github.com/repos/{owner}/{repo}"
        headers: dict[str, str] = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "GraphOne-Crawler",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        cached_etag, cached_stars = self._etag_cache.get(repo_key, (None, None))
        if cached_etag:
            headers["If-None-Match"] = cached_etag

        # 3. Query GitHub API
        try:
            # Check if http_client has underlying session or custom request handling
            session = getattr(self.http_client, "session", None) or getattr(self.http_client, "_session", None)
            if session is not None and hasattr(session, "get"):
                async with session.get(api_url, headers=headers) as resp:
                    # Update rate limit tracking from headers
                    rem = resp.headers.get("X-RateLimit-Remaining")
                    if rem is not None:
                        try:
                            self._rate_limit_remaining = int(rem)
                        except ValueError:
                            pass

                    rst = resp.headers.get("X-RateLimit-Reset")
                    if rst is not None:
                        try:
                            self._rate_limit_reset = float(rst)
                        except ValueError:
                            pass

                    etag = resp.headers.get("ETag")

                    if resp.status == 304:
                        # 304 Not Modified: return cached stars count
                        self._log.debug("github_etag_304_not_modified", repo=repo_key)
                        if cached_stars is not None:
                            self._stars_cache[repo_key] = cached_stars
                            return cached_stars

                    if resp.status == 200:
                        data = await resp.json()
                        stars = data.get("stargazers_count")
                        if stars is not None:
                            stars_int = int(stars)
                            self._stars_cache[repo_key] = stars_int
                            if etag:
                                self._etag_cache[repo_key] = (etag, stars_int)
                            return stars_int

                    if resp.status in (403, 429):
                        self._log.warning("github_rate_limit_exceeded", repo=repo_key, status=resp.status)
                        return cached_stars

                    self._log.warning("github_api_error_status", repo=repo_key, status=resp.status)
                    return cached_stars

            # Fallback if standard get_text is used
            resp_text = await self.http_client.get_text(api_url, headers=headers)
            if not resp_text:
                return cached_stars

            data = json.loads(resp_text)
            if isinstance(data, dict):
                stars = data.get("stargazers_count")
                if stars is not None:
                    stars_int = int(stars)
                    self._stars_cache[repo_key] = stars_int
                    return stars_int
                # If message indicates rate limit
                if "API rate limit exceeded" in data.get("message", ""):
                    self._log.warning("github_rate_limit_message", repo=repo_key)
                    return cached_stars

        except Exception as exc:
            self._log.warning("github_stars_fetch_failed", repo=repo_key, error=str(exc))
            return cached_stars

        return None
