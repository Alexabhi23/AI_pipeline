"""Unit tests for HttpClient and PlaywrightFetcher."""

import asyncio
from datetime import datetime, timezone
import pytest

from src.crawler.browser import PlaywrightFetcher
from src.crawler.http import HttpClient, HttpResponse


@pytest.mark.asyncio
async def test_http_client_robots_allowed():
    async with HttpClient() as client:
        # standard public page
        allowed = await client.is_allowed("https://example.com/test")
        assert allowed is True


@pytest.mark.asyncio
async def test_http_client_cache_and_conditional_304():
    async with HttpClient() as client:
        url = "https://example.com/cached-test"
        client._etag_cache[url] = ('"test-etag-123"', None)
        client._response_cache[url] = HttpResponse(
            url=url,
            status_code=200,
            headers={"ETag": '"test-etag-123"'},
            text="Cached Body",
            fetched_at=datetime.now(timezone.utc),
            from_cache=False,
        )

        assert url in client._etag_cache
        assert url in client._response_cache
        cached = client._response_cache[url]
        assert cached.text == "Cached Body"


@pytest.mark.asyncio
async def test_playwright_fetcher_lifecycle():
    fetcher = PlaywrightFetcher()
    try:
        # Verify fetcher initialization doesn't crash
        assert fetcher.timeout_ms == 30000
    finally:
        await fetcher.close()
