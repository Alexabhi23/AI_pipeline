"""Playwright browser fetcher for dynamic, JavaScript-rendered web pages.

Provides the ``PlaywrightFetcher`` class which uses headless Chromium to render
pages with modern JavaScript frameworks (SPAs, React, Vue, Next.js, etc.) and
extract full DOM state and response headers.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

import structlog

from src.config import Settings, get_settings

if TYPE_CHECKING:
    from src.crawler.http import HttpResponse

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


class PlaywrightFetcher:
    """Asynchronous headless browser fetcher powered by Playwright Chromium.

    Features:
    - Lazy browser and playwright initialization on first request
    - Configurable User-Agent and realistic desktop viewport (1280x800)
    - Concurrency-safe initialization via ``asyncio.Lock``
    - Graceful fallback and error logging when Playwright or browser binaries
      are not installed or execution fails.
    """

    def __init__(
        self,
        user_agent: Optional[str] = None,
        timeout_ms: int = 30000,
        settings: Optional[Settings] = None,
    ) -> None:
        cfg = settings or get_settings()
        self.user_agent: str = user_agent or cfg.USER_AGENT
        self.timeout_ms: int = timeout_ms
        self._lock: asyncio.Lock = asyncio.Lock()
        self._playwright: Any = None
        self._browser: Any = None
        self._closed: bool = False

    async def _init_browser(self) -> bool:
        """Lazily initialize the Playwright Chromium instance.

        Returns True if successful, False if playwright is unavailable.
        """
        if self._browser is not None:
            return True

        async with self._lock:
            if self._browser is not None:
                return True

            try:
                from playwright.async_api import async_playwright

                self._playwright = await async_playwright().start()
                self._browser = await self._playwright.chromium.launch(
                    headless=True,
                    args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
                )
                logger.info("playwright_browser_launched", headless=True)
                return True
            except ImportError:
                logger.warning(
                    "playwright_not_installed",
                    msg="playwright package is not installed; dynamic browser fetching is disabled.",
                )
                return False
            except Exception as exc:
                logger.error(
                    "playwright_init_failed",
                    error=str(exc),
                    msg="Failed to launch Playwright Chromium browser.",
                )
                if self._playwright:
                    try:
                        await self._playwright.stop()
                    except Exception:
                        pass
                    self._playwright = None
                return False

    async def fetch(self, url: str) -> HttpResponse:
        """Navigate to *url* using headless Chromium and return page state.

        Waits for 'domcontentloaded' lifecycle event or timeout.
        Returns an ``HttpResponse`` dataclass instance.
        """
        from src.crawler.http import HttpResponse

        ok = await self._init_browser()
        if not ok or self._browser is None:
            logger.warning("playwright_fetch_fallback_invoked", url=url)
            return HttpResponse(
                url=url,
                status_code=503,
                headers={},
                text="",
                fetched_at=datetime.now(timezone.utc),
                from_cache=False,
            )

        context = None
        page = None
        try:
            context = await self._browser.new_context(
                user_agent=self.user_agent,
                viewport={"width": 1280, "height": 800},
                ignore_https_errors=True,
            )
            page = await context.new_page()

            response = await page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=self.timeout_ms,
            )

            status_code = response.status if response else 200
            headers: dict[str, str] = {}
            if response:
                try:
                    headers = dict(await response.all_headers())
                except Exception:
                    headers = dict(response.headers)

            content = await page.content()

            logger.debug(
                "playwright_fetch_success",
                url=url,
                status_code=status_code,
                content_length=len(content),
            )

            return HttpResponse(
                url=url,
                status_code=status_code,
                headers=headers,
                text=content,
                fetched_at=datetime.now(timezone.utc),
                from_cache=False,
            )

        except Exception as exc:
            logger.error("playwright_fetch_error", url=url, error=str(exc))
            return HttpResponse(
                url=url,
                status_code=500,
                headers={},
                text=f"Browser fetch error: {exc}",
                fetched_at=datetime.now(timezone.utc),
                from_cache=False,
            )
        finally:
            if page is not None:
                try:
                    await page.close()
                except Exception:
                    pass
            if context is not None:
                try:
                    await context.close()
                except Exception:
                    pass

    async def close(self) -> None:
        """Close browser instance and terminate Playwright subprocess."""
        async with self._lock:
            if self._closed:
                return
            self._closed = True

            if self._browser is not None:
                try:
                    await self._browser.close()
                    logger.debug("playwright_browser_closed")
                except Exception as exc:
                    logger.warning("playwright_browser_close_error", error=str(exc))
                self._browser = None

            if self._playwright is not None:
                try:
                    await self._playwright.stop()
                    logger.debug("playwright_stopped")
                except Exception as exc:
                    logger.warning("playwright_stop_error", error=str(exc))
                self._playwright = None

    async def __aenter__(self) -> PlaywrightFetcher:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()
