"""Base source adapter and configuration for the GraphOne crawler pipeline.

Provides the abstract ``BaseSource`` class that every concrete source adapter
(arXiv, RSS, web scraper, job board …) must subclass, the ``SourceConfig``
Pydantic model that mirrors a single entry in ``sources.yaml``, and the
``load_sources`` helper that reads and validates the YAML configuration file.
"""

from __future__ import annotations

import asyncio
import pathlib
import time
from abc import ABC, abstractmethod
from typing import Any, Sequence

import structlog
import yaml
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Structured logger
# ---------------------------------------------------------------------------

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration models
# ---------------------------------------------------------------------------


class RateLimitConfig(BaseModel):
    """Rate-limiting knobs for a single source."""

    requests_per_second: float = Field(
        default=1.0,
        gt=0,
        description="Maximum requests per second allowed by the source.",
    )

    @property
    def delay_seconds(self) -> float:
        """Minimum inter-request delay derived from the rate limit."""
        return 1.0 / self.requests_per_second


class SourceConfig(BaseModel):
    """Validated representation of one source entry in ``sources.yaml``."""

    name: str = Field(..., min_length=1, description="Unique human-readable source name.")
    type: str = Field(
        ...,
        min_length=1,
        description="Source transport type: 'api', 'web', or 'rss'.",
    )
    adapter: str = Field(
        ...,
        min_length=1,
        description="Adapter identifier used by the registry to look up the concrete class.",
    )
    enabled: bool = Field(default=True, description="Whether this source is active.")
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="Adapter-specific configuration dict (URLs, filters, etc.).",
    )
    rate_limit: RateLimitConfig = Field(
        default_factory=RateLimitConfig,
        description="Rate-limiting settings for this source.",
    )


# ---------------------------------------------------------------------------
# HTTP client protocol (structural typing)
# ---------------------------------------------------------------------------


class HttpClient(ABC):
    """Minimal async HTTP client interface expected by source adapters.

    Concrete implementations (e.g. an ``aiohttp.ClientSession`` wrapper) must
    satisfy this protocol.  The base module only declares the interface so that
    adapters can be tested against a stub without importing ``aiohttp``.
    """

    @abstractmethod
    async def get(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> bytes:
        """Perform an HTTP GET and return the raw response body."""
        ...

    @abstractmethod
    async def get_text(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> str:
        """Perform an HTTP GET and return the response body decoded as text."""
        ...


# ---------------------------------------------------------------------------
# Abstract base source adapter
# ---------------------------------------------------------------------------


class BaseSource(ABC):
    """Abstract base class for all source adapters.

    Subclasses **must** implement:

    * ``discover`` – return a list of URLs (or pseudo-URLs) to fetch.
    * ``fetch_and_parse`` – given one URL, fetch the resource and return a list
      of raw data dicts ready for downstream normalisation.

    The concrete ``run`` method orchestrates the full discover → fetch cycle
    with per-request rate limiting and structured error handling.
    """

    def __init__(self, source_config: SourceConfig, http_client: HttpClient) -> None:
        self._config: SourceConfig = source_config
        self._http: HttpClient = http_client
        self._log: structlog.stdlib.BoundLogger = logger.bind(
            source=source_config.name,
            adapter=source_config.adapter,
            source_type=source_config.type,
        )
        self._last_request_ts: float = 0.0

    # -- Properties ----------------------------------------------------------

    @property
    def http_client(self) -> HttpClient:
        """HTTP client instance used to fetch resources."""
        return self._http

    @property
    def name(self) -> str:
        """Human-readable name of this source."""
        return self._config.name

    @property
    def source_config(self) -> SourceConfig:
        """The full validated source configuration."""
        return self._config

    @property
    def adapter_config(self) -> dict[str, Any]:
        """Shortcut to the adapter-specific configuration dict."""
        return self._config.config

    # -- Rate limiting -------------------------------------------------------

    async def _respect_rate_limit(self) -> None:
        """Sleep if necessary to honour the configured rate limit."""
        delay = self._config.rate_limit.delay_seconds
        elapsed = time.monotonic() - self._last_request_ts
        if elapsed < delay:
            wait = delay - elapsed
            self._log.debug("rate_limit_wait", wait_seconds=round(wait, 3))
            await asyncio.sleep(wait)
        self._last_request_ts = time.monotonic()

    # -- Abstract interface --------------------------------------------------

    @abstractmethod
    async def discover(self) -> list[str]:
        """Return a list of URLs (or identifiers) to fetch.

        For API sources this might be a single endpoint URL; for web scrapers
        it could be a paginated list of detail-page URLs.
        """
        ...

    @abstractmethod
    async def fetch_and_parse(self, url: str) -> list[dict[str, Any]]:
        """Fetch *url* and return a list of raw data dictionaries.

        Each dict should contain at minimum an ``"id"`` or ``"url"`` key so
        downstream stages can deduplicate.
        """
        ...

    # -- Orchestrator --------------------------------------------------------

    async def run(self) -> list[dict[str, Any]]:
        """Execute the full discover → fetch → parse pipeline.

        Returns a flat list of raw data dicts aggregated from every discovered
        URL.  Individual fetch failures are logged and skipped so that one
        broken URL does not abort the entire source.
        """
        self._log.info("source_run_start")
        all_records: list[dict[str, Any]] = []

        # --- Discover -------------------------------------------------------
        try:
            urls: list[str] = await self.discover()
        except Exception:
            self._log.exception("discover_failed")
            return all_records

        self._log.info("discover_complete", url_count=len(urls))

        # --- Fetch & Parse --------------------------------------------------
        for idx, url in enumerate(urls):
            try:
                await self._respect_rate_limit()
                records = await self.fetch_and_parse(url)
                all_records.extend(records)
                self._log.debug(
                    "fetch_ok",
                    url=url,
                    record_count=len(records),
                    progress=f"{idx + 1}/{len(urls)}",
                )
            except Exception:
                self._log.exception("fetch_failed", url=url, progress=f"{idx + 1}/{len(urls)}")
                continue

        self._log.info(
            "source_run_complete",
            total_records=len(all_records),
            urls_attempted=len(urls),
        )
        return all_records


# ---------------------------------------------------------------------------
# YAML configuration loader
# ---------------------------------------------------------------------------


def load_sources(yaml_path: str | pathlib.Path) -> list[SourceConfig]:
    """Parse ``sources.yaml`` and return a list of validated ``SourceConfig`` objects.

    Parameters
    ----------
    yaml_path:
        Absolute or relative path to the YAML file containing the ``sources``
        key.

    Returns
    -------
    list[SourceConfig]
        Only **enabled** sources are included in the returned list.

    Raises
    ------
    FileNotFoundError
        If *yaml_path* does not exist.
    ValueError
        If the YAML structure is invalid or cannot be validated.
    """
    path = pathlib.Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Source config file not found: {path}")

    raw_text = path.read_text(encoding="utf-8")
    data: dict[str, Any] = yaml.safe_load(raw_text)

    if not isinstance(data, dict) or "sources" not in data:
        raise ValueError(
            f"Expected a top-level 'sources' key in {path}; "
            f"got keys: {list(data.keys()) if isinstance(data, dict) else type(data).__name__}"
        )

    raw_sources: list[dict[str, Any]] = data["sources"]
    if not isinstance(raw_sources, list):
        raise ValueError(f"'sources' must be a list, got {type(raw_sources).__name__}")

    configs: list[SourceConfig] = []
    for entry in raw_sources:
        try:
            cfg = SourceConfig.model_validate(entry)
        except Exception as exc:
            logger.warning(
                "source_config_invalid",
                entry_name=entry.get("name", "<unknown>"),
                error=str(exc),
            )
            continue

        if cfg.enabled:
            configs.append(cfg)
            logger.debug("source_loaded", name=cfg.name, adapter=cfg.adapter)
        else:
            logger.debug("source_skipped_disabled", name=cfg.name)

    logger.info(
        "sources_loaded",
        total_defined=len(raw_sources),
        total_enabled=len(configs),
    )
    return configs
