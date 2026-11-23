"""Configuration module for the GraphOne/FrontierAtlas pipeline.

Loads environment variables from .env via python-dotenv and exposes a
cached ``get_settings()`` helper that returns a validated
``Settings`` instance backed by Pydantic v2 Settings.
"""

from __future__ import annotations

import functools
from typing import Optional

import structlog
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Ensure .env is loaded as early as possible
# ---------------------------------------------------------------------------
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

logger = structlog.get_logger(__name__)

# A realistic Chrome-on-Windows User-Agent string (Chrome 120, Win 10).
_DEFAULT_USER_AGENT: str = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


class Settings(BaseSettings):
    """Central, validated configuration for every pipeline component.

    Values are read – in descending priority – from:
    1. Explicit environment variables
    2. A ``.env`` file in the project root
    3. The defaults declared below
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    # ── LLM provider keys ─────────────────────────────────────────────
    GEMINI_API_KEY: Optional[str] = Field(
        default=None,
        description="Google Gemini / GenAI API key.",
    )
    GROQ_API_KEY: Optional[str] = Field(
        default=None,
        description="Groq cloud API key.",
    )
    DEEPSEEK_API_KEY: Optional[str] = Field(
        default=None,
        description="DeepSeek API key.",
    )

    # ── Storage ────────────────────────────────────────────────────────
    REDIS_URL: Optional[str] = Field(
        default=None,
        description="Redis connection URL (e.g. redis://localhost:6379/0).",
    )
    SQLITE_DB_PATH: str = Field(
        default="data/pipeline.db",
        description="Path to the local SQLite database file.",
    )

    # ── Google Sheets export ───────────────────────────────────────────
    GOOGLE_SHEETS_CREDS_PATH: Optional[str] = Field(
        default=None,
        description="Path to the Google service-account JSON credentials.",
    )
    SPREADSHEET_ID: Optional[str] = Field(
        default=None,
        description="Target Google Spreadsheet ID for data export.",
    )

    # ── Pipeline tuning ────────────────────────────────────────────────
    MAX_CONCURRENCY: int = Field(
        default=20,
        ge=1,
        description="Maximum number of concurrent async tasks.",
    )
    PER_DOMAIN_DELAY: float = Field(
        default=1.0,
        ge=0.0,
        description="Minimum seconds between requests to the same domain.",
    )
    LLM_TIMEOUT: int = Field(
        default=30,
        ge=1,
        description="Timeout in seconds for a single LLM API call.",
    )
    FRESHNESS_HOURS: int = Field(
        default=24,
        ge=1,
        description="Re-scrape pages older than this many hours.",
    )

    # ── Observability ──────────────────────────────────────────────────
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Root log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )

    # ── Source definition ──────────────────────────────────────────────
    SOURCES_YAML_PATH: str = Field(
        default="sources.yaml",
        description="Path to the YAML file that lists scraping sources.",
    )

    # ── HTTP ───────────────────────────────────────────────────────────
    USER_AGENT: str = Field(
        default=_DEFAULT_USER_AGENT,
        description="User-Agent header sent with outgoing HTTP requests.",
    )


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide cached ``Settings`` instance.

    The first call validates the environment; subsequent calls return the
    same object without re-reading the environment.
    """
    settings = Settings()  # type: ignore[call-arg]
    logger.info(
        "settings_loaded",
        log_level=settings.LOG_LEVEL,
        max_concurrency=settings.MAX_CONCURRENCY,
        sqlite_db=settings.SQLITE_DB_PATH,
        freshness_hours=settings.FRESHNESS_HOURS,
    )
    return settings
