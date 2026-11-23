"""Pydantic v2 domain models shared across the GraphOne pipeline.

Defines the canonical data-transfer objects for records, entity mappings,
LLM-call telemetry, and pipeline settings.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, List, Optional
from pydantic import BaseModel, Field, HttpUrl


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PricingModel(str, Enum):
    """Product/Startup pricing tiers."""
    FREE = "FREE"
    FREEMIUM = "FREEMIUM"
    PAID = "PAID"
    ENTERPRISE = "ENTERPRISE"


class DateSource(str, Enum):
    """Origin source of the extracted publication date."""
    META = "meta"
    RELATIVE = "relative"
    HEURISTIC = "heuristic"
    API = "api"


class DateConfidence(str, Enum):
    """Confidence tier for publication date."""
    HIGH = "high"
    LOW = "low"


class RecordType(str, Enum):
    """Discriminator for pipeline record types."""
    STARTUP = "STARTUP"
    PRODUCT = "PRODUCT"
    PAPER = "PAPER"
    JOB = "JOB"
    NEWS = "NEWS"


# ---------------------------------------------------------------------------
# Base & Domain Record models
# ---------------------------------------------------------------------------

class BaseRecord(BaseModel):
    """Canonical base record extracted from a crawled page."""

    schemaVersion: str = Field(default="1.0", description="Schema specification version.")
    recordType: RecordType = Field(..., description="Record discriminator.")
    sourceUrl: str = Field(..., description="Verified source URL that returned HTTP 200.")
    collectedAt: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="ISO-8601 UTC timestamp when collected.",
    )
    urlHash: str = Field(..., description="Hash of canonical source URL.")
    contentHash: str = Field(..., description="Hash of page content.")
    dateSource: Optional[DateSource] = Field(default=None, description="Origin of date.")
    dateConfidence: Optional[DateConfidence] = Field(default=None, description="Confidence in date.")

    def to_db_row(self) -> dict[str, Any]:
        """Convert to database storage row."""
        return {
            "record_type": self.recordType.value,
            "url_hash": self.urlHash,
            "payload_json": self.model_dump_json(exclude={"schemaVersion", "recordType", "sourceUrl", "collectedAt", "urlHash", "contentHash", "dateSource", "dateConfidence"}),
            "date_source": self.dateSource.value if self.dateSource else None,
            "date_confidence": self.dateConfidence.value if self.dateConfidence else None,
            "collected_at": self.collectedAt.isoformat(),
        }


class Startup(BaseRecord):
    """Curated AI startup record."""
    recordType: RecordType = Field(default=RecordType.STARTUP)
    name: str = Field(..., description="Startup name.")
    description: Optional[str] = Field(default=None, description="One-line or short summary.")
    founders: Optional[List[str]] = Field(default=None, description="List of founder names.")
    foundedDate: Optional[str] = Field(default=None, description="Founding date or year.")
    headquarters: Optional[str] = Field(default=None, description="City, Country.")
    employeeCount: Optional[int] = Field(default=None, description="Total headcount.")
    funding: Optional[str] = Field(default=None, description="Total funding or last round info.")
    pricingModel: Optional[PricingModel] = Field(default=None, description="Pricing tier.")
    website: Optional[str] = Field(default=None, description="Official website URL.")
    categories: Optional[List[str]] = Field(default=None, description="AI sector tags.")
    canonicalName: Optional[str] = Field(default=None, description="Resolved entity name.")
    entityResolutionMethod: Optional[str] = Field(default=None, description="exact|alias|compact|fuzzy|unresolved.")
    entityResolutionScore: Optional[float] = Field(default=None, description="Resolution confidence 0.0-1.0.")


class Product(BaseRecord):
    """AI software tool or product."""
    recordType: RecordType = Field(default=RecordType.PRODUCT)
    name: str = Field(..., description="Product name.")
    description: Optional[str] = Field(default=None, description="Product summary.")
    company: Optional[str] = Field(default=None, description="Maker or parent company.")
    launchDate: Optional[str] = Field(default=None, description="Launch date string.")
    pricingModel: Optional[PricingModel] = Field(default=None, description="Pricing tier.")
    website: Optional[str] = Field(default=None, description="Product link or home.")
    categories: Optional[List[str]] = Field(default=None, description="Tags/categories.")
    upvotes: Optional[int] = Field(default=None, description="Product Hunt upvotes or rating.")
    canonicalCompany: Optional[str] = Field(default=None, description="Resolved company name.")


class ResearchPaper(BaseRecord):
    """Academic AI research paper."""
    recordType: RecordType = Field(default=RecordType.PAPER)
    title: str = Field(..., description="Paper title.")
    authors: List[str] = Field(default_factory=list, description="Paper authors.")
    abstract: Optional[str] = Field(default=None, description="Abstract summary.")
    publishedAt: Optional[datetime] = Field(default=None, description="ISO-8601 UTC publish date.")
    arxivId: Optional[str] = Field(default=None, description="ArXiv ID (e.g. 2401.12345).")
    githubUrl: Optional[str] = Field(default=None, description="Linked GitHub code repository.")
    githubStars: Optional[int] = Field(default=None, description="Live GitHub stargazers count.")
    categories: Optional[List[str]] = Field(default=None, description="Categories e.g. cs.AI, cs.LG.")
    paperUrl: Optional[str] = Field(default=None, description="PDF or paper landing URL.")


class Job(BaseRecord):
    """Fresh AI engineering / research job posting."""
    recordType: RecordType = Field(default=RecordType.JOB)
    title: str = Field(..., description="Job role title.")
    company: str = Field(..., description="Hiring company.")
    location: Optional[str] = Field(default=None, description="Office location or Remote.")
    remote: Optional[bool] = Field(default=None, description="Whether remote is supported.")
    salaryMin: Optional[int] = Field(default=None, description="Minimum compensation.")
    salaryMax: Optional[int] = Field(default=None, description="Maximum compensation.")
    currency: Optional[str] = Field(default="USD", description="Currency symbol/code.")
    postedAt: Optional[datetime] = Field(default=None, description="ISO-8601 UTC post time.")
    applyUrl: Optional[str] = Field(default=None, description="Direct application link.")
    categories: Optional[List[str]] = Field(default=None, description="Job skill/discipline tags.")
    source: Optional[str] = Field(default=None, description="Source board name.")


class News(BaseRecord):
    """Fresh AI ecosystem news article."""
    recordType: RecordType = Field(default=RecordType.NEWS)
    title: str = Field(..., description="Article headline.")
    summary: Optional[str] = Field(default=None, description="Article summary / lead.")
    publishedAt: Optional[datetime] = Field(default=None, description="ISO-8601 UTC publish time.")
    author: Optional[str] = Field(default=None, description="Author byline.")
    source: Optional[str] = Field(default=None, description="Publisher name (e.g. TechCrunch).")
    categories: Optional[List[str]] = Field(default=None, description="Topic tags.")
    sentiment: Optional[str] = Field(default=None, description="Positive/Neutral/Negative.")


# ---------------------------------------------------------------------------
# Telemetry and Resolution Mapping
# ---------------------------------------------------------------------------

class EntityMapping(BaseModel):
    """Maps a raw entity name to its canonical form."""
    rawName: str
    canonicalName: str
    method: str = Field(..., description="Resolution method (exact, alias, compact, fuzzy, unresolved).")
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score.")
    createdAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class LLMCallLog(BaseModel):
    """Telemetry record for a single LLM API call."""
    tier: str = Field(..., description="Provider tier (Gemini Flash, Groq Llama 3, DeepSeek).")
    statusCode: Optional[int] = Field(default=None, description="HTTP response code.")
    retries: int = Field(default=0, description="Retry count.")
    tokensIn: Optional[int] = Field(default=None, description="Input tokens.")
    latencyMs: Optional[float] = Field(default=None, description="Latency in ms.")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    success: bool = Field(default=True, description="Call success status.")
