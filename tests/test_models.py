"""Unit tests for Pydantic v2 data models in src.models."""

from datetime import datetime, timezone
import pytest
from src.models import (
    BaseRecord,
    DateConfidence,
    DateSource,
    Job,
    News,
    PricingModel,
    Product,
    RecordType,
    ResearchPaper,
    Startup,
)


def test_startup_model(sample_startup_payload, sample_now):
    startup = Startup(
        sourceUrl="https://example.com/anthropic",
        urlHash="hash_123",
        contentHash="chash_456",
        dateSource=DateSource.META,
        dateConfidence=DateConfidence.HIGH,
        **sample_startup_payload,
    )
    assert startup.recordType == RecordType.STARTUP
    assert startup.name == "Anthropic"
    assert startup.employeeCount == 500
    assert startup.pricingModel == PricingModel.PAID
    assert "Dario Amodei" in startup.founders

    # Test DB row serialization
    row = startup.to_db_row()
    assert row["record_type"] == "STARTUP"
    assert row["url_hash"] == "hash_123"
    assert "Anthropic" in row["payload_json"]


def test_product_model(sample_product_payload):
    product = Product(
        sourceUrl="https://example.com/claude",
        urlHash="h1",
        contentHash="c1",
        **sample_product_payload,
    )
    assert product.recordType == RecordType.PRODUCT
    assert product.name == "Claude 3.5 Sonnet"
    assert product.upvotes == 1250


def test_paper_model(sample_paper_payload):
    paper = ResearchPaper(
        sourceUrl="https://arxiv.org/abs/1706.03762",
        urlHash="h2",
        contentHash="c2",
        publishedAt=datetime(2017, 6, 12, tzinfo=timezone.utc),
        **sample_paper_payload,
    )
    assert paper.recordType == RecordType.PAPER
    assert paper.arxivId == "1706.03762"
    assert paper.githubStars == 14500
    assert isinstance(paper.githubStars, int)


def test_job_model():
    job = Job(
        sourceUrl="https://jobs.example.com/1",
        urlHash="h3",
        contentHash="c3",
        title="Senior AI Research Engineer",
        company="OpenAI",
        remote=True,
        salaryMin=250000,
        salaryMax=350000,
        currency="USD",
        postedAt=datetime.now(timezone.utc),
    )
    assert job.recordType == RecordType.JOB
    assert job.remote is True
    assert job.salaryMax == 350000


def test_news_model():
    news = News(
        sourceUrl="https://techcrunch.com/article/ai",
        urlHash="h4",
        contentHash="c4",
        title="Next Generation Foundation Model Released",
        summary="A new open-weights model sets benchmarks.",
        publishedAt=datetime.now(timezone.utc),
        source="TechCrunch",
    )
    assert news.recordType == RecordType.NEWS
    assert news.source == "TechCrunch"
