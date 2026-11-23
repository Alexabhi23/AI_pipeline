"""Unit tests for date extraction engine."""

from datetime import datetime, timezone, timedelta
import pytest
from src.crawler.dates import extract_published, is_fresh, compute_content_hash, compute_url_hash
from src.models import DateSource


def test_meta_tag_date():
    html = """
    <html>
      <head>
        <meta property="article:published_time" content="2026-11-20T14:30:00Z" />
      </head>
      <body><h1>AI Model Release</h1></body>
    </html>
    """
    dt, source = extract_published(html, {}, "https://example.com/post")
    assert dt is not None
    assert dt.year == 2026
    assert source in (DateSource.META, "meta")


def test_json_ld_date():
    html = """
    <html>
      <head>
        <script type="application/ld+json">
        {
          "@context": "https://schema.org",
          "@type": "NewsArticle",
          "headline": "Breakthrough in AI",
          "datePublished": "2026-11-21T09:00:00Z"
        }
        </script>
      </head>
      <body>Content</body>
    </html>
    """
    dt, source = extract_published(html, {}, "https://example.com/news")
    assert dt is not None
    assert dt.day == 21
    assert source in (DateSource.META, "meta")


def test_time_element_date():
    html = """
    <html>
      <body>
        <article>
          <time datetime="2026-11-22T08:00:00Z">Yesterday</time>
          <p>Text here</p>
        </article>
      </body>
    </html>
    """
    dt, source = extract_published(html, {}, "https://example.com/article")
    assert dt is not None
    assert dt.day == 22


def test_http_last_modified_header():
    html = "<html><body>No dates inside</body></html>"
    headers = {"Last-Modified": "Sat, 22 Nov 2026 12:00:00 GMT"}
    dt, source = extract_published(html, headers, "https://example.com/page")
    assert dt is not None
    assert dt.year == 2026


def test_freshness_filter():
    now = datetime.now(timezone.utc)
    fresh_time = now - timedelta(hours=5)
    stale_time = now - timedelta(hours=36)

    assert is_fresh(fresh_time, hours=24) is True
    assert is_fresh(stale_time, hours=24) is False
    assert is_fresh(None, hours=24) is False


def test_hashing_consistency():
    url1 = "https://example.com/page?utm_source=twitter"
    url2 = "https://example.com/page"
    # Canonical hash strips tracking queries
    h1 = compute_url_hash(url1)
    h2 = compute_url_hash(url2)
    assert len(h1) > 0
    assert len(h2) > 0

    c_hash = compute_content_hash("Some test text")
    assert len(c_hash) > 0
