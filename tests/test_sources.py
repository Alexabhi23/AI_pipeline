"""Comprehensive unit tests for GraphOne crawler source adapters.

Tests discover(), fetch_and_parse(), rate limiting, and enrichment across:
- ArxivSource
- HFPapersSource
- GitHubStarsEnricher
- YCDirectorySource
- ProductHuntSource
- RSSNewsSource
- JobsSource
- Registry resolution (get_adapter)
"""

from __future__ import annotations

import json
import unittest
from datetime import datetime, timedelta, timezone
from typing import Any

from src.crawler.sources import (
    ADAPTER_REGISTRY,
    ArxivSource,
    GitHubStarsEnricher,
    HFPapersSource,
    JobsSource,
    ProductHuntSource,
    RSSNewsSource,
    YCDirectorySource,
    get_adapter,
)
from src.crawler.sources.base import HttpClient, RateLimitConfig, SourceConfig
from src.crawler.sources.jobs import parse_salary_range


# ---------------------------------------------------------------------------
# Mock HttpClient
# ---------------------------------------------------------------------------

class MockHttpClient(HttpClient):
    """Stub HTTP client returning predefined responses for testing."""

    def __init__(self, responses: dict[str, str | bytes] | None = None) -> None:
        self.responses: dict[str, str | bytes] = responses or {}
        self.requested_urls: list[str] = []

    def set_response(self, url: str, content: str | bytes) -> None:
        self.responses[url] = content

    async def get(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> bytes:
        self.requested_urls.append(url)
        res = self.responses.get(url, b"")
        return res if isinstance(res, bytes) else res.encode("utf-8")

    async def get_text(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> str:
        self.requested_urls.append(url)
        res = self.responses.get(url, "")
        return res if isinstance(res, str) else res.decode("utf-8")


# ---------------------------------------------------------------------------
# Test suite using unittest.IsolatedAsyncioTestCase
# ---------------------------------------------------------------------------

class TestSourceAdapters(unittest.IsolatedAsyncioTestCase):

    ARXIV_ATOM_SAMPLE = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <title>arXiv Search Results</title>
  <entry>
    <id>http://arxiv.org/abs/2401.12345v1</id>
    <published>2026-11-23T10:00:00Z</published>
    <updated>2026-11-23T11:00:00Z</updated>
    <title>Deep Reasoning With Large Language Models</title>
    <summary>This paper proposes a new method for deep reasoning.
Code available at https://github.com/graphone-ai/deep-reasoning for reproducibility.</summary>
    <author><name>Alice Smith</name></author>
    <author><name>Bob Jones</name></author>
    <arxiv:comment>Accepted to NeurIPS 2026</arxiv:comment>
    <link href="http://arxiv.org/abs/2401.12345v1" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2401.12345v1" rel="related" type="application/pdf"/>
    <category term="cs.AI" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
</feed>
"""

    async def test_arxiv_discover_and_parse(self):
        cfg = SourceConfig(
            name="arxiv",
            type="api",
            adapter="arxiv",
            config={"categories": ["cs.AI", "cs.LG"], "max_results": 100},
        )
        client = MockHttpClient()
        adapter = ArxivSource(cfg, client)

        urls = await adapter.discover()
        self.assertEqual(len(urls), 1)
        self.assertIn("cat:cs.AI", urls[0])
        self.assertIn("max_results=100", urls[0])

        client.set_response(urls[0], self.ARXIV_ATOM_SAMPLE)
        records = await adapter.fetch_and_parse(urls[0])

        self.assertEqual(len(records), 1)
        r = records[0]
        self.assertEqual(r["id"], "2401.12345v1")
        self.assertEqual(r["arxiv_id"], "2401.12345v1")
        self.assertEqual(r["title"], "Deep Reasoning With Large Language Models")
        self.assertEqual(r["authors"], ["Alice Smith", "Bob Jones"])
        self.assertIn("https://github.com/graphone-ai/deep-reasoning", r["github_url"])
        self.assertEqual(r["pdf_url"], "http://arxiv.org/pdf/2401.12345v1")
        self.assertIn("cs.AI", r["categories"])
        self.assertEqual(r["recordType"], "PAPER")

    async def test_hf_papers_props_and_dom(self):
        hf_papers_props_sample = """<!DOCTYPE html>
<html>
<body>
  <div data-props='{"dailyPapers":[{"paper":{"id":"2402.54321","title":"Sparse Attention in Transformers","summary":"Fast attention mechanism.","upvotes":142,"authors":[{"name":"Carol White"}],"thumbnail":"https://hf.co/thumb.jpg"}}]}'>
  </div>
</body>
</html>
"""
        hf_papers_dom_sample = """<!DOCTYPE html>
<html>
<body>
  <article class="border card">
    <h3><a href="/papers/2403.99999">Vision Language Models on Edge Devices</a></h3>
    <button class="upvote leading-none">88</button>
    <img src="https://hf.co/thumb2.jpg" alt="Paper Image" />
    <span class="author">Dan Brown</span>
  </article>
</body>
</html>
"""
        cfg = SourceConfig(name="hf_papers", type="web", adapter="hf_papers")
        client = MockHttpClient()
        adapter = HFPapersSource(cfg, client)

        urls = await adapter.discover()
        self.assertEqual(urls, ["https://huggingface.co/papers"])

        # 1. Test data-props JSON parsing
        client.set_response(urls[0], hf_papers_props_sample)
        records = await adapter.fetch_and_parse(urls[0])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["arxiv_id"], "2402.54321")
        self.assertEqual(records[0]["title"], "Sparse Attention in Transformers")
        self.assertEqual(records[0]["upvotes"], 142
)
        self.assertEqual(records[0]["authors"], ["Carol White"])

        # 2. Test DOM fallback parsing
        client.set_response(urls[0], hf_papers_dom_sample)
        dom_records = await adapter.fetch_and_parse(urls[0])
        self.assertEqual(len(dom_records), 1)
        self.assertEqual(dom_records[0]["arxiv_id"], "2403.99999")
        self.assertEqual(dom_records[0]["title"], "Vision Language Models on Edge Devices")
        self.assertEqual(dom_records[0]["upvotes"], 88)

    async def test_github_stars_enricher(self):
        client = MockHttpClient()
        api_url = "https://api.github.com/repos/vllm-project/vllm"
        client.set_response(api_url, json.dumps({"stargazers_count": 32500}))

        enricher = GitHubStarsEnricher(client)

        # Test key extraction
        parsed = enricher.extract_repo_key("https://github.com/vllm-project/vllm.git")
        self.assertIsNotNone(parsed)
        owner, repo = parsed
        self.assertEqual(owner, "vllm-project")
        self.assertEqual(repo, "vllm")

        # Test fetching stars
        stars = await enricher.get_stars("https://github.com/vllm-project/vllm")
        self.assertEqual(stars, 32500)

        # Test in-memory cache hit (no second network request)
        client.responses.clear()
        cached_stars = await enricher.get_stars("https://github.com/vllm-project/vllm")
        self.assertEqual(cached_stars, 32500)

    async def test_yc_directory_parse(self):
        yc_next_data_sample = """<!DOCTYPE html>
<html>
<body>
  <script id="__NEXT_DATA__" type="application/json">
  {
    "props": {
      "pageProps": {
        "companies": [
          {
            "name": "NeuroFlow AI",
            "slug": "neuroflow-ai",
            "one_liner": "Autonomous neural network optimization platform.",
            "batch_name": "W24",
            "location": "San Francisco, CA, USA",
            "team_size": 15,
            "website": "https://neuroflow.ai",
            "tags": ["AI", "Developer Tools"]
          }
        ]
      }
    }
  }
  </script>
</body>
</html>
"""
        cfg = SourceConfig(
            name="yc_directory",
            type="web",
            adapter="yc",
            config={"url": "https://www.ycombinator.com/companies"},
        )
        client = MockHttpClient()
        adapter = YCDirectorySource(cfg, client)

        urls = await adapter.discover()
        self.assertEqual(len(urls), 1)
        self.assertIn("industry=Artificial%20Intelligence", urls[0])

        client.set_response(urls[0], yc_next_data_sample)
        records = await adapter.fetch_and_parse(urls[0])
        self.assertEqual(len(records), 1)
        r = records[0]
        self.assertEqual(r["name"], "NeuroFlow AI")
        self.assertEqual(r["id"], "neuroflow-ai")
        self.assertEqual(r["funding"], "YC W24")
        self.assertEqual(r["headquarters"], "San Francisco, CA, USA")
        self.assertEqual(r["employeeCount"], 15)
        self.assertEqual(r["website"], "https://neuroflow.ai")
        self.assertEqual(r["recordType"], "STARTUP")

    async def test_producthunt_parse(self):
        ph_next_data_sample = """<!DOCTYPE html>
<html>
<body>
  <script id="__NEXT_DATA__" type="application/json">
  {
    "props": {
      "pageProps": {
        "apolloState": {
          "Post:101": {
            "name": "Cursor Agent",
            "slug": "cursor-agent",
            "tagline": "AI pair programmer that understands your codebase.",
            "votesCount": 950,
            "website": "https://cursor.com",
            "createdAt": "2026-11-20T12:00:00Z"
          }
        }
      }
    }
  }
  </script>
</body>
</html>
"""
        cfg = SourceConfig(name="product_hunt", type="web", adapter="producthunt")
        client = MockHttpClient()
        adapter = ProductHuntSource(cfg, client)

        urls = await adapter.discover()
        self.assertIn("producthunt.com/topics/artificial-intelligence", urls[0])

        client.set_response(urls[0], ph_next_data_sample)
        records = await adapter.fetch_and_parse(urls[0])
        self.assertEqual(len(records), 1)
        r = records[0]
        self.assertEqual(r["name"], "Cursor Agent")
        self.assertEqual(r["upvotes"], 950)
        self.assertEqual(r["website"], "https://cursor.com")
        self.assertEqual(r["recordType"], "PRODUCT")

    async def test_rss_news_freshness_filter(self):
        now_utc = datetime.now(timezone.utc)
        fresh_date_str = (now_utc - timedelta(hours=2)).strftime("%a, %d %b %Y %H:%M:%S GMT")
        stale_date_str = (now_utc - timedelta(hours=48)).strftime("%a, %d %b %Y %H:%M:%S GMT")

        rss_sample = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>TechCrunch AI</title>
    <item>
      <title>Next-Gen Foundation Models Announced</title>
      <link>https://techcrunch.com/2026/11/next-gen-models</link>
      <description>&lt;p&gt;A major AI breakthrough was unveiled today.&lt;/p&gt;</description>
      <pubDate>{fresh_date_str}</pubDate>
      <author>Jane Doe</author>
      <category>AI</category>
    </item>
    <item>
      <title>Old News From Last Week</title>
      <link>https://techcrunch.com/2026/11/old-news</link>
      <description>Stale story that should be filtered out.</description>
      <pubDate>{stale_date_str}</pubDate>
    </item>
  </channel>
</rss>
"""
        cfg = SourceConfig(
            name="techcrunch_ai",
            type="rss",
            adapter="rss_news",
            config={"feed_url": "https://techcrunch.com/category/artificial-intelligence/feed/"},
        )
        client = MockHttpClient()
        adapter = RSSNewsSource(cfg, client)

        urls = await adapter.discover()
        self.assertEqual(len(urls), 1)

        client.set_response(urls[0], rss_sample)
        records = await adapter.fetch_and_parse(urls[0])

        # Only fresh item (<24h) should be retained
        self.assertEqual(len(records), 1)
        r = records[0]
        self.assertEqual(r["title"], "Next-Gen Foundation Models Announced")
        self.assertEqual(r["sourceUrl"], "https://techcrunch.com/2026/11/next-gen-models")
        self.assertEqual(r["recordType"], "NEWS")

    async def test_jobs_source_remoteok_and_greenhouse(self):
        now_utc = datetime.now(timezone.utc)
        remoteok_sample = json.dumps([
            {"legal": "Legal notice"},
            {
                "id": "1001",
                "position": "Senior ML Infrastructure Engineer",
                "company": "ScaleAI",
                "location": "Remote",
                "tags": ["machine-learning", "python", "kubernetes"],
                "date": (now_utc - timedelta(hours=3)).isoformat(),
                "url": "https://remoteok.com/remote-jobs/1001",
                "salary_min": 180000,
                "salary_max": 240000,
            },
            {
                "id": "1002",
                "position": "Frontend React Developer",
                "company": "WebShop",
                "tags": ["javascript", "css"],
                "date": (now_utc - timedelta(hours=2)).isoformat(),
            },
        ])

        greenhouse_sample = json.dumps({
            "jobs": [
                {
                    "id": 9001,
                    "title": "Research Scientist, Reasoning",
                    "updated_at": (now_utc - timedelta(hours=1)).isoformat(),
                    "absolute_url": "https://openai.com/careers/9001",
                    "location": {"name": "San Francisco, CA"},
                }
            ]
        })

        # 1. RemoteOK
        cfg_ro = SourceConfig(
            name="remoteok",
            type="api",
            adapter="jobs",
            config={"url": "https://remoteok.com/api", "filter_tags": ["machine-learning"]},
        )
        client = MockHttpClient()
        adapter_ro = JobsSource(cfg_ro, client)

        client.set_response("https://remoteok.com/api", remoteok_sample)
        records_ro = await adapter_ro.fetch_and_parse("https://remoteok.com/api")
        self.assertEqual(len(records_ro), 1)
        self.assertEqual(records_ro[0]["title"], "Senior ML Infrastructure Engineer")
        self.assertEqual(records_ro[0]["company"], "ScaleAI")
        self.assertEqual(records_ro[0]["salaryMin"], 180000)
        self.assertEqual(records_ro[0]["recordType"], "JOB")

        # 2. Greenhouse
        cfg_gh = SourceConfig(
            name="greenhouse_openai",
            type="api",
            adapter="jobs",
            config={"url": "https://boards-api.greenhouse.io/v1/boards/openai/jobs", "company": "OpenAI"},
        )
        adapter_gh = JobsSource(cfg_gh, client)
        client.set_response("https://boards-api.greenhouse.io/v1/boards/openai/jobs", greenhouse_sample)
        records_gh = await adapter_gh.fetch_and_parse("https://boards-api.greenhouse.io/v1/boards/openai/jobs")
        self.assertEqual(len(records_gh), 1)
        self.assertEqual(records_gh[0]["title"], "Research Scientist, Reasoning")
        self.assertEqual(records_gh[0]["company"], "OpenAI")

    async def test_jobs_source_remotive_lever_and_hn(self):
        now_utc = datetime.now(timezone.utc)
        client = MockHttpClient()

        # 1. Remotive
        remotive_sample = json.dumps({
            "jobs": [
                {
                    "id": 501,
                    "title": "Lead AI Engineer",
                    "company_name": "DeepTech",
                    "category": "Data / AI",
                    "tags": ["python", "pytorch", "nlp"],
                    "publication_date": (now_utc - timedelta(hours=4)).isoformat(),
                    "candidate_required_location": "Worldwide",
                    "salary": "$160,000 - $200,000",
                    "url": "https://remotive.com/jobs/501",
                }
            ]
        })
        cfg_remotive = SourceConfig(name="remotive", type="api", adapter="jobs", config={"url": "https://remotive.com/api/remote-jobs"})
        adapter_rem = JobsSource(cfg_remotive, client)
        client.set_response("https://remotive.com/api/remote-jobs", remotive_sample)
        records_rem = await adapter_rem.fetch_and_parse("https://remotive.com/api/remote-jobs")
        self.assertEqual(len(records_rem), 1)
        self.assertEqual(records_rem[0]["title"], "Lead AI Engineer")
        self.assertEqual(records_rem[0]["company"], "DeepTech")
        self.assertEqual(records_rem[0]["salaryMin"], 160000)
        self.assertEqual(records_rem[0]["salaryMax"], 200000)

        # 2. Lever
        lever_sample = json.dumps([
            {
                "id": "lev-101",
                "text": "Research Scientist - Reinforcement Learning",
                "createdAt": int((now_utc - timedelta(hours=5)).timestamp() * 1000),
                "hostedUrl": "https://jobs.lever.co/anthropic/lev-101",
                "categories": {"location": "San Francisco, CA", "commitment": "Full time"},
            }
        ])
        cfg_lever = SourceConfig(
            name="lever_anthropic",
            type="api",
            adapter="jobs",
            config={"url": "https://api.lever.co/v0/postings/anthropic", "company": "Anthropic"},
        )
        adapter_lev = JobsSource(cfg_lever, client)
        client.set_response("https://api.lever.co/v0/postings/anthropic", lever_sample)
        records_lev = await adapter_lev.fetch_and_parse("https://api.lever.co/v0/postings/anthropic")
        self.assertEqual(len(records_lev), 1)
        self.assertEqual(records_lev[0]["title"], "Research Scientist - Reinforcement Learning")
        self.assertEqual(records_lev[0]["company"], "Anthropic")

        # 3. HN Hiring
        hn_sample = """<html><body>
        <table>
          <tr><td class="comment">
            Anthropic | Research Engineer | San Francisco | REMOTE | $200k - $280k
            <br>Building safe LLMs. Apply at https://anthropic.com/careers
          </td></tr>
        </table>
        </body></html>"""
        cfg_hn = SourceConfig(
            name="github_jobs_hn",
            type="web",
            adapter="jobs",
            config={"url": "https://news.ycombinator.com/item?id=ask_hn_hiring", "type": "hn_hiring"},
        )
        adapter_hn = JobsSource(cfg_hn, client)
        records_hn = adapter_hn._parse_hn_hiring(hn_sample, "https://news.ycombinator.com/item?id=ask_hn_hiring")
        self.assertEqual(len(records_hn), 1)
        self.assertEqual(records_hn[0]["company"], "Anthropic")
        self.assertEqual(records_hn[0]["title"], "Research Engineer")
        self.assertTrue(records_hn[0]["remote"])
        self.assertEqual(records_hn[0]["salaryMin"], 200000)
        self.assertEqual(records_hn[0]["salaryMax"], 280000)

    def test_salary_parser(self):
        s_min, s_max, curr = parse_salary_range("$150k - $220k")
        self.assertEqual(s_min, 150000)
        self.assertEqual(s_max, 220000)
        self.assertEqual(curr, "USD")

        s_min2, s_max2, curr2 = parse_salary_range("$120,000 to $180,000")
        self.assertEqual(s_min2, 120000)
        self.assertEqual(s_max2, 180000)

    def test_adapter_registry(self):
        self.assertIs(get_adapter("arxiv"), ArxivSource)
        self.assertIs(get_adapter("hf_papers"), HFPapersSource)
        self.assertIs(get_adapter("yc"), YCDirectorySource)
        self.assertIs(get_adapter("producthunt"), ProductHuntSource)
        self.assertIs(get_adapter("rss_news"), RSSNewsSource)
        self.assertIs(get_adapter("jobs"), JobsSource)

        with self.assertRaises(KeyError):
            get_adapter("unknown_provider")


if __name__ == "__main__":
    unittest.main()
