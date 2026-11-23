# GraphOne / FrontierAtlas — AI Engineer Demo Task

Production-grade, zero-hallucination async data ingestion pipeline engineered for global AI ecosystem intelligence (startups, products, foundational research papers, talent movements, and market developments).

---

## Architecture Highlights

1. **Deterministic Entity Resolution**:
   - 4-tier waterfall: `Exact Normalize` $\to$ `Curated Alias Table` $\to$ `Compact Spacing ("open ai" = "openai")` $\to$ `RapidFuzz Token Sort (>= 92%)`.
   - Seed database contains 50 real AI frontier startups with realistic aliases.
   - Ambiguous or unseen entities cleanly tagged as `unresolved` without hallucinated mappings.

2. **Zero-Hallucination Mechanical Grounding**:
   - LLMs act strictly as semantic extractors using structured Pydantic v2 schemas.
   - Prompt explicitly commands: *"Return null for any field not present in the text. Never infer."*
   - Every string field undergoes post-extraction grounding via `rapidfuzz.fuzz.partial_ratio` against raw HTTP 200 text ($\ge 85\%$). Ungrounded strings are nullified. If required fields fail grounding, the record is dropped.

3. **Multi-Tier LLM Fallback Chain with Circuit Breakers**:
   - Fallback hierarchy: **Tier 1 (Gemini Flash)** $\to$ **Tier 2 (Groq Llama 3.3 70B)** $\to$ **Tier 3 (DeepSeek Chat)**.
   - Per-provider token-bucket rate limiter and 5-failure circuit breaker.
   - 429 backoff: Exponential backoff + full jitter, respecting `Retry-After`.
   - 413 context overflow: Priority chunking (`Title` > `Headings` > `Lead` > `Body`), budget halving before tier escalation.

4. **Robust Async Crawler Core**:
   - HTTP core built on `aiohttp` with bounded concurrency semaphore and per-domain token bucket.
   - Cached `robots.txt` enforcement and conditional HTTP requests (`ETag` / `If-Modified-Since`).
   - Playwright async fallback strictly for JavaScript-rendered and Cloudflare-protected targets.

5. **Cross-Run & Cross-Node Deduplication**:
   - `xxhash64(url)` and `xxhash64(content)` stored in Redis (`SETNX`) or local SQLite.
   - First-seen / last-seen timestamps and date provenance (`meta`, `relative`, `heuristic`, `api`).

---

## Repository Structure

```
graphone-pipeline/
├── .env.example              # Configuration template
├── requirements.txt          # Python dependencies
├── sources.yaml              # Config-driven sources (15+ sources)
├── architecture.md           # 3-page scale and system design document
├── generate_pdf.py           # Architecture PDF generator
├── data/
│   ├── seed_startups.json    # 50 real AI startups + alias table
│   └── exports/              # CSV and JSON exports
├── src/
│   ├── config.py             # Pydantic Settings & environment loader
│   ├── models.py             # Pydantic v2 schemas & enum definitions
│   ├── main.py               # Central async pipeline CLI runner
│   ├── crawler/
│   │   ├── http.py           # Async HTTP client, semaphore, robots.txt, ETag
│   │   ├── browser.py        # Playwright async fallback
│   │   ├── dates.py          # Multi-source date extraction & freshness filter
│   │   └── sources/          # Base adapter & concrete source crawlers
│   │       ├── arxiv.py      # arXiv API scraper & paper parser
│   │       ├── hf_papers.py  # Hugging Face papers adapter
│   │       ├── github_stars.py# GitHub live stars enricher
│   │       ├── yc.py         # Y Combinator AI companies directory
│   │       ├── producthunt.py# Product Hunt AI products
│   │       ├── rss_news.py   # 5 AI RSS news feeds (TechCrunch, Verge, etc.)
│   │       └── jobs.py       # Multi-board AI jobs scraper
│   ├── llm/
│   │   ├── orchestrator.py   # 3-tier fallback, circuit breaker, token bucket
│   │   ├── chunker.py        # Priority chunking & 413 budget halver
│   │   └── grounding.py      # RapidFuzz fuzzy grounding check
│   ├── resolver/
│   │   └── resolver.py       # 4-tier deterministic entity resolver
│   ├── storage/
│   │   ├── seen.py           # Deduplication store (Redis / SQLite)
│   │   └── database.py       # SQLite state store (6 tables)
│   └── export/
│       ├── sheets.py         # Google Sheets 6-tab automated exporter
│       └── csv_export.py     # Local multi-file CSV exporter
└── tests/
    ├── conftest.py           # Pytest shared fixtures
    ├── test_dates.py         # Date parsing & relative date tests
    ├── test_resolver.py      # Entity resolution test suite
    ├── test_grounding.py     # Grounding validation tests
    ├── test_models.py        # Pydantic model validation tests
    └── validate_output.py    # Invariant and data quality verification script
```

---

## Quick Start & Installation

### 1. Environment Setup

```bash
# Clone and enter directory
cd graphone-pipeline

# Create and activate Python virtual environment (Python 3.11 recommended)
python -m venv venv
# Linux / macOS:
source venv/bin/activate
# Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Install Playwright browser binaries (for JS fallback)
playwright install chromium
```

### 2. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` to configure your API keys:
- `GEMINI_API_KEY`: Google Gemini Flash API key.
- `GROQ_API_KEY`: Groq Cloud API key.
- `DEEPSEEK_API_KEY`: DeepSeek API key.
- `GOOGLE_SHEETS_CREDS_PATH`: Optional path to GCP service account JSON.
- `SPREADSHEET_ID`: Optional Google Spreadsheet ID.

*(Note: If no LLM keys are supplied, the pipeline safely falls back to direct heuristic parsing for demonstration).*

---

## Execution Commands

### Run Full Ingestion Pipeline
```bash
python -m src.main --phase all
```

### Run Specific Ingestion Phase
```bash
# Papers only (arXiv & Hugging Face + GitHub Stars)
python -m src.main --phase papers

# AI Startups only (Y Combinator)
python -m src.main --phase startups

# AI Products only (Product Hunt)
python -m src.main --phase products

# Fresh News only (RSS feeds within 24h)
python -m src.main --phase news

# Fresh Jobs only (RemoteOK, Remotive, Greenhouse, Lever within 24h)
python -m src.main --phase jobs
```

### Export Options
```bash
# Export to Google Sheets (creates 6 tabs & prints public link)
python -m src.main --phase all --export sheets

# Export to CSV (writes files to data/exports/)
python -m src.main --phase all --export csv

# Export to both
python -m src.main --phase all --export both
```

### Run Automated Quality & Invariant Validation
```bash
python -m tests.validate_output
```

### Run Unit Test Suite
```bash
pytest tests/ -v
```

### Generate Architecture PDF
```bash
python generate_pdf.py
```

---

## State Schema Reference

| Table | Primary Key | Key Columns | Purpose |
|---|---|---|---|
| `seen_items` | `url_hash` | `content_hash, source, record_type, first_seen, last_seen, status` | Cross-run dedup & freshness tracking |
| `raw_pages` | `url_hash` | `url, http_status, fetched_at, html_path` | Source traceability & HTTP 200 audit |
| `records` | `id` | `record_type, url_hash, payload_json, date_source, date_confidence, collected_at` | Final validated structured intelligence |
| `entity_map` | `id` | `raw_name, canonical_name, method, score, created_at` | Entity resolution audit log |
| `llm_calls` | `id` | `tier, status_code, retries, tokens_in, latency_ms, ts, success` | LLM orchestration telemetry |
| `seed_entities` | `canonical` | `aliases_json` | Seed startup knowledge base |

---

## Known Limitations & Design Trade-offs

| Domain | Limitation | Production Mitigation |
|---|---|---|
| **JavaScript Rendering** | Headless Playwright incurs high memory overhead per concurrent tab. | Isolate browser instances to a dedicated headless rendering microservice (e.g. Browserless). |
| **Local Storage** | SQLite is single-node and lacks native concurrent vector search. | Migrate to PostgreSQL with `pgvector` and Redis Sentinel / Cluster in production. |
| **API Quotas** | Unauthenticated GitHub API allows 60 req/hr for live star lookups. | Inject `GITHUB_TOKEN` in `.env` to raise limit to 5,000 req/hr. |
| **Anti-Bot Defense** | Cloudflare Turnstile blocks automated scrapers. | Respect `robots.txt`, utilize official RSS/APIs, and configure residential proxy rotation. |
