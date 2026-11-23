# GraphOne / FrontierAtlas AI Engineer Demo Task: Pipeline Architecture

## 1. Executive Summary & Design Tenets

GraphOne / FrontierAtlas ingests intelligence across the global AI ecosystem (startups, products, foundational research papers, talent movements, and market developments). This system is designed as a **zero-hallucination, high-throughput, config-driven ingestion pipeline** engineered to scale from initial seed runs to **500,000+ entities** across thousands of domains.

### Key Architectural Tenets
1. **Zero Hallucination via Mechanical Grounding**: LLMs are used strictly as semantic extractors under schema constraints. Every string output must be grounded via fuzzy/exact matching against raw HTTP 200 source text; ungrounded strings are nullified. If a mandatory attribute fails grounding, the record is dropped.
2. **Deterministic Entity Resolution**: AI entity resolution follows a strict 4-tier pipeline: Case/Suffix Normalization $\to$ Curated Alias Index $\to$ Compact Spacing Match $\to$ Levenshtein/TokenSort $\ge 92\%$. Ambiguous entities are cleanly flagged as `unresolved` rather than guessed.
3. **Defense-in-Depth Reliability**: Rate limiting uses per-domain semaphores and token buckets, with circuit breakers across an automatic 3-tier fallback LLM hierarchy (Gemini Flash $\to$ Groq Llama 3 $\to$ DeepSeek).
4. **Config-Driven Architecture**: Adding or modifying sources requires only editing `sources.yaml` without changing application code.

---

## 2. End-to-End System Topology

```
┌────────────────────────────────────────────────────────────────────────┐
│                        SOURCES.YAML CONFIGURATION                      │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │
           ┌─────────────────────────┼─────────────────────────┐
           ▼                         ▼                         ▼
   ┌──────────────┐          ┌──────────────┐          ┌──────────────┐
   │ Papers (API) │          │ News / Jobs  │          │ Directories  │
   │ arXiv / HF   │          │ RSS / JSON   │          │ YC / PH      │
   └───────┬──────┘          └───────┬──────┘          └───────┬──────┘
           │                         │                         │
           └─────────────────────────┼─────────────────────────┘
                                     ▼
           ┌───────────────────────────────────────────────────┐
           │            ASYNC HTTP CORE & DISCOVERY            │
           │  • Bounded Concurrency (Global Semaphore)         │
           │  • Per-Domain Token Bucket & robots.txt Cache     │
           │  • Conditional Requests (ETag / If-Modified-Since)│
           │  • Playwright Async Fallback (JS/Cloudflare)      │
           └─────────────────────────┬─────────────────────────┘
                                     ▼
           ┌───────────────────────────────────────────────────┐
           │              STATEFUL DEDUPLICATION               │
           │  • URL Hash (xxHash64) + Content Hash Deduplication│
           │  • Redis SETNX (Cluster) / SQLite (Single-Node)   │
           │  • Provenance Tracking: First-Seen & Last-Seen    │
           └─────────────────────────┬─────────────────────────┘
                                     ▼
           ┌───────────────────────────────────────────────────┐
           │               DATE EXTRACTION ENGINE              │
           │  • Meta tags / JSON-LD / <time> / HTTP Headers    │
           │  • 24h Strict Freshness Filter (News & Jobs)      │
           │  • Date Provenance & Confidence Categorization    │
           └─────────────────────────┬─────────────────────────┘
                                     ▼
           ┌───────────────────────────────────────────────────┐
           │              LLM ORCHESTRATION LAYER              │
           │  • Tier 1: Gemini Flash (Fast, High TPM)          │
           │  • Tier 2: Groq Llama 3.3 70B (Fast Fallback)     │
           │  • Tier 3: DeepSeek (Cost-Effective Backup)       │
           │  • Circuit Breaker + Token Bucket Rate Limiting   │
           │  • Priority Chunking & 413 Halving Protocol       │
           └─────────────────────────┬─────────────────────────┘
                                     ▼
           ┌───────────────────────────────────────────────────┐
           │            GROUNDING & ENTITY RESOLVER            │
           │  • Strict Grounding Verification (Fuzzy >= 85%)   │
           │  • 4-Tier Entity Resolver against Seed Database   │
           │  • GitHub Live Stargazer Enrichment               │
           └─────────────────────────┬─────────────────────────┘
                                     ▼
           ┌───────────────────────────────────────────────────┐
           │               STORAGE & EXPORT LAYER              │
           │  • Production: PostgreSQL + pgvector (Embeddings) │
           │  • Trial / Seed: SQLite State Store               │
           │  • Google Sheets (6 Tabs) & CSV Exporter          │
           └───────────────────────────────────────────────────┘
```

---

## 3. Scale Strategy (500k+ Records)

To scale to 500,000+ records and thousands of sources without memory bloat or bottlenecking:

### 3.1 Distributed Stateless Worker Pool
- **Architecture**: Decouple discovery from extraction. Frontier tasks are queued into a Redis/Celery queue.
- **Domain Sharding**: URLs are partitioned by domain hash across worker nodes. This guarantees that per-domain rate limits (e.g. 1 req/sec for arXiv) are enforced globally without inter-worker lock contention.
- **Node Elasticity**: Workers are completely stateless. Workers pull batches, process through the LLM pipeline, and write straight to database storage.

### 3.2 Storage Architecture: Postgres + pgvector
- **Core Partitioning**: The `records` table is horizontally partitioned by `record_type` (`STARTUP`, `PRODUCT`, `PAPER`, `JOB`, `NEWS`).
- **Semantic Vector Search**: High-dimensional embeddings (768d or 1536d) generated via Gemini or open-source embeddings are indexed using HNSW on `pgvector`, enabling cross-document similarity search (e.g. matching an AI paper to relevant startups).
- **Knowledge Graph Representation**: Entity relationships (`Company -> Produces -> Product`, `Paper -> AuthoredBy -> Researcher`, `Company -> Hires -> Job`) are modeled via relational join tables or indexed in Neo4j for graph traversals.

---

## 4. Reliability at Scale: 413 and 429 Protocol

High-concurrency LLM extraction encounters rate limits (429) and context overflows (413). The pipeline handles both deterministically:

### 4.1 429 (Rate Limit / Quota Exhaustion) Handling
1. **Token Bucket Rate Limiter**: Every LLM provider has an async token bucket calibrated to its RPM/TPM quota. Requests pause before dispatching rather than spamming the API.
2. **Exponential Backoff with Full Jitter**: On 429 receipt, the worker calculates delay via:
   $$t_{\text{sleep}} = \min(t_{\text{max}}, \text{Retry-After} \parallel (2^{\text{retry}} \times \text{Uniform}(0.5, 1.5)))$$
3. **Circuit Breaker**: If 5 consecutive failures occur within 60 seconds on any tier, the tier's circuit breaker trips to `OPEN`. All incoming requests immediately bypass this tier to the next tier without incurring latency penalties.

### 4.2 413 / Context Overflow Handling: Priority Chunking
Raw scraped web pages can exceed LLM context windows or token budgets:
1. **Priority Hierarchy**: Content is parsed into structural components:
   $$\text{Title} > \text{Headings} (H1-H3) > \text{Lead Paragraphs} > \text{Body Content} > \text{Tables} > \text{Footers/Navigation}$$
2. **Dynamic Halving**: When an LLM returns a 413 or token count exceeds the tier limit, `halve_budget()` drops low-priority elements (footers, disclaimers, extraneous body paragraphs) and retries on the same tier once before cascading to the next provider tier.

---

## 5. Zero-Hallucination & Entity Resolution Rigor

### 5.1 Verification and Grounding
- Traditional LLM extraction risks fabrication. GraphOne implements mechanical validation:
  1. Extracted JSON is coerced to strict Pydantic v2 domain schemas.
  2. For every scalar string value (e.g., founder name, company headquarters, pricing tier), the engine executes normalized fuzzy token matching against the HTTP 200 source text using `rapidfuzz.fuzz.partial_ratio`.
  3. If $\text{score} < 85\%$, the field is forcibly overwritten with `None` and an audit reason is logged.
  4. Non-LLM attributes (dates, GitHub stars, HTTP status, URLs) are parsed purely through deterministic code parsers.

### 5.2 Deterministic Entity Resolution
- Company naming variations across directories are mapped to canonical entities:
  - **Stage 1 (Exact)**: `openai` $\to$ `OpenAI` (1.0 confidence)
  - **Stage 2 (Alias Table)**: `OpenAI Inc.`, `OpenAI LLC` $\to$ `OpenAI` (1.0 confidence)
  - **Stage 3 (Compact Form)**: `open ai` $\to$ `OpenAI` (0.98 confidence)
  - **Stage 4 (Fuzzy Match)**: RapidFuzz token sort $\ge 92\% \to$ matched canonical name
  - **Stage 5 (Fallback)**: Flagged as `unresolved` with original raw string preserved.
