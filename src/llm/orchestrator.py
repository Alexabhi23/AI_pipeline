"""Production LLM Orchestrator with Multi-Tier Fallback, Rate Limiting, and Grounding.

Provides:
- CircuitBreaker: Failure tracking and circuit opening/resetting.
- TokenBucket: Async token bucket rate limiter (RPM/TPM).
- LLMTier: Provider abstraction for Gemini Flash, Groq Llama 3, and DeepSeek.
- LLMOrchestrator: Tier fallback, retry handling (429/413/5xx/timeout), Pydantic
  validation, grounding checks, and heuristic fallback for dry-run/keyless operation.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import structlog
try:
    from selectolax.lexbor import LexborHTMLParser as HTMLParser
except ImportError:
    try:
        from selectolax.parser import HTMLParser
    except ImportError:
        HTMLParser = None

from src.config import Settings
from src.llm.chunker import halve_budget, priority_chunk
from src.llm.grounding import ground_check
from src.models import (
    Job,
    LLMCallLog,
    News,
    PricingModel,
    Product,
    RecordType,
    ResearchPaper,
    Startup,
)

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# Default extraction system prompt
DEFAULT_SYSTEM_PROMPT: str = (
    "You are a senior data extraction engine. Return ONLY a valid JSON object matching "
    "the required schema. Return null for any field not explicitly present in the text. "
    "Never infer, fabricate, or hallucinate data."
)

# Schema guidance prompt for each record type
SCHEMA_DESCRIPTIONS: dict[RecordType, str] = {
    RecordType.STARTUP: (
        "Extract information about the startup mentioned in the text. Return a JSON object with:\n"
        "- name: (string, required) Name of the startup\n"
        "- description: (string or null) Short summary\n"
        "- founders: (list of strings or null) List of founder names\n"
        "- foundedDate: (string or null) Year or date founded\n"
        "- headquarters: (string or null) City, Country or headquarters location\n"
        "- employeeCount: (integer or null) Total employee headcount\n"
        "- funding: (string or null) Total funding amount or last round info\n"
        "- pricingModel: (string or null, must be one of 'FREE', 'FREEMIUM', 'PAID', 'ENTERPRISE')\n"
        "- website: (string or null) Official website URL\n"
        "- categories: (list of strings or null) Relevant AI sector categories or tags\n"
    ),
    RecordType.PRODUCT: (
        "Extract information about the AI product or software tool. Return a JSON object with:\n"
        "- name: (string, required) Name of the product\n"
        "- description: (string or null) Summary of the product\n"
        "- company: (string or null) Maker or company behind the product\n"
        "- launchDate: (string or null) Launch date or date string\n"
        "- pricingModel: (string or null, must be one of 'FREE', 'FREEMIUM', 'PAID', 'ENTERPRISE')\n"
        "- website: (string or null) Product URL or landing page\n"
        "- categories: (list of strings or null) Relevant product categories\n"
        "- upvotes: (integer or null) Upvotes or rating\n"
    ),
    RecordType.PAPER: (
        "Extract information about the academic research paper. Return a JSON object with:\n"
        "- title: (string, required) Full paper title\n"
        "- authors: (list of strings, required) Authors of the paper\n"
        "- abstract: (string or null) Paper abstract or summary\n"
        "- arxivId: (string or null) ArXiv identifier\n"
        "- githubUrl: (string or null) Code repository URL if mentioned\n"
        "- categories: (list of strings or null) Subject categories (e.g. cs.AI, cs.LG)\n"
        "- paperUrl: (string or null) Paper URL or PDF link\n"
    ),
    RecordType.JOB: (
        "Extract information about the job posting. Return a JSON object with:\n"
        "- title: (string, required) Job position or title\n"
        "- company: (string, required) Hiring company name\n"
        "- location: (string or null) Location or office\n"
        "- remote: (boolean or null) True if remote work is supported\n"
        "- salaryMin: (integer or null) Minimum compensation\n"
        "- salaryMax: (integer or null) Maximum compensation\n"
        "- currency: (string or null) Currency code, default 'USD'\n"
        "- applyUrl: (string or null) Direct application or posting link\n"
        "- categories: (list of strings or null) Skill tags or discipline categories\n"
        "- source: (string or null) Job board source name\n"
    ),
    RecordType.NEWS: (
        "Extract information about the news article. Return a JSON object with:\n"
        "- title: (string, required) Headline or article title\n"
        "- summary: (string or null) Summary or lead of the article\n"
        "- author: (string or null) Author name\n"
        "- source: (string or null) Publisher name\n"
        "- categories: (list of strings or null) Topic tags\n"
        "- sentiment: (string or null, 'Positive', 'Neutral', or 'Negative')\n"
    ),
}

# Required fields for grounding check per record type
REQUIRED_GROUNDING_FIELDS: dict[RecordType, list[str]] = {
    RecordType.STARTUP: ["name"],
    RecordType.PRODUCT: ["name"],
    RecordType.PAPER: ["title"],
    RecordType.JOB: ["title", "company"],
    RecordType.NEWS: ["title"],
}


class _CircuitBreakerBool(int):
    """Subclass of int/bool that supports both boolean evaluation and callable invocations."""

    def __new__(cls, val: bool) -> _CircuitBreakerBool:
        return super().__new__(cls, 1 if val else 0)

    def __call__(self) -> bool:
        return bool(self)


class CircuitBreaker:
    """Tracks provider errors and opens the circuit after repeated failures."""

    def __init__(self, failure_threshold: int = 5, reset_timeout: float = 60.0) -> None:
        self.failure_threshold: int = failure_threshold
        self.reset_timeout: float = reset_timeout
        self.failure_count: int = 0
        self.last_failure_time: float = 0.0

    @property
    def is_open(self) -> _CircuitBreakerBool:
        """Return True if circuit is open (failing), False if closed (healthy/probe)."""
        if self.failure_count >= self.failure_threshold:
            if time.time() - self.last_failure_time > self.reset_timeout:
                # Reset timeout elapsed: half-open probe allowed
                return _CircuitBreakerBool(False)
            return _CircuitBreakerBool(True)
        return _CircuitBreakerBool(False)

    def record_success(self) -> None:
        """Reset failure counter on successful request."""
        self.failure_count = 0
        self.last_failure_time = 0.0

    def record_failure(self) -> None:
        """Increment failure counter and record timestamp."""
        self.failure_count += 1
        self.last_failure_time = time.time()


class TokenBucket:
    """Asynchronous rate-limiting token bucket per provider (RPM / TPM)."""

    def __init__(self, rpm: int = 60, capacity: Optional[int] = None) -> None:
        self.rpm: int = rpm
        self.capacity: float = float(capacity if capacity is not None else rpm)
        self.tokens: float = float(self.capacity)
        self.fill_rate: float = float(rpm) / 60.0  # tokens per second
        self.last_refill: float = time.monotonic()
        self._lock: asyncio.Lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> bool:
        """Acquire tokens asynchronously, sleeping briefly if rate limit reached."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.fill_rate)
            self.last_refill = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True

            needed = tokens - self.tokens
            wait_time = needed / self.fill_rate
            # Do not block indefinitely if waiting would exceed pipeline tolerances
            if wait_time > 15.0:
                logger.warning(
                    "token_bucket.rate_limit_exceeded",
                    needed=needed,
                    wait_time=wait_time,
                )
                return False

            await asyncio.sleep(wait_time)
            self.tokens = 0.0
            self.last_refill = time.monotonic()
            return True


class LLMTier:
    """Encapsulates a specific LLM provider tier with rate limiting and circuit breaker."""

    def __init__(
        self,
        name: str,
        tier_number: int,
        model_name: str,
        api_key: Optional[str],
        rpm: int = 60,
        timeout: float = 30.0,
    ) -> None:
        self.name: str = name
        self.tier_number: int = tier_number
        self.model_name: str = model_name
        self.api_key: Optional[str] = api_key
        self.timeout: float = timeout
        self.enabled: bool = bool(api_key and api_key.strip())
        self.circuit_breaker: CircuitBreaker = CircuitBreaker(
            failure_threshold=5, reset_timeout=60.0
        )
        self.token_bucket: TokenBucket = TokenBucket(rpm=rpm)
        self._client: Any = None
        self._init_client()

    def _init_client(self) -> None:
        """Instantiate SDK client if API key is configured."""
        if not self.enabled:
            return

        try:
            if self.tier_number == 1:
                from google import genai

                self._client = genai.Client(api_key=self.api_key)
            elif self.tier_number == 2:
                from groq import AsyncGroq

                self._client = AsyncGroq(api_key=self.api_key)
            elif self.tier_number == 3:
                from openai import AsyncOpenAI

                self._client = AsyncOpenAI(
                    api_key=self.api_key, base_url="https://api.deepseek.com"
                )
        except Exception as exc:
            logger.error(
                "llm_tier.client_init_failed",
                tier=self.name,
                error=str(exc),
            )
            self.enabled = False

    async def call(
        self,
        prompt: str,
        system_prompt: str,
    ) -> tuple[str, Optional[int], float]:
        """Execute extraction call against this provider tier.

        Returns (response_text, tokens_in, latency_ms).
        """
        start_ts = time.monotonic()

        if self.tier_number == 1:
            # Gemini Flash via google-genai SDK
            from google.genai import types

            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                temperature=0.0,
            )
            response = await asyncio.wait_for(
                self._client.aio.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                ),
                timeout=self.timeout,
            )
            latency_ms = round((time.monotonic() - start_ts) * 1000, 2)
            tokens_in = None
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                tokens_in = getattr(response.usage_metadata, "prompt_token_count", None)
            return response.text or "", tokens_in, latency_ms

        elif self.tier_number == 2:
            # Groq Llama 3 via groq SDK
            response = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                ),
                timeout=self.timeout,
            )
            latency_ms = round((time.monotonic() - start_ts) * 1000, 2)
            content = response.choices[0].message.content or ""
            tokens_in = response.usage.prompt_tokens if response.usage else None
            return content, tokens_in, latency_ms

        elif self.tier_number == 3:
            # DeepSeek via openai SDK
            response = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                ),
                timeout=self.timeout,
            )
            latency_ms = round((time.monotonic() - start_ts) * 1000, 2)
            content = response.choices[0].message.content or ""
            tokens_in = response.usage.prompt_tokens if response.usage else None
            return content, tokens_in, latency_ms

        raise ValueError(f"Unknown tier number: {self.tier_number}")


class LLMOrchestrator:
    """Production LLM orchestration engine with 3-tier fallback and verification."""

    def __init__(self, settings: Settings) -> None:
        self.settings: Settings = settings
        self.timeout: float = float(settings.LLM_TIMEOUT)
        self.system_prompt: str = DEFAULT_SYSTEM_PROMPT

        # Tier 1: Gemini Flash (gemini-2.0-flash)
        self.tier1 = LLMTier(
            name="Gemini Flash",
            tier_number=1,
            model_name="gemini-2.0-flash",
            api_key=settings.GEMINI_API_KEY,
            rpm=120,
            timeout=self.timeout,
        )

        # Tier 2: Groq Llama 3 (llama-3.3-70b-versatile)
        self.tier2 = LLMTier(
            name="Groq Llama 3",
            tier_number=2,
            model_name="llama-3.3-70b-versatile",
            api_key=settings.GROQ_API_KEY,
            rpm=30,
            timeout=self.timeout,
        )

        # Tier 3: DeepSeek (deepseek-chat)
        self.tier3 = LLMTier(
            name="DeepSeek",
            tier_number=3,
            model_name="deepseek-chat",
            api_key=settings.DEEPSEEK_API_KEY,
            rpm=60,
            timeout=self.timeout,
        )

        self.tiers: list[LLMTier] = [self.tier1, self.tier2, self.tier3]
        enabled_count = sum(1 for t in self.tiers if t.enabled)
        logger.info(
            "llm_orchestrator.initialized",
            enabled_tiers=enabled_count,
            tier1=self.tier1.enabled,
            tier2=self.tier2.enabled,
            tier3=self.tier3.enabled,
        )

    # ------------------------------------------------------------------
    # JSON Cleaning & Validation Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_json_text(text: str) -> dict[str, Any]:
        """Strip markdown code-fences and extract valid JSON dictionary."""
        text = text.strip()
        # Remove ```json ... ``` wrapper
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if match:
            text = match.group(1).strip()

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        # Locate first '{' and last '}'
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            slice_text = text[start : end + 1]
            try:
                parsed = json.loads(slice_text)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

        raise ValueError(f"Could not parse valid JSON object from LLM response: {text[:200]}")

    def _validate_with_pydantic(
        self,
        target_type: RecordType,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate extracted fields against canonical Pydantic models."""
        dummy_base = {
            "sourceUrl": "https://graphone.internal/dummy",
            "urlHash": "0000000000000000",
            "contentHash": "0000000000000000",
        }

        model_cls: type[Any]
        if target_type == RecordType.STARTUP:
            model_cls = Startup
        elif target_type == RecordType.PRODUCT:
            model_cls = Product
        elif target_type == RecordType.PAPER:
            model_cls = ResearchPaper
        elif target_type == RecordType.JOB:
            model_cls = Job
        elif target_type == RecordType.NEWS:
            model_cls = News
        else:
            model_cls = Startup

        # Validate with dummy base attributes
        record_obj = model_cls(**dummy_base, **data)
        # Extract model fields
        cleaned: dict[str, Any] = {}
        for k in data.keys():
            if hasattr(record_obj, k):
                val = getattr(record_obj, k)
                if isinstance(val, PricingModel):
                    cleaned[k] = val.value
                else:
                    cleaned[k] = val
        return cleaned

    # ------------------------------------------------------------------
    # Heuristic Extractor (Dry-run / No LLM API Key Fallback)
    # ------------------------------------------------------------------

    def _heuristic_extract(
        self,
        raw_data: Optional[dict[str, Any]],
        record_type: RecordType,
        source_text: str,
    ) -> dict[str, Any]:
        """Extract structured record directly from raw attributes or HTML."""
        data = raw_data or {}
        extracted: dict[str, Any] = {}

        # Parse HTML if source text contains markup
        html_title = ""
        html_desc = ""
        if "<" in source_text and ">" in source_text:
            try:
                tree = HTMLParser(source_text)
                title_node = tree.css_first("title") or tree.css_first("h1")
                if title_node:
                    html_title = title_node.text(strip=True)
                desc_node = tree.css_first("meta[name='description']") or tree.css_first("p")
                if desc_node:
                    html_desc = (
                        desc_node.attributes.get("content", "")
                        if "content" in desc_node.attributes
                        else desc_node.text(strip=True)
                    )
            except Exception:
                pass

        if record_type == RecordType.STARTUP:
            name = (
                data.get("name")
                or data.get("title")
                or html_title
                or (source_text.splitlines()[0] if source_text else "AI Startup")
            )
            extracted["name"] = str(name).strip()
            extracted["description"] = data.get("description") or html_desc or None
            extracted["founders"] = data.get("founders") if isinstance(data.get("founders"), list) else None
            extracted["foundedDate"] = str(data.get("foundedDate") or data.get("founded_date") or "") or None
            extracted["headquarters"] = data.get("headquarters") or data.get("location") or None
            extracted["employeeCount"] = data.get("employeeCount") or data.get("employee_count") or None
            extracted["funding"] = data.get("funding") or None
            extracted["website"] = data.get("website") or data.get("url") or None
            extracted["categories"] = data.get("categories") or ["AI"]

        elif record_type == RecordType.PRODUCT:
            name = (
                data.get("name")
                or data.get("title")
                or html_title
                or "AI Tool"
            )
            extracted["name"] = str(name).strip()
            extracted["description"] = data.get("description") or html_desc or None
            extracted["company"] = data.get("company") or data.get("maker") or None
            extracted["website"] = data.get("website") or data.get("url") or None
            extracted["categories"] = data.get("categories") or ["Productivity"]
            extracted["upvotes"] = data.get("upvotes") or data.get("votes") or None

        elif record_type == RecordType.PAPER:
            title = data.get("title") or html_title or "AI Research Paper"
            extracted["title"] = str(title).strip()
            authors = data.get("authors")
            if isinstance(authors, list):
                extracted["authors"] = [str(a).strip() for a in authors if str(a).strip()]
            elif isinstance(authors, str):
                extracted["authors"] = [a.strip() for a in authors.split(",") if a.strip()]
            else:
                extracted["authors"] = []
            extracted["abstract"] = data.get("abstract") or data.get("summary") or html_desc or None
            extracted["arxivId"] = data.get("arxivId") or data.get("arxiv_id") or None
            extracted["githubUrl"] = data.get("githubUrl") or data.get("github_url") or None
            extracted["categories"] = data.get("categories") or ["cs.AI"]
            extracted["paperUrl"] = data.get("paperUrl") or data.get("url") or None

        elif record_type == RecordType.JOB:
            title = data.get("title") or html_title or "AI Engineer"
            company = data.get("company") or "Tech Corp"
            extracted["title"] = str(title).strip()
            extracted["company"] = str(company).strip()
            extracted["location"] = data.get("location") or None
            extracted["remote"] = bool(data.get("remote")) if "remote" in data else None
            extracted["salaryMin"] = data.get("salaryMin") or data.get("salary_min") or None
            extracted["salaryMax"] = data.get("salaryMax") or data.get("salary_max") or None
            extracted["currency"] = data.get("currency", "USD")
            extracted["applyUrl"] = data.get("applyUrl") or data.get("apply_url") or data.get("url") or None
            extracted["categories"] = data.get("categories") or ["Machine Learning"]

        elif record_type == RecordType.NEWS:
            title = data.get("title") or html_title or "AI Ecosystem Update"
            extracted["title"] = str(title).strip()
            extracted["summary"] = data.get("summary") or data.get("description") or html_desc or None
            extracted["author"] = data.get("author") or None
            extracted["source"] = data.get("source") or None
            extracted["categories"] = data.get("categories") or ["Industry"]

        return extracted

    # ------------------------------------------------------------------
    # Error classification helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_rate_limit(exc: Exception) -> tuple[bool, Optional[float]]:
        """Determine if exception is HTTP 429 and parse Retry-After header."""
        msg = str(exc).lower()
        if "429" in msg or "rate limit" in msg or "too many requests" in msg:
            # Check for Retry-After
            retry_after = None
            response = getattr(exc, "response", None)
            if response and hasattr(response, "headers"):
                val = response.headers.get("retry-after") or response.headers.get("Retry-After")
                if val:
                    try:
                        retry_after = float(val)
                    except ValueError:
                        pass
            return True, retry_after
        return False, None

    @staticmethod
    def _is_context_overflow(exc: Exception) -> bool:
        """Determine if exception indicates context size overflow / 413."""
        msg = str(exc).lower()
        return (
            "413" in msg
            or "context length" in msg
            or "context_length_exceeded" in msg
            or "maximum context length" in msg
            or "prompt is too long" in msg
            or "token limit" in msg
            or "too large" in msg
        )

    @staticmethod
    def _is_server_or_timeout(exc: Exception) -> bool:
        """Determine if exception is a 5xx server error or network timeout."""
        if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
            return True
        msg = str(exc).lower()
        return (
            "500" in msg
            or "502" in msg
            or "503" in msg
            or "504" in msg
            or "timeout" in msg
            or "service unavailable" in msg
            or "internal server error" in msg
        )

    # ------------------------------------------------------------------
    # Main Orchestration Extraction Method
    # ------------------------------------------------------------------

    async def extract(
        self,
        raw_data: Optional[dict[str, Any]] = None,
        record_type: Union[RecordType, str] = RecordType.STARTUP,
        source_text: str = "",
    ) -> tuple[Optional[dict[str, Any]], LLMCallLog]:
        """Extract structured record using LLM fallback chain.

        Fallback chain: Tier 1 (Gemini Flash) -> Tier 2 (Groq Llama 3) -> Tier 3 (DeepSeek).
        Gracefully falls back to heuristic extraction if no LLM keys are configured.
        Returns (grounded_record_dict_or_none, llm_call_log).
        """
        # Normalize record_type
        if isinstance(record_type, str):
            try:
                target_type = RecordType(record_type.upper())
            except ValueError:
                target_type = RecordType.STARTUP
        else:
            target_type = record_type

        # Ensure source text is populated
        if not source_text and raw_data:
            source_text = (
                raw_data.get("content")
                or raw_data.get("text")
                or raw_data.get("description")
                or raw_data.get("html")
                or str(raw_data)
            )

        # --------------------------------------------------------------
        # Fallback if no LLM API keys configured (Dry-run mode)
        # --------------------------------------------------------------
        active_tiers = [t for t in self.tiers if t.enabled]
        if not active_tiers:
            logger.warning(
                "llm_orchestrator.no_keys_configured",
                msg="No LLM API keys provided. Running heuristic extractor.",
            )
            raw_extracted = self._heuristic_extract(raw_data, target_type, source_text)
            try:
                validated = self._validate_with_pydantic(target_type, raw_extracted)
            except Exception as e:
                logger.warning(
                    "llm_orchestrator.heuristic_validation_failed",
                    error=str(e),
                )
                validated = raw_extracted

            # Grounding check
            req_fields = REQUIRED_GROUNDING_FIELDS.get(target_type, ["name"])
            grounded, warnings = ground_check(
                extracted=validated,
                source_text=source_text,
                required_fields=req_fields,
            )
            log = LLMCallLog(
                tier="Heuristic Fallback",
                statusCode=200,
                retries=0,
                tokensIn=0,
                latencyMs=0.0,
                success=True,
            )
            if not grounded:
                logger.warning(
                    "llm_orchestrator.heuristic_grounding_failed",
                    warnings=warnings,
                )
                return None, log
            return grounded, log

        # --------------------------------------------------------------
        # Multi-Tier Fallback Loop: Tier 1 -> Tier 2 -> Tier 3
        # --------------------------------------------------------------
        schema_desc = SCHEMA_DESCRIPTIONS.get(
            target_type, SCHEMA_DESCRIPTIONS[RecordType.STARTUP]
        )
        total_retries = 0

        for tier in active_tiers:
            # Check Circuit Breaker
            if tier.circuit_breaker.is_open:
                logger.warning(
                    "llm_orchestrator.circuit_open_skipping_tier",
                    tier=tier.name,
                    failure_count=tier.circuit_breaker.failure_count,
                )
                continue

            tier_retries = 0
            current_source = source_text

            while tier_retries < 3:
                # 1. Acquire TokenBucket capacity
                acquired = await tier.token_bucket.acquire(tokens=1)
                if not acquired:
                    logger.warning(
                        "llm_orchestrator.rate_limit_token_wait_failed",
                        tier=tier.name,
                    )
                    # Proceed or fall through
                    tier.circuit_breaker.record_failure()
                    break

                prompt = (
                    f"Target Schema:\n{schema_desc}\n\n"
                    f"Source Text:\n{current_source}\n\n"
                    f"Return ONLY valid JSON matching the schema above. Set unmentioned fields to null."
                )

                try:
                    logger.debug(
                        "llm_orchestrator.invoking_tier",
                        tier=tier.name,
                        attempt=tier_retries + 1,
                    )
                    raw_resp, tokens_in, latency_ms = await tier.call(
                        prompt=prompt,
                        system_prompt=self.system_prompt,
                    )

                    # Successful response: record success in circuit breaker
                    tier.circuit_breaker.record_success()

                    # Clean and parse JSON
                    parsed_dict = self._clean_json_text(raw_resp)

                    # Post-extraction Pydantic validation
                    validated_dict = self._validate_with_pydantic(target_type, parsed_dict)

                    # Grounding verification
                    req_fields = REQUIRED_GROUNDING_FIELDS.get(target_type, ["name"])
                    grounded_dict, warnings = ground_check(
                        extracted=validated_dict,
                        source_text=source_text,
                        required_fields=req_fields,
                    )

                    call_log = LLMCallLog(
                        tier=tier.name,
                        statusCode=200,
                        retries=tier_retries,
                        tokensIn=tokens_in,
                        latencyMs=latency_ms,
                        success=True,
                    )

                    if not grounded_dict:
                        logger.warning(
                            "llm_orchestrator.record_dropped_grounding_failed",
                            tier=tier.name,
                            warnings=warnings,
                        )
                        return None, call_log

                    logger.info(
                        "llm_orchestrator.extraction_succeeded",
                        tier=tier.name,
                        record_type=target_type.value,
                        latency_ms=latency_ms,
                    )
                    return grounded_dict, call_log

                except Exception as exc:
                    tier_retries += 1
                    total_retries += 1
                    logger.warning(
                        "llm_orchestrator.tier_call_failed",
                        tier=tier.name,
                        attempt=tier_retries,
                        error=str(exc),
                    )

                    # Handle 429: Exponential backoff with full jitter
                    is_rate, retry_after = self._is_rate_limit(exc)
                    if is_rate:
                        if retry_after and retry_after > 0:
                            sleep_dur = min(15.0, retry_after)
                        else:
                            sleep_dur = min(10.0, (1.5 ** tier_retries) + random.uniform(0.1, 0.9))
                        logger.warning(
                            "llm_orchestrator.rate_limit_backoff",
                            tier=tier.name,
                            sleep_dur=round(sleep_dur, 2),
                        )
                        await asyncio.sleep(sleep_dur)
                        continue

                    # Handle 413 / Context Overflow
                    if self._is_context_overflow(exc):
                        logger.warning(
                            "llm_orchestrator.context_overflow_detected",
                            tier=tier.name,
                        )
                        # Priority chunking first
                        current_source = priority_chunk(current_source, max_tokens=2048)
                        try:
                            # Immediate retry with chunked text
                            raw_resp, tokens_in, latency_ms = await tier.call(
                                prompt=f"Target Schema:\n{schema_desc}\n\nSource Text:\n{current_source}\n\nReturn JSON:",
                                system_prompt=self.system_prompt,
                            )
                            tier.circuit_breaker.record_success()
                            parsed_dict = self._clean_json_text(raw_resp)
                            validated_dict = self._validate_with_pydantic(target_type, parsed_dict)
                            req_fields = REQUIRED_GROUNDING_FIELDS.get(target_type, ["name"])
                            grounded_dict, warnings = ground_check(
                                extracted=validated_dict,
                                source_text=source_text,
                                required_fields=req_fields,
                            )
                            call_log = LLMCallLog(
                                tier=tier.name,
                                statusCode=200,
                                retries=tier_retries,
                                tokensIn=tokens_in,
                                latencyMs=latency_ms,
                                success=True,
                            )
                            if not grounded_dict:
                                return None, call_log
                            return grounded_dict, call_log
                        except Exception as inner_exc:
                            if self._is_context_overflow(inner_exc):
                                # Halve budget and retry once more on same tier
                                logger.warning(
                                    "llm_orchestrator.halving_budget_retry",
                                    tier=tier.name,
                                )
                                current_source = halve_budget(current_source, current_budget=2048)
                                try:
                                    raw_resp, tokens_in, latency_ms = await tier.call(
                                        prompt=f"Target Schema:\n{schema_desc}\n\nSource Text:\n{current_source}\n\nReturn JSON:",
                                        system_prompt=self.system_prompt,
                                    )
                                    tier.circuit_breaker.record_success()
                                    parsed_dict = self._clean_json_text(raw_resp)
                                    validated_dict = self._validate_with_pydantic(target_type, parsed_dict)
                                    req_fields = REQUIRED_GROUNDING_FIELDS.get(target_type, ["name"])
                                    grounded_dict, warnings = ground_check(
                                        extracted=validated_dict,
                                        source_text=source_text,
                                        required_fields=req_fields,
                                    )
                                    call_log = LLMCallLog(
                                        tier=tier.name,
                                        statusCode=200,
                                        retries=tier_retries,
                                        tokensIn=tokens_in,
                                        latencyMs=latency_ms,
                                        success=True,
                                    )
                                    if not grounded_dict:
                                        return None, call_log
                                    return grounded_dict, call_log
                                except Exception:
                                    pass
                        # Exhausted context handling for this tier
                        tier.circuit_breaker.record_failure()
                        break

                    # Handle 5xx / Timeout: Retry with jitter
                    if self._is_server_or_timeout(exc):
                        sleep_dur = 1.0 + random.uniform(0.1, 0.8)
                        logger.warning(
                            "llm_orchestrator.server_or_timeout_retry",
                            tier=tier.name,
                            sleep_dur=round(sleep_dur, 2),
                        )
                        await asyncio.sleep(sleep_dur)
                        continue

                    # Unhandled failure on this tier
                    tier.circuit_breaker.record_failure()
                    break

            # If all retries exhausted on this tier, record failure and fall through
            tier.circuit_breaker.record_failure()
            logger.warning(
                "llm_orchestrator.tier_exhausted_falling_through",
                exhausted_tier=tier.name,
            )

        # All tiers exhausted
        logger.error(
            "llm_orchestrator.all_tiers_failed",
            total_retries=total_retries,
        )
        failure_log = LLMCallLog(
            tier="All Tiers Failed",
            statusCode=500,
            retries=total_retries,
            tokensIn=0,
            latencyMs=0.0,
            success=False,
        )
        return None, failure_log
