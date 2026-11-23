"""Entity resolution module for the GraphOne/FrontierAtlas pipeline.

Provides :class:`EntityResolver` which normalises raw company names and
maps them to canonical entries from a curated seed-startups catalogue
using a multi-stage resolution chain (exact → alias → compact → fuzzy).
"""

from __future__ import annotations

import json
import re
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import structlog
from rapidfuzz import fuzz

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SUFFIX_RE: Final[re.Pattern[str]] = re.compile(
    r",?\s*\b(inc\.?|llc\.?|ltd\.?|corp\.?|co\.?|gmbh|b\.?v\.?|sas|pbc|plc|limited|corporation|incorporated)\b\.?",
    re.IGNORECASE,
)

_PUNCT_TABLE: Final[dict[int, None]] = str.maketrans("", "", string.punctuation)

_FUZZY_THRESHOLD: Final[float] = 92.0

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EntityMapping:
    """Immutable record of a single resolution decision."""

    raw_name: str
    canonical_name: str
    method: str
    score: float


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


class EntityResolver:
    """Resolve raw company names to canonical forms.

    Resolution chain (stops at first match):

    1. **Exact** – normalised input matches a normalised canonical name
       ➜ ``(canonical, "exact", 1.0)``
    2. **Alias** – normalised input matches a normalised alias
       ➜ ``(canonical, "alias", 1.0)``
    3. **Compact** – spaces removed from both sides match
       ➜ ``(canonical, "compact", 0.98)``
    4. **Fuzzy** – ``rapidfuzz.fuzz.token_sort_ratio >= 92``
       ➜ ``(canonical, "fuzzy", score / 100)``
    5. **Unresolved** – nothing matched
       ➜ ``(raw_name, "unresolved", 0.0)``
    """

    # ------------------------------------------------------------------ init

    def __init__(self, seed_path: str | Path | None = None) -> None:
        """Load seed startups and build internal lookup indices.

        Parameters
        ----------
        seed_path:
            Path to the ``seed_startups.json`` file.  When *None* the
            resolver looks for it at ``<repo>/data/seed_startups.json``.
        """
        if seed_path is None:
            seed_path = Path(__file__).resolve().parents[2] / "data" / "seed_startups.json"
        else:
            seed_path = Path(seed_path)

        self._seed_path: Path = seed_path

        # Canonical names keyed by their normalised form.
        self._norm_to_canonical: dict[str, str] = {}
        # Alias (normalised) → canonical name.
        self._alias_to_canonical: dict[str, str] = {}
        # Compact (no-space normalised) → canonical name.
        self._compact_to_canonical: dict[str, str] = {}
        # Ordered list of (normalised canonical, canonical) for fuzzy scan.
        self._fuzzy_candidates: list[tuple[str, str]] = []

        # Public log of every resolution decision.
        self.mapping_log: list[EntityMapping] = []

        self._load_seed_data()
        logger.info(
            "entity_resolver.initialised",
            seed_path=str(self._seed_path),
            canonical_count=len(self._norm_to_canonical),
            alias_count=len(self._alias_to_canonical),
        )

    # -------------------------------------------------------------- public API

    def resolve(self, raw_name: str) -> tuple[str, str, float]:
        """Resolve *raw_name* to a canonical company name.

        Returns
        -------
        tuple[str, str, float]
            ``(canonical_name, method, score)`` where *method* is one of
            ``"exact"``, ``"alias"``, ``"compact"``, ``"fuzzy"``, or
            ``"unresolved"`` and *score* is a confidence in ``[0, 1]``.
        """
        normed = self._normalise(raw_name)

        # 1. Exact match -------------------------------------------------
        canonical = self._norm_to_canonical.get(normed)
        if canonical is not None:
            self.add_mapping(raw_name, canonical, "exact", 1.0)
            return canonical, "exact", 1.0

        # 2. Alias table --------------------------------------------------
        canonical = self._alias_to_canonical.get(normed)
        if canonical is not None:
            self.add_mapping(raw_name, canonical, "alias", 1.0)
            return canonical, "alias", 1.0

        # 3. Compact form (remove all spaces) -----------------------------
        compact = normed.replace(" ", "")
        canonical = self._compact_to_canonical.get(compact)
        if canonical is not None:
            self.add_mapping(raw_name, canonical, "compact", 0.98)
            return canonical, "compact", 0.98

        # 4. Fuzzy matching ------------------------------------------------
        best_score: float = 0.0
        best_canonical: str | None = None
        for cand_norm, cand_canonical in self._fuzzy_candidates:
            score = fuzz.token_sort_ratio(normed, cand_norm)
            if score > best_score:
                best_score = score
                best_canonical = cand_canonical

        if best_score >= _FUZZY_THRESHOLD and best_canonical is not None:
            final_score = round(best_score / 100.0, 4)
            self.add_mapping(raw_name, best_canonical, "fuzzy", final_score)
            return best_canonical, "fuzzy", final_score

        # 5. Unresolved ----------------------------------------------------
        self.add_mapping(raw_name, raw_name, "unresolved", 0.0)
        logger.warning(
            "entity_resolver.unresolved",
            raw_name=raw_name,
            normalised=normed,
        )
        return raw_name, "unresolved", 0.0

    def add_mapping(
        self,
        raw: str,
        canonical: str,
        method: str,
        score: float,
    ) -> None:
        """Record a resolution mapping and log it via *structlog*.

        Parameters
        ----------
        raw:
            The original (un-normalised) input string.
        canonical:
            The resolved canonical company name.
        method:
            Resolution method used (``"exact"``, ``"alias"``, etc.).
        score:
            Confidence score in ``[0, 1]``.
        """
        mapping = EntityMapping(
            raw_name=raw,
            canonical_name=canonical,
            method=method,
            score=score,
        )
        self.mapping_log.append(mapping)
        logger.info(
            "entity_resolver.mapped",
            raw=raw,
            canonical=canonical,
            method=method,
            score=score,
        )

    # -------------------------------------------------------------- internals

    @staticmethod
    def _normalise(name: str) -> str:
        """Normalise a company name for comparison.

        Steps:
        1. Lowercase
        2. Strip leading/trailing whitespace
        3. Remove corporate suffixes (Inc, LLC, Ltd, Corp, …)
        4. Remove punctuation
        5. Collapse multiple spaces to one and strip again
        """
        name = name.lower().strip()
        name = _SUFFIX_RE.sub("", name)
        name = name.translate(_PUNCT_TABLE)
        name = re.sub(r"\s+", " ", name).strip()
        return name

    def _load_seed_data(self) -> None:
        """Parse ``seed_startups.json`` and build lookup indices."""
        if not self._seed_path.exists():
            logger.error(
                "entity_resolver.seed_file_missing",
                path=str(self._seed_path),
            )
            raise FileNotFoundError(
                f"Seed startups file not found: {self._seed_path}"
            )

        with self._seed_path.open("r", encoding="utf-8") as fh:
            entries: list[dict[str, object]] = json.load(fh)

        for entry in entries:
            canonical = str(entry["canonical"])
            aliases: list[str] = [str(a) for a in entry.get("aliases", [])]

            norm_canonical = self._normalise(canonical)

            # Exact index
            self._norm_to_canonical[norm_canonical] = canonical

            # Compact index for canonical
            compact_canonical = norm_canonical.replace(" ", "")
            self._compact_to_canonical[compact_canonical] = canonical

            # Alias index
            for alias in aliases:
                norm_alias = self._normalise(alias)
                self._alias_to_canonical[norm_alias] = canonical
                # Also add compact form for each alias
                compact_alias = norm_alias.replace(" ", "")
                self._compact_to_canonical[compact_alias] = canonical

            # Fuzzy candidates – canonical + all aliases
            self._fuzzy_candidates.append((norm_canonical, canonical))
            for alias in aliases:
                self._fuzzy_candidates.append(
                    (self._normalise(alias), canonical)
                )
