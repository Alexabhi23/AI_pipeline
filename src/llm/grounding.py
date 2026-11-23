"""Grounding check for LLM-extracted structured data.

Verifies that every string value returned by the LLM actually appears
(or fuzzy-matches) in the source text.  Values that cannot be grounded
are nullified so downstream consumers never receive hallucinated data.

Uses ``rapidfuzz.fuzz.partial_ratio`` for approximate matching.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

import structlog
from rapidfuzz.fuzz import partial_ratio

logger = structlog.get_logger(__name__)

# Minimum fuzzy-match score (0–100) to consider a value grounded.
_GROUNDING_THRESHOLD: float = 85.0


def _normalise(text: str) -> str:
    """Lower-case, strip, collapse whitespace and remove punctuation."""
    text = unicodedata.normalize("NFKD", text)
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _is_grounded(value: str, source_norm: str) -> tuple[bool, float]:
    """Return ``(grounded, score)`` for a single string *value*."""
    val_norm = _normalise(value)
    if not val_norm:
        return True, 100.0  # Empty strings are trivially grounded.

    # Fast exact substring check first.
    if val_norm in source_norm:
        return True, 100.0

    score: float = partial_ratio(val_norm, source_norm)
    return score >= _GROUNDING_THRESHOLD, score


def _walk_and_ground(
    obj: Any,
    source_norm: str,
    path: str,
    warnings: list[str],
) -> Any:
    """Recursively walk *obj*, nullifying ungrounded string leaves.

    Returns a (possibly modified) copy of *obj*.
    """
    if isinstance(obj, dict):
        grounded: dict[str, Any] = {}
        for key, value in obj.items():
            grounded[key] = _walk_and_ground(
                value, source_norm, f"{path}.{key}", warnings
            )
        return grounded

    if isinstance(obj, list):
        return [
            _walk_and_ground(item, source_norm, f"{path}[{i}]", warnings)
            for i, item in enumerate(obj)
        ]

    if isinstance(obj, str):
        ok, score = _is_grounded(obj, source_norm)
        if not ok:
            reason = (
                f"Field '{path}' value '{obj[:80]}' not found in source "
                f"(best partial_ratio={score:.1f}, threshold={_GROUNDING_THRESHOLD})"
            )
            logger.warning("grounding.nullified", path=path, score=score, value=obj[:80])
            warnings.append(reason)
            return None
        return obj

    # Non-string primitives (int, float, bool, None) pass through.
    return obj


def ground_check(
    extracted: dict[str, Any],
    source_text: str,
    required_fields: list[str],
) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate that extracted values are grounded in *source_text*.

    Parameters
    ----------
    extracted:
        The structured dict returned by the LLM.
    source_text:
        The original text (or HTML-stripped text) the LLM was asked to
        extract from.
    required_fields:
        Top-level field names that **must** survive grounding for the
        record to be considered valid.

    Returns
    -------
    tuple[dict | None, list[str]]
        ``(grounded_dict, warnings)`` – if any *required_fields* are
        ``None`` after grounding the first element is ``None``.
    """
    if not extracted:
        return None, ["extracted dict is empty or None"]

    source_norm = _normalise(source_text)
    if not source_norm:
        return None, ["source_text is empty after normalisation"]

    warnings: list[str] = []
    grounded = _walk_and_ground(extracted, source_norm, "$", warnings)

    # Check required fields.
    missing: list[str] = []
    for field_name in required_fields:
        value = grounded.get(field_name)
        if value is None:
            missing.append(field_name)

    if missing:
        reason = f"Required fields missing after grounding: {missing}"
        logger.error("grounding.required_fields_missing", fields=missing)
        warnings.append(reason)
        return None, warnings

    logger.info(
        "grounding.passed",
        warnings_count=len(warnings),
        required_ok=True,
    )
    return grounded, warnings
