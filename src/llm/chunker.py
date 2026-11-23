"""Priority-based text truncation for LLM context windows.

Parses HTML content and extracts sections by priority, truncating
lower-priority sections first when the token budget is exceeded.
Uses trafilatura for main content extraction and selectolax for
structural parsing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import structlog
import trafilatura
try:
    from selectolax.lexbor import LexborHTMLParser as HTMLParser
except ImportError:
    from selectolax.parser import HTMLParser

logger = structlog.get_logger(__name__)

# Approximate token-to-character ratio (GPT-family heuristic).
CHARS_PER_TOKEN: int = 4

# Priority levels – lower number == higher priority.
PRIORITY_TITLE: int = 0
PRIORITY_HEADINGS: int = 1
PRIORITY_LEAD: int = 2
PRIORITY_BODY: int = 3
PRIORITY_FOOTER: int = 4


@dataclass(slots=True)
class _Section:
    """A prioritised slice of the source document."""

    priority: int
    label: str
    text: str


def _estimate_tokens(text: str) -> int:
    """Return a rough token count for *text*."""
    return max(1, len(text) // CHARS_PER_TOKEN)


def _extract_sections(html: str) -> list[_Section]:
    """Parse *html* and return document sections ordered by priority."""
    sections: list[_Section] = []

    tree = HTMLParser(html)

    # --- title -------------------------------------------------------
    title_node = tree.css_first("title")
    if title_node and title_node.text(strip=True):
        sections.append(_Section(PRIORITY_TITLE, "title", title_node.text(strip=True)))

    # Also consider <h1> as title-level if no <title>.
    if not sections:
        h1 = tree.css_first("h1")
        if h1 and h1.text(strip=True):
            sections.append(_Section(PRIORITY_TITLE, "h1-title", h1.text(strip=True)))

    # --- headings (h2–h6) -------------------------------------------
    heading_parts: list[str] = []
    for tag in ("h2", "h3", "h4", "h5", "h6"):
        for node in tree.css(tag):
            txt = node.text(strip=True)
            if txt:
                heading_parts.append(txt)
    if heading_parts:
        sections.append(
            _Section(PRIORITY_HEADINGS, "headings", "\n".join(heading_parts))
        )

    # --- lead paragraph ----------------------------------------------
    first_p = tree.css_first("article p, main p, .content p, p")
    if first_p and first_p.text(strip=True):
        sections.append(_Section(PRIORITY_LEAD, "lead", first_p.text(strip=True)))

    # --- body (trafilatura main content) -----------------------------
    body_text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        deduplicate=True,
        favor_recall=True,
    )
    if body_text:
        sections.append(_Section(PRIORITY_BODY, "body", body_text))

    # --- footer ------------------------------------------------------
    footer_node = tree.css_first("footer")
    if footer_node and footer_node.text(strip=True):
        sections.append(
            _Section(PRIORITY_FOOTER, "footer", footer_node.text(strip=True))
        )

    # Sort by priority (ascending = highest priority first).
    sections.sort(key=lambda s: s.priority)
    return sections


def _truncate_text(text: str, max_chars: int) -> str:
    """Truncate *text* to at most *max_chars*, breaking at a word boundary."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    # Try to break at the last whitespace.
    last_space = truncated.rfind(" ")
    if last_space > max_chars // 2:
        truncated = truncated[:last_space]
    return truncated.rstrip() + " …"


def priority_chunk(html: str, max_tokens: int) -> str:
    """Return text extracted from *html* fitting within *max_tokens*.

    Sections are ordered by editorial priority:
        title > headings > lead paragraph > body > footer

    When the budget is exceeded the lowest-priority sections are dropped
    first; the body section is tail-truncated before being removed
    entirely.

    Parameters
    ----------
    html:
        Raw HTML string.
    max_tokens:
        Maximum number of (approximate) tokens for the returned text.

    Returns
    -------
    str
        Truncated plain-text suitable for an LLM prompt.
    """
    if not html or not html.strip():
        logger.warning("priority_chunk.empty_input")
        return ""

    sections = _extract_sections(html)
    if not sections:
        # Fallback: strip tags naively.
        fallback = re.sub(r"<[^>]+>", " ", html)
        fallback = re.sub(r"\s+", " ", fallback).strip()
        return _truncate_text(fallback, max_tokens * CHARS_PER_TOKEN)

    max_chars = max_tokens * CHARS_PER_TOKEN

    # Accumulate sections highest-priority first, dropping from the tail.
    result_parts: list[str] = []
    chars_used = 0

    for section in sections:
        section_len = len(section.text)
        remaining = max_chars - chars_used

        if remaining <= 0:
            logger.debug(
                "priority_chunk.section_dropped",
                section=section.label,
                reason="budget_exhausted",
            )
            break

        if section_len <= remaining:
            result_parts.append(section.text)
            chars_used += section_len
        else:
            # Partial fit – truncate the section.
            result_parts.append(_truncate_text(section.text, remaining))
            chars_used += remaining
            logger.debug(
                "priority_chunk.section_truncated",
                section=section.label,
                kept_chars=remaining,
                total_chars=section_len,
            )
            break

    combined = "\n\n".join(result_parts)
    logger.info(
        "priority_chunk.done",
        sections_kept=len(result_parts),
        estimated_tokens=_estimate_tokens(combined),
        max_tokens=max_tokens,
    )
    return combined


def halve_budget(text: str, current_budget: int) -> str:
    """Re-chunk *text* (treated as HTML or plain-text) at half the budget.

    Parameters
    ----------
    text:
        Source HTML **or** plain-text.  If it looks like plain-text the
        function wraps it in a minimal ``<body>`` so the section parser
        can still operate.
    current_budget:
        The token budget that proved too large.

    Returns
    -------
    str
        Truncated text fitting within ``current_budget // 2`` tokens.
    """
    new_budget = max(64, current_budget // 2)
    logger.info(
        "halve_budget",
        old_budget=current_budget,
        new_budget=new_budget,
    )

    # Wrap plain text so the HTML parser has something to work with.
    if "<" not in text:
        text = f"<body><p>{text}</p></body>"

    return priority_chunk(text, new_budget)
