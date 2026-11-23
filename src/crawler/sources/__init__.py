"""Source adapters registry and exports for the GraphOne crawler pipeline.

Exposes concrete adapters for arXiv, Hugging Face papers, Y Combinator directory,
Product Hunt, RSS feeds, and multi-format job boards, alongside the GitHubStarsEnricher
and adapter resolution helper `get_adapter`.
"""

from __future__ import annotations

from typing import Type

from src.crawler.sources.arxiv import ArxivSource
from src.crawler.sources.base import BaseSource, HttpClient, RateLimitConfig, SourceConfig, load_sources
from src.crawler.sources.github_stars import GitHubStarsEnricher
from src.crawler.sources.hf_papers import HFPapersSource
from src.crawler.sources.jobs import JobsSource
from src.crawler.sources.producthunt import ProductHuntSource
from src.crawler.sources.rss_news import RSSNewsSource
from src.crawler.sources.yc import YCDirectorySource

# Adapter registry mapping configuration adapter key to concrete adapter class
ADAPTER_REGISTRY: dict[str, type[BaseSource]] = {
    "arxiv": ArxivSource,
    "hf_papers": HFPapersSource,
    "yc": YCDirectorySource,
    "producthunt": ProductHuntSource,
    "rss_news": RSSNewsSource,
    "jobs": JobsSource,
}


def get_adapter(adapter_name: str) -> type[BaseSource]:
    """Retrieve the source adapter class mapped to the specified adapter name.

    Args:
        adapter_name: The adapter identifier configured in sources.yaml
                      (e.g., 'arxiv', 'hf_papers', 'yc', 'producthunt', 'rss_news', 'jobs').

    Returns:
        type[BaseSource]: Concrete adapter class.

    Raises:
        KeyError: If adapter_name is not registered.
    """
    cleaned = adapter_name.strip().lower()
    if cleaned not in ADAPTER_REGISTRY:
        available = ", ".join(sorted(ADAPTER_REGISTRY.keys()))
        raise KeyError(
            f"Unknown source adapter '{adapter_name}'. Registered adapters are: {available}"
        )
    return ADAPTER_REGISTRY[cleaned]


__all__ = [
    "ADAPTER_REGISTRY",
    "ArxivSource",
    "BaseSource",
    "GitHubStarsEnricher",
    "HFPapersSource",
    "HttpClient",
    "JobsSource",
    "ProductHuntSource",
    "RSSNewsSource",
    "RateLimitConfig",
    "SourceConfig",
    "YCDirectorySource",
    "get_adapter",
    "load_sources",
]
