"""Unit tests for entity resolution against seed data."""

import pytest
from src.resolver.resolver import EntityResolver


@pytest.fixture
def resolver():
    return EntityResolver()


def test_exact_resolution(resolver):
    canonical, method, score = resolver.resolve("OpenAI")
    assert canonical == "OpenAI"
    assert method in ("exact", "alias")
    assert score == 1.0


def test_alias_resolution(resolver):
    canonical, method, score = resolver.resolve("OpenAI, Inc.")
    assert canonical == "OpenAI"
    assert method in ("alias", "exact")
    assert score >= 0.95


def test_compact_resolution(resolver):
    canonical, method, score = resolver.resolve("Open AI")
    assert canonical == "OpenAI"
    assert method in ("compact", "alias", "exact", "fuzzy")
    assert score >= 0.9


def test_fuzzy_resolution(resolver):
    canonical, method, score = resolver.resolve("Anthropic PBC")
    assert canonical == "Anthropic"
    assert score >= 0.90


def test_unresolved_fallback(resolver):
    raw_name = "TotallyUnknownStartup999X"
    canonical, method, score = resolver.resolve(raw_name)
    assert canonical == raw_name
    assert method == "unresolved"
    assert score == 0.0
