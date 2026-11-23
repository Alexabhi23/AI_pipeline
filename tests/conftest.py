"""Pytest fixtures for GraphOne pipeline test suite."""

import pytest
from datetime import datetime, timezone


@pytest.fixture
def sample_now():
    return datetime.now(timezone.utc)


@pytest.fixture
def sample_startup_payload():
    return {
        "name": "Anthropic",
        "description": "AI safety and research company building reliable, interpretable systems.",
        "founders": ["Dario Amodei", "Daniela Amodei"],
        "headquarters": "San Francisco, CA",
        "employeeCount": 500,
        "pricingModel": "PAID",
        "website": "https://anthropic.com",
        "categories": ["AI Safety", "LLM", "Foundation Models"],
    }


@pytest.fixture
def sample_product_payload():
    return {
        "name": "Claude 3.5 Sonnet",
        "description": "State of the art language model by Anthropic.",
        "company": "Anthropic",
        "pricingModel": "FREEMIUM",
        "website": "https://claude.ai",
        "categories": ["LLM", "Chatbot"],
        "upvotes": 1250,
    }


@pytest.fixture
def sample_paper_payload():
    return {
        "title": "Attention Is All You Need",
        "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit"],
        "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.",
        "arxivId": "1706.03762",
        "githubUrl": "https://github.com/tensorflow/tensor2tensor",
        "githubStars": 14500,
        "categories": ["cs.CL", "cs.LG"],
    }
