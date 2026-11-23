"""Unit tests for string grounding check."""

import pytest
from src.llm.grounding import ground_check


def test_grounding_exact_and_fuzzy():
    source_text = """
    Weights & Biases is the machine learning platform for developers to build better models faster.
    Founded by Lukas Biewald and Chris Van Pelt in San Francisco.
    Total funding exceeds $200 million with thousands of enterprise customers.
    """
    extracted = {
        "name": "Weights & Biases",
        "founders": ["Lukas Biewald", "Chris Van Pelt"],
        "headquarters": "San Francisco",
        "hallucinated_investor": "Fictional Ventures LLC",
    }
    required = ["name"]

    grounded, warnings = ground_check(extracted, source_text, required)
    assert grounded is not None
    assert grounded["name"] == "Weights & Biases"
    assert "Lukas Biewald" in grounded["founders"]
    assert grounded["headquarters"] == "San Francisco"
    # Hallucinated field must be nullified
    assert grounded.get("hallucinated_investor") is None


def test_grounding_required_field_dropped():
    source_text = "This article talks only about autonomous vehicles in Detroit."
    extracted = {
        "name": "QuantumSupremacy AI",
        "headquarters": "Tokyo",
    }
    required = ["name"]

    grounded, warnings = ground_check(extracted, source_text, required)
    # Because 'name' is hallucinated and required, the entire record must be dropped (None returned)
    assert grounded is None
