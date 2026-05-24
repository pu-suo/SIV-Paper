"""Pytest configuration for the SIV test suite."""
import os

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_llm: test requires a live LLM API call "
        "(skipped unless OPENAI_API_KEY is set).",
    )


def pytest_collection_modifyitems(config, items):
    """Skip @pytest.mark.requires_llm tests when no API key is configured."""
    if os.environ.get("OPENAI_API_KEY"):
        return
    skip_llm = pytest.mark.skip(reason="OPENAI_API_KEY not set")
    for item in items:
        if "requires_llm" in item.keywords:
            item.add_marker(skip_llm)
