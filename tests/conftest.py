"""Pytest configuration and shared fixtures."""

import pytest


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return "This is a sample review text for testing purposes."
