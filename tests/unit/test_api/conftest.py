"""Fixtures for test_api unit tests."""

import pytest

from reviewmind.api.rate_limit import limiter


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset slowapi in-memory counters before each test to avoid cross-test rate limiting."""
    limiter._storage.reset()
    yield
