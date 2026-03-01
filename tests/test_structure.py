"""Smoke tests for project structure and imports."""

import importlib
import os


def test_reviewmind_import():
    """Test that reviewmind package can be imported."""
    import reviewmind

    assert reviewmind.__version__ == "0.1.0"


def test_subpackages_importable():
    """Test that all subpackages can be imported."""
    subpackages = [
        "reviewmind.api",
        "reviewmind.api.endpoints",
        "reviewmind.bot",
        "reviewmind.bot.handlers",
        "reviewmind.core",
        "reviewmind.scrapers",
        "reviewmind.ingestion",
        "reviewmind.db",
        "reviewmind.db.repositories",
        "reviewmind.vectorstore",
        "reviewmind.cache",
        "reviewmind.workers",
        "reviewmind.services",
    ]
    for pkg in subpackages:
        mod = importlib.import_module(pkg)
        assert mod is not None, f"Failed to import {pkg}"


def test_project_structure_exists():
    """Test that key directories exist."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    expected_dirs = [
        "src/reviewmind",
        "src/reviewmind/api",
        "src/reviewmind/api/endpoints",
        "src/reviewmind/bot",
        "src/reviewmind/bot/handlers",
        "src/reviewmind/core",
        "src/reviewmind/scrapers",
        "src/reviewmind/ingestion",
        "src/reviewmind/db",
        "src/reviewmind/db/repositories",
        "src/reviewmind/vectorstore",
        "src/reviewmind/cache",
        "src/reviewmind/workers",
        "src/reviewmind/services",
        "scripts",
        "tests",
        "alembic",
        "alembic/versions",
    ]
    for d in expected_dirs:
        assert os.path.isdir(os.path.join(base, d)), f"Directory missing: {d}"
