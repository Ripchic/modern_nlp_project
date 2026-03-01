"""Tests for reviewmind.config module."""

import pytest
from pydantic import ValidationError

from reviewmind.config import Settings


class TestAdminUserIdsParsing:
    """Test ADMIN_USER_IDS parsing from comma-separated string."""

    def test_parse_comma_separated(self):
        s = Settings(
            telegram_bot_token="test-token",
            openai_api_key="test-key",
            admin_user_ids="123,456",
        )
        assert s.admin_user_ids == [123, 456]

    def test_parse_single_id(self):
        s = Settings(
            telegram_bot_token="test-token",
            openai_api_key="test-key",
            admin_user_ids="789",
        )
        assert s.admin_user_ids == [789]

    def test_parse_empty_string(self):
        s = Settings(
            telegram_bot_token="test-token",
            openai_api_key="test-key",
            admin_user_ids="",
        )
        assert s.admin_user_ids == []

    def test_parse_with_spaces(self):
        s = Settings(
            telegram_bot_token="test-token",
            openai_api_key="test-key",
            admin_user_ids=" 123 , 456 , 789 ",
        )
        assert s.admin_user_ids == [123, 456, 789]

    def test_default_empty_list(self):
        s = Settings(
            telegram_bot_token="test-token",
            openai_api_key="test-key",
        )
        assert s.admin_user_ids == []

    def test_parse_list_input(self):
        s = Settings(
            telegram_bot_token="test-token",
            openai_api_key="test-key",
            admin_user_ids=[111, 222],
        )
        assert s.admin_user_ids == [111, 222]


class TestSettingsDefaults:
    """Test default values for settings."""

    def _make_settings(self, **overrides):
        defaults = {
            "telegram_bot_token": "test-token",
            "openai_api_key": "test-key",
        }
        defaults.update(overrides)
        return Settings(**defaults)

    def test_openai_base_url_default(self):
        s = self._make_settings()
        assert s.openai_base_url == "https://models.inference.ai.azure.com"

    def test_openai_base_url_override(self):
        s = self._make_settings(openai_base_url="https://api.openai.com/v1")
        assert s.openai_base_url == "https://api.openai.com/v1"

    def test_openai_model_default(self):
        s = self._make_settings()
        assert s.openai_model == "gpt-4o-mini"

    def test_postgres_defaults(self):
        s = self._make_settings()
        assert s.postgres_user == "reviewmind"
        assert s.postgres_db == "reviewmind"

    def test_redis_url_default(self):
        s = self._make_settings()
        assert s.redis_url == "redis://localhost:6379/0"

    def test_qdrant_url_default(self):
        s = self._make_settings()
        assert s.qdrant_url == "http://localhost:6333"

    def test_llm_defaults(self):
        s = self._make_settings()
        assert s.llm_temperature == 0.3
        assert s.llm_max_tokens == 1000

    def test_session_ttl_default(self):
        s = self._make_settings()
        assert s.session_ttl_seconds == 1800

    def test_reddit_user_agent_default(self):
        s = self._make_settings()
        assert s.reddit_user_agent == "ReviewMind/1.0"

    def test_embedding_model_default(self):
        s = self._make_settings()
        assert s.openai_embedding_model == "text-embedding-3-small"


class TestSettingsValidation:
    """Test that missing required fields raise ValidationError."""

    def test_missing_telegram_bot_token(self, monkeypatch):
        # Clear env vars that could satisfy the requirement
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValidationError) as exc_info:
            Settings(_env_file=None)  # type: ignore[call-arg]
        errors = exc_info.value.errors()
        fields = [e["loc"][0] for e in errors]
        assert "telegram_bot_token" in fields

    def test_missing_openai_api_key(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValidationError) as exc_info:
            Settings(_env_file=None)  # type: ignore[call-arg]
        errors = exc_info.value.errors()
        fields = [e["loc"][0] for e in errors]
        assert "openai_api_key" in fields


class TestSettingsFromEnv:
    """Test that settings are correctly loaded from environment variables."""

    def test_from_env_vars(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "my-bot-token")
        monkeypatch.setenv("OPENAI_API_KEY", "my-api-key")
        monkeypatch.setenv("ADMIN_USER_IDS", "100,200,300")
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@db:5432/test")
        monkeypatch.setenv("REDIS_URL", "redis://myredis:6379/1")
        monkeypatch.setenv("QDRANT_URL", "http://myqdrant:6333")

        s = Settings(_env_file=None)  # type: ignore[call-arg]

        assert s.telegram_bot_token == "my-bot-token"
        assert s.openai_api_key == "my-api-key"
        assert s.admin_user_ids == [100, 200, 300]
        assert s.database_url == "postgresql+asyncpg://u:p@db:5432/test"
        assert s.redis_url == "redis://myredis:6379/1"
        assert s.qdrant_url == "http://myqdrant:6333"

    def test_model_dump_contains_all_keys(self):
        s = Settings(
            telegram_bot_token="t",
            openai_api_key="k",
        )
        dump = s.model_dump()
        expected_keys = {
            "telegram_bot_token",
            "admin_user_ids",
            "openai_api_key",
            "openai_base_url",
            "openai_model",
            "openai_embedding_model",
            "youtube_api_key",
            "reddit_client_id",
            "reddit_client_secret",
            "reddit_user_agent",
            "tavily_api_key",
            "postgres_user",
            "postgres_password",
            "postgres_db",
            "database_url",
            "redis_url",
            "qdrant_url",
            "llm_temperature",
            "llm_max_tokens",
            "session_ttl_seconds",
        }
        assert expected_keys.issubset(dump.keys())
