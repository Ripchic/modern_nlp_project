"""reviewmind/config.py — pydantic-settings: все настройки из .env."""

from __future__ import annotations

import json as _json
from functools import lru_cache
from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.main import PydanticBaseSettingsSource

try:
    from pydantic_settings.sources.providers.dotenv import DotEnvSettingsSource
    from pydantic_settings.sources.providers.env import EnvSettingsSource
except ImportError:  # pydantic-settings < 2.8
    from pydantic_settings import DotEnvSettingsSource, EnvSettingsSource  # type: ignore[attr-defined,no-redef]


class _GracefulEnvSource(EnvSettingsSource):
    """Env source that gracefully handles comma-separated list fields."""

    def prepare_field_value(self, field_name: str, field: Any, value: Any, value_is_complex: bool) -> Any:
        if value is not None and self.field_is_complex(field) and isinstance(value, str):
            try:
                return _json.loads(value)
            except (_json.JSONDecodeError, ValueError):
                return value
        return super().prepare_field_value(field_name, field, value, value_is_complex)


class _GracefulDotEnvSource(DotEnvSettingsSource):
    """DotEnv source that gracefully handles comma-separated list fields."""

    def prepare_field_value(self, field_name: str, field: Any, value: Any, value_is_complex: bool) -> Any:
        if value is not None and self.field_is_complex(field) and isinstance(value, str):
            try:
                return _json.loads(value)
            except (_json.JSONDecodeError, ValueError):
                return value
        return super().prepare_field_value(field_name, field, value, value_is_complex)


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Telegram ──────────────────────────────────────────────
    telegram_bot_token: str
    admin_user_ids: list[int] = []

    # ── OpenAI / GitHub Models ────────────────────────────────
    openai_api_key: str
    openai_base_url: str = "https://models.inference.ai.azure.com"
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # ── YouTube ───────────────────────────────────────────────
    youtube_api_key: str = ""
    youtube_cookies_path: str = ""

    # ── Reddit ────────────────────────────────────────────────
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "ReviewMind/1.0"

    # ── Tavily ────────────────────────────────────────────────
    tavily_api_key: str = ""

    # ── PostgreSQL ────────────────────────────────────────────
    postgres_user: str = "reviewmind"
    postgres_password: str = ""
    postgres_db: str = "reviewmind"
    database_url: str = "postgresql+asyncpg://reviewmind:password@localhost:5432/reviewmind"

    # ── Redis ─────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── Celery ────────────────────────────────────────────────
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # ── Qdrant ────────────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"

    # ── LLM defaults ─────────────────────────────────────────
    llm_temperature: float = 0.3
    llm_max_tokens: int = 1000

    # ── Session ───────────────────────────────────────────────
    session_ttl_seconds: int = 1800  # 30 minutes

    # ── Rate limiting ─────────────────────────────────────────
    rate_limit_per_minute: int = 10

    @field_validator("admin_user_ids", mode="before")
    @classmethod
    def parse_admin_user_ids(cls, v: object) -> list[int]:
        """Parse comma-separated string of user IDs into a list of ints."""
        if isinstance(v, int):
            return [v]
        if isinstance(v, str):
            if not v.strip():
                return []
            return [int(uid.strip()) for uid in v.split(",") if uid.strip()]
        if isinstance(v, list):
            return [int(uid) for uid in v]
        return []

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Use graceful env source that handles comma-separated lists."""
        # Respect _env_file override (e.g. _env_file=None in tests)
        env_file = getattr(dotenv_settings, "env_file", cls.model_config.get("env_file", ".env"))
        env_file_encoding = getattr(
            dotenv_settings, "env_file_encoding", cls.model_config.get("env_file_encoding", "utf-8"),
        )
        return (
            init_settings,
            _GracefulEnvSource(settings_cls),
            _GracefulDotEnvSource(
                settings_cls,
                env_file=env_file,
                env_file_encoding=env_file_encoding,
            ),
            file_secret_settings,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings singleton."""
    return Settings()  # type: ignore[call-arg]


class _SettingsProxy:
    """Lazy proxy that defers Settings instantiation until first attribute access."""

    def __getattr__(self, name: str) -> Any:
        return getattr(get_settings(), name)


settings: Settings = _SettingsProxy()  # type: ignore[assignment]
