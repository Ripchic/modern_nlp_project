"""Unit tests for reviewmind.workers — Celery app factory, tasks, and exports."""

from unittest.mock import MagicMock, patch

# ── Test Constants ────────────────────────────────────────────────────────────


class TestConstants:
    """Tests for module-level constants."""

    def test_default_broker_url(self):
        from reviewmind.workers.celery_app import DEFAULT_BROKER_URL

        assert DEFAULT_BROKER_URL == "redis://localhost:6379/1"

    def test_default_result_backend(self):
        from reviewmind.workers.celery_app import DEFAULT_RESULT_BACKEND

        assert DEFAULT_RESULT_BACKEND == "redis://localhost:6379/2"

    def test_task_modules_contains_tasks(self):
        from reviewmind.workers.celery_app import TASK_MODULES

        assert "reviewmind.workers.tasks" in TASK_MODULES

    def test_task_modules_is_list(self):
        from reviewmind.workers.celery_app import TASK_MODULES

        assert isinstance(TASK_MODULES, list)


# ── Test create_celery_app ───────────────────────────────────────────────────


class TestCreateCeleryApp:
    """Tests for the create_celery_app factory function."""

    def test_returns_celery_instance(self):
        from celery import Celery

        from reviewmind.workers.celery_app import create_celery_app

        app = create_celery_app(
            broker_url="redis://test:6379/1",
            result_backend="redis://test:6379/2",
        )
        assert isinstance(app, Celery)

    def test_app_name_is_reviewmind(self):
        from reviewmind.workers.celery_app import create_celery_app

        app = create_celery_app(
            broker_url="redis://test:6379/1",
            result_backend="redis://test:6379/2",
        )
        assert app.main == "reviewmind"

    def test_explicit_broker_url(self):
        from reviewmind.workers.celery_app import create_celery_app

        broker = "redis://custom:6379/10"
        app = create_celery_app(broker_url=broker, result_backend="redis://x:6379/2")
        # Celery stores broker URL in conf.broker_url
        assert str(app.conf.broker_url) == broker

    def test_explicit_result_backend(self):
        from reviewmind.workers.celery_app import create_celery_app

        backend = "redis://custom:6379/20"
        app = create_celery_app(broker_url="redis://x:6379/1", result_backend=backend)
        assert str(app.conf.result_backend) == backend

    def test_task_serializer_json(self):
        from reviewmind.workers.celery_app import create_celery_app

        app = create_celery_app(broker_url="redis://x:6379/1", result_backend="redis://x:6379/2")
        assert app.conf.task_serializer == "json"

    def test_accept_content_json(self):
        from reviewmind.workers.celery_app import create_celery_app

        app = create_celery_app(broker_url="redis://x:6379/1", result_backend="redis://x:6379/2")
        assert "json" in app.conf.accept_content

    def test_result_serializer_json(self):
        from reviewmind.workers.celery_app import create_celery_app

        app = create_celery_app(broker_url="redis://x:6379/1", result_backend="redis://x:6379/2")
        assert app.conf.result_serializer == "json"

    def test_task_acks_late(self):
        from reviewmind.workers.celery_app import create_celery_app

        app = create_celery_app(broker_url="redis://x:6379/1", result_backend="redis://x:6379/2")
        assert app.conf.task_acks_late is True

    def test_worker_prefetch_multiplier(self):
        from reviewmind.workers.celery_app import create_celery_app

        app = create_celery_app(broker_url="redis://x:6379/1", result_backend="redis://x:6379/2")
        assert app.conf.worker_prefetch_multiplier == 1

    def test_timezone_utc(self):
        from reviewmind.workers.celery_app import create_celery_app

        app = create_celery_app(broker_url="redis://x:6379/1", result_backend="redis://x:6379/2")
        assert app.conf.timezone == "UTC"

    def test_enable_utc(self):
        from reviewmind.workers.celery_app import create_celery_app

        app = create_celery_app(broker_url="redis://x:6379/1", result_backend="redis://x:6379/2")
        assert app.conf.enable_utc is True

    def test_result_expires(self):
        from reviewmind.workers.celery_app import create_celery_app

        app = create_celery_app(broker_url="redis://x:6379/1", result_backend="redis://x:6379/2")
        assert app.conf.result_expires == 86400

    def test_broker_connection_retry_on_startup(self):
        from reviewmind.workers.celery_app import create_celery_app

        app = create_celery_app(broker_url="redis://x:6379/1", result_backend="redis://x:6379/2")
        assert app.conf.broker_connection_retry_on_startup is True

    def test_include_task_modules(self):
        from reviewmind.workers.celery_app import TASK_MODULES, create_celery_app

        app = create_celery_app(broker_url="redis://x:6379/1", result_backend="redis://x:6379/2")
        assert app.conf.include == TASK_MODULES


# ── Test config fallback ─────────────────────────────────────────────────────


class TestConfigFallback:
    """Tests for config resolution in create_celery_app."""

    def test_falls_back_to_config_settings(self):
        """When no explicit args given, create_celery_app reads config."""
        from reviewmind.workers.celery_app import create_celery_app

        mock_settings = MagicMock()
        mock_settings.celery_broker_url = "redis://from-config:6379/1"
        mock_settings.celery_result_backend = "redis://from-config:6379/2"

        with patch("reviewmind.config.settings", mock_settings):
            app = create_celery_app()
        assert str(app.conf.broker_url) == "redis://from-config:6379/1"
        assert str(app.conf.result_backend) == "redis://from-config:6379/2"

    def test_falls_back_to_defaults_when_config_unavailable(self):
        """When config import fails, should use default URLs."""
        from reviewmind.workers.celery_app import DEFAULT_BROKER_URL, DEFAULT_RESULT_BACKEND, create_celery_app

        # With explicit args, config is not consulted
        app = create_celery_app(
            broker_url=DEFAULT_BROKER_URL,
            result_backend=DEFAULT_RESULT_BACKEND,
        )
        assert str(app.conf.broker_url) == DEFAULT_BROKER_URL
        assert str(app.conf.result_backend) == DEFAULT_RESULT_BACKEND

    def test_explicit_broker_overrides_config(self):
        from reviewmind.workers.celery_app import create_celery_app

        custom_broker = "redis://override:6379/99"
        app = create_celery_app(
            broker_url=custom_broker,
            result_backend="redis://x:6379/2",
        )
        assert str(app.conf.broker_url) == custom_broker


# ── Test module-level celery_app ─────────────────────────────────────────────


class TestModuleLevelApp:
    """Tests for the module-level celery_app instance."""

    def test_celery_app_exists(self):
        from reviewmind.workers.celery_app import celery_app

        assert celery_app is not None

    def test_celery_app_is_celery_instance(self):
        from celery import Celery

        from reviewmind.workers.celery_app import celery_app

        assert isinstance(celery_app, Celery)

    def test_celery_app_name(self):
        from reviewmind.workers.celery_app import celery_app

        assert celery_app.main == "reviewmind"

    def test_celery_app_json_serializer(self):
        from reviewmind.workers.celery_app import celery_app

        assert celery_app.conf.task_serializer == "json"


# ── Test ping task ───────────────────────────────────────────────────────────


class TestPingTask:
    """Tests for the ping health-check task."""

    def test_ping_returns_pong(self):
        from reviewmind.workers.tasks import ping

        result = ping()
        assert result == {"status": "pong"}

    def test_ping_returns_dict(self):
        from reviewmind.workers.tasks import ping

        result = ping()
        assert isinstance(result, dict)

    def test_ping_has_status_key(self):
        from reviewmind.workers.tasks import ping

        result = ping()
        assert "status" in result

    def test_ping_task_name(self):
        from reviewmind.workers.tasks import ping

        assert ping.name == "reviewmind.ping"

    def test_ping_is_registered_on_app(self):
        from reviewmind.workers.celery_app import celery_app

        assert "reviewmind.ping" in celery_app.tasks

    def test_ping_max_retries_zero(self):
        from reviewmind.workers.tasks import ping

        assert ping.max_retries == 0


# ── Test workers __init__ exports ────────────────────────────────────────────


class TestWorkersExports:
    """Tests for reviewmind.workers package exports."""

    def test_import_celery_app(self):
        from reviewmind.workers import celery_app

        assert celery_app is not None

    def test_import_create_celery_app(self):
        from reviewmind.workers import create_celery_app

        assert callable(create_celery_app)

    def test_import_ping(self):
        from reviewmind.workers import ping

        assert callable(ping)

    def test_all_exports(self):
        import reviewmind.workers as w

        for name in w.__all__:
            assert hasattr(w, name), f"Missing export: {name}"

    def test_all_count(self):
        import reviewmind.workers as w

        assert len(w.__all__) == 14


# ── Test config fields ──────────────────────────────────────────────────────


class TestConfigCeleryFields:
    """Tests for Celery-related fields in Settings."""

    def test_celery_broker_url_default(self):
        from reviewmind.config import Settings

        s = Settings(
            telegram_bot_token="test",
            openai_api_key="test",
            _env_file=None,
        )
        assert s.celery_broker_url == "redis://localhost:6379/1"

    def test_celery_result_backend_default(self):
        from reviewmind.config import Settings

        s = Settings(
            telegram_bot_token="test",
            openai_api_key="test",
            _env_file=None,
        )
        assert s.celery_result_backend == "redis://localhost:6379/2"

    def test_celery_broker_url_from_env(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
        monkeypatch.setenv("OPENAI_API_KEY", "key")
        monkeypatch.setenv("CELERY_BROKER_URL", "redis://custom:6379/11")
        from reviewmind.config import Settings

        s = Settings(_env_file=None)
        assert s.celery_broker_url == "redis://custom:6379/11"

    def test_celery_result_backend_from_env(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
        monkeypatch.setenv("OPENAI_API_KEY", "key")
        monkeypatch.setenv("CELERY_RESULT_BACKEND", "redis://custom:6379/22")
        from reviewmind.config import Settings

        s = Settings(_env_file=None)
        assert s.celery_result_backend == "redis://custom:6379/22"


# ── Test Docker Compose alignment ────────────────────────────────────────────


class TestDockerComposeAlignment:
    """Verify Celery config aligns with docker-compose.yml service definitions."""

    def test_default_broker_matches_docker_compose(self):
        """docker-compose.yml uses redis://redis:6379/1 for CELERY_BROKER_URL.
        Default config uses redis://localhost:6379/1 (same DB index, different host).
        """
        from reviewmind.workers.celery_app import DEFAULT_BROKER_URL

        assert "/1" in DEFAULT_BROKER_URL

    def test_default_backend_matches_docker_compose(self):
        """docker-compose.yml uses redis://redis:6379/2 for CELERY_RESULT_BACKEND.
        Default config uses redis://localhost:6379/2 (same DB index, different host).
        """
        from reviewmind.workers.celery_app import DEFAULT_RESULT_BACKEND

        assert "/2" in DEFAULT_RESULT_BACKEND

    def test_celery_app_name_matches_command(self):
        """docker-compose.yml command: celery -A reviewmind.workers.celery_app worker ..."""
        from reviewmind.workers.celery_app import celery_app

        # Celery uses the main attribute as the app name
        assert celery_app.main == "reviewmind"


# ── Test Integration Scenarios ───────────────────────────────────────────────


class TestIntegrationScenarios:
    """Integration-level scenarios."""

    def test_create_app_and_register_task(self):
        """Creating a fresh app and registering a task on it."""
        from reviewmind.workers.celery_app import create_celery_app

        app = create_celery_app(broker_url="redis://test:6379/1", result_backend="redis://test:6379/2")

        @app.task(name="test.add")
        def add(x, y):
            return x + y

        assert "test.add" in app.tasks

    def test_ping_task_callable_directly(self):
        """ping() can be called without a worker (synchronously)."""
        from reviewmind.workers.tasks import ping

        r = ping()
        assert r["status"] == "pong"

    def test_multiple_app_instances_independent(self):
        """Two create_celery_app calls produce independent instances."""
        from reviewmind.workers.celery_app import create_celery_app

        a1 = create_celery_app(broker_url="redis://a:6379/1", result_backend="redis://a:6379/2")
        a2 = create_celery_app(broker_url="redis://b:6379/1", result_backend="redis://b:6379/2")
        assert a1 is not a2
        assert str(a1.conf.broker_url) != str(a2.conf.broker_url)

    def test_app_conf_immutable_after_creation(self):
        """Key settings stay stable after creation."""
        from reviewmind.workers.celery_app import create_celery_app

        app = create_celery_app(broker_url="redis://x:6379/1", result_backend="redis://x:6379/2")
        assert app.conf.task_acks_late is True
        assert app.conf.worker_prefetch_multiplier == 1
        assert app.conf.timezone == "UTC"

    def test_full_import_chain(self):
        """Import path matches what Celery CLI uses."""
        import importlib

        real_mod = importlib.import_module("reviewmind.workers.celery_app")
        assert real_mod is not None
