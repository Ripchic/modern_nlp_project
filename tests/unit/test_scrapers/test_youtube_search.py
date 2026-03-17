"""Unit tests for YouTubeScraper.search_videos — YouTube Data API v3 search."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx

from reviewmind.scrapers.youtube import (
    DEFAULT_SEARCH_MAX_RESULTS,
    MAX_AGE_DAYS,
    YOUTUBE_API_BASE_URL,
    VideoInfo,
    YouTubeScraper,
    _parse_search_items,
    _published_after_iso,
)

# ═══════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════


def _make_api_response(items: list[dict] | None = None) -> dict:
    """Build a fake YouTube Data API v3 search response."""
    if items is None:
        items = [
            _make_api_item("vid1", "Review of Sony WH-1000XM5", "TechChannel"),
            _make_api_item("vid2", "Sony XM5 vs Bose QC Ultra", "AudioExpert"),
            _make_api_item("vid3", "Best headphones 2026", "GadgetGuru"),
        ]
    return {"kind": "youtube#searchListResponse", "items": items}


def _make_api_item(
    video_id: str = "dQw4w9WgXcQ",
    title: str = "Test Video",
    channel: str = "TestChannel",
    published: str = "2026-01-15T10:00:00Z",
    description: str = "A test video description.",
) -> dict:
    return {
        "id": {"kind": "youtube#video", "videoId": video_id},
        "snippet": {
            "title": title,
            "channelTitle": channel,
            "publishedAt": published,
            "description": description,
        },
    }


def _make_httpx_response(data: dict, status_code: int = 200) -> httpx.Response:
    """Create a fake httpx.Response with JSON body."""
    response = httpx.Response(
        status_code=status_code,
        json=data,
        request=httpx.Request("GET", "https://example.com"),
    )
    return response


# ═══════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════


class TestSearchConstants:
    def test_default_max_results(self):
        assert DEFAULT_SEARCH_MAX_RESULTS == 10

    def test_youtube_api_base_url(self):
        assert "googleapis.com" in YOUTUBE_API_BASE_URL

    def test_max_age_days(self):
        assert MAX_AGE_DAYS == 365

    def test_constants_importable_from_init(self):
        from reviewmind.scrapers import DEFAULT_SEARCH_MAX_RESULTS as DSMR
        from reviewmind.scrapers import MAX_AGE_DAYS as MAD
        from reviewmind.scrapers import YOUTUBE_API_BASE_URL as YABU

        assert DSMR == 10
        assert MAD == 365
        assert "googleapis" in YABU


# ═══════════════════════════════════════════════════════════════
#  VideoInfo dataclass
# ═══════════════════════════════════════════════════════════════


class TestVideoInfo:
    def test_basic_creation(self):
        v = VideoInfo(
            video_id="abc123",
            title="Test",
            channel_title="Channel",
            published_at="2026-01-01T00:00:00Z",
            description="Desc",
            url="https://www.youtube.com/watch?v=abc123",
        )
        assert v.video_id == "abc123"
        assert v.title == "Test"
        assert v.url == "https://www.youtube.com/watch?v=abc123"

    def test_url_format(self):
        v = VideoInfo(
            video_id="XYZ789abcde",
            title="T",
            channel_title="C",
            published_at="",
            description="",
            url="https://www.youtube.com/watch?v=XYZ789abcde",
        )
        assert v.video_id in v.url

    def test_importable_from_init(self):
        from reviewmind.scrapers import VideoInfo as VI

        assert VI is VideoInfo


# ═══════════════════════════════════════════════════════════════
#  _published_after_iso
# ═══════════════════════════════════════════════════════════════


class TestPublishedAfterIso:
    def test_returns_iso_string(self):
        result = _published_after_iso()
        assert result.endswith("Z")
        assert "T" in result

    def test_custom_days(self):
        result = _published_after_iso(max_age_days=30)
        assert result.endswith("Z")

    def test_format_is_parseable(self):
        from datetime import datetime

        result = _published_after_iso()
        dt = datetime.strptime(result, "%Y-%m-%dT%H:%M:%SZ")
        assert dt.year >= 2025


# ═══════════════════════════════════════════════════════════════
#  _parse_search_items
# ═══════════════════════════════════════════════════════════════


class TestParseSearchItems:
    def test_parses_items(self):
        items = [
            _make_api_item("v1", "Title 1", "Ch1"),
            _make_api_item("v2", "Title 2", "Ch2"),
        ]
        result = _parse_search_items(items)
        assert len(result) == 2
        assert result[0].video_id == "v1"
        assert result[1].video_id == "v2"

    def test_empty_items(self):
        assert _parse_search_items([]) == []

    def test_skips_items_without_video_id(self):
        items = [
            {"id": {}, "snippet": {"title": "No ID"}},
            _make_api_item("v1", "Has ID"),
        ]
        result = _parse_search_items(items)
        assert len(result) == 1
        assert result[0].video_id == "v1"

    def test_url_constructed_correctly(self):
        items = [_make_api_item("dQw4w9WgXcQ")]
        result = _parse_search_items(items)
        assert result[0].url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_all_fields_populated(self):
        items = [_make_api_item("v1", "Title", "Channel", "2026-01-01T00:00:00Z", "Desc")]
        result = _parse_search_items(items)
        v = result[0]
        assert v.title == "Title"
        assert v.channel_title == "Channel"
        assert v.published_at == "2026-01-01T00:00:00Z"
        assert v.description == "Desc"

    def test_missing_snippet_fields_default_to_empty(self):
        items = [{"id": {"videoId": "v1"}, "snippet": {}}]
        result = _parse_search_items(items)
        assert result[0].title == ""
        assert result[0].channel_title == ""


# ═══════════════════════════════════════════════════════════════
#  search_videos — success paths
# ═══════════════════════════════════════════════════════════════


class TestSearchVideosSuccess:
    def test_returns_video_list(self):
        scraper = YouTubeScraper()
        api_resp = _make_api_response()
        fake_resp = _make_httpx_response(api_resp)

        with patch("reviewmind.scrapers.youtube.httpx.Client") as mock_client:
            mock_instance = MagicMock()
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=False)
            mock_instance.get.return_value = fake_resp
            mock_client.return_value = mock_instance

            results = scraper.search_videos("Sony XM5 review", api_key="test-key")

        assert len(results) == 3
        assert all(isinstance(r, VideoInfo) for r in results)

    def test_passes_correct_params(self):
        scraper = YouTubeScraper()
        api_resp = _make_api_response([])
        fake_resp = _make_httpx_response(api_resp)

        with patch("reviewmind.scrapers.youtube.httpx.Client") as mock_client:
            mock_instance = MagicMock()
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=False)
            mock_instance.get.return_value = fake_resp
            mock_client.return_value = mock_instance

            scraper.search_videos("test query", api_key="my-key", max_results=5)

            call_args = mock_instance.get.call_args
            params = call_args[1]["params"]
            assert params["q"] == "test query"
            assert params["maxResults"] == 5
            assert params["type"] == "video"
            assert params["key"] == "my-key"
            assert params["order"] == "relevance"
            assert "publishedAfter" in params

    def test_url_contains_api_base(self):
        scraper = YouTubeScraper()
        api_resp = _make_api_response([])
        fake_resp = _make_httpx_response(api_resp)

        with patch("reviewmind.scrapers.youtube.httpx.Client") as mock_client:
            mock_instance = MagicMock()
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=False)
            mock_instance.get.return_value = fake_resp
            mock_client.return_value = mock_instance

            scraper.search_videos("test", api_key="k")

            call_args = mock_instance.get.call_args
            url = call_args[0][0]
            assert YOUTUBE_API_BASE_URL in url

    def test_max_results_clamped_to_50(self):
        scraper = YouTubeScraper()
        api_resp = _make_api_response([])
        fake_resp = _make_httpx_response(api_resp)

        with patch("reviewmind.scrapers.youtube.httpx.Client") as mock_client:
            mock_instance = MagicMock()
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=False)
            mock_instance.get.return_value = fake_resp
            mock_client.return_value = mock_instance

            scraper.search_videos("test", api_key="k", max_results=100)

            params = mock_instance.get.call_args[1]["params"]
            assert params["maxResults"] == 50

    def test_max_results_clamped_to_1(self):
        scraper = YouTubeScraper()
        api_resp = _make_api_response([])
        fake_resp = _make_httpx_response(api_resp)

        with patch("reviewmind.scrapers.youtube.httpx.Client") as mock_client:
            mock_instance = MagicMock()
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=False)
            mock_instance.get.return_value = fake_resp
            mock_client.return_value = mock_instance

            scraper.search_videos("test", api_key="k", max_results=-5)

            params = mock_instance.get.call_args[1]["params"]
            assert params["maxResults"] == 1


# ═══════════════════════════════════════════════════════════════
#  search_videos — empty / invalid input
# ═══════════════════════════════════════════════════════════════


class TestSearchVideosEmptyInput:
    def test_empty_query(self):
        scraper = YouTubeScraper()
        assert scraper.search_videos("", api_key="k") == []

    def test_whitespace_query(self):
        scraper = YouTubeScraper()
        assert scraper.search_videos("   ", api_key="k") == []

    def test_none_query(self):
        scraper = YouTubeScraper()
        assert scraper.search_videos(None, api_key="k") == []  # type: ignore[arg-type]

    def test_non_string_query(self):
        scraper = YouTubeScraper()
        assert scraper.search_videos(123, api_key="k") == []  # type: ignore[arg-type]


# ═══════════════════════════════════════════════════════════════
#  search_videos — no API key
# ═══════════════════════════════════════════════════════════════


class TestSearchVideosNoApiKey:
    def test_no_key_returns_empty(self):
        scraper = YouTubeScraper()
        with patch("reviewmind.scrapers.youtube.settings", create=True) as mock_settings:
            mock_settings.youtube_api_key = ""
            results = scraper.search_videos("test query")
        assert results == []

    def test_none_key_returns_empty(self):
        scraper = YouTubeScraper()
        with patch("reviewmind.scrapers.youtube.settings", create=True) as mock_settings:
            mock_settings.youtube_api_key = ""
            results = scraper.search_videos("test query", api_key=None)
        assert results == []


# ═══════════════════════════════════════════════════════════════
#  search_videos — error handling
# ═══════════════════════════════════════════════════════════════


class TestSearchVideosErrors:
    def test_http_error_returns_empty(self):
        scraper = YouTubeScraper()

        with patch("reviewmind.scrapers.youtube.httpx.Client") as mock_client:
            mock_instance = MagicMock()
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=False)
            error_resp = httpx.Response(
                status_code=403,
                request=httpx.Request("GET", "https://example.com"),
            )
            mock_instance.get.return_value = error_resp
            mock_client.return_value = mock_instance

            results = scraper.search_videos("test", api_key="bad-key")

        assert results == []

    def test_network_error_returns_empty(self):
        scraper = YouTubeScraper()

        with patch("reviewmind.scrapers.youtube.httpx.Client") as mock_client:
            mock_instance = MagicMock()
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=False)
            mock_instance.get.side_effect = httpx.ConnectError("Connection refused")
            mock_client.return_value = mock_instance

            results = scraper.search_videos("test", api_key="k")

        assert results == []

    def test_timeout_returns_empty(self):
        scraper = YouTubeScraper()

        with patch("reviewmind.scrapers.youtube.httpx.Client") as mock_client:
            mock_instance = MagicMock()
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=False)
            mock_instance.get.side_effect = httpx.ReadTimeout("Timeout")
            mock_client.return_value = mock_instance

            results = scraper.search_videos("test", api_key="k")

        assert results == []

    def test_unexpected_error_returns_empty(self):
        scraper = YouTubeScraper()

        with patch("reviewmind.scrapers.youtube.httpx.Client") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=False)
            mock_instance.get.side_effect = RuntimeError("Something broke")
            mock_cls.return_value = mock_instance

            results = scraper.search_videos("test", api_key="k")

        assert results == []

    def test_quota_exceeded_403_returns_empty(self):
        scraper = YouTubeScraper()

        with patch("reviewmind.scrapers.youtube.httpx.Client") as mock_client:
            mock_instance = MagicMock()
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=False)
            error_resp = httpx.Response(
                status_code=403,
                json={"error": {"message": "quotaExceeded"}},
                request=httpx.Request("GET", "https://example.com"),
            )
            mock_instance.get.return_value = error_resp
            mock_client.return_value = mock_instance

            results = scraper.search_videos("test", api_key="k")

        assert results == []


# ═══════════════════════════════════════════════════════════════
#  search_videos — config fallback for API key
# ═══════════════════════════════════════════════════════════════


class TestSearchVideosConfigFallback:
    def test_falls_back_to_config_key(self):
        scraper = YouTubeScraper()
        api_resp = _make_api_response([_make_api_item("v1")])
        fake_resp = _make_httpx_response(api_resp)

        with (
            patch("reviewmind.scrapers.youtube.httpx.Client") as mock_client,
            patch("reviewmind.config.settings") as mock_settings,
        ):
            mock_settings.youtube_api_key = "config-key"
            mock_instance = MagicMock()
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=False)
            mock_instance.get.return_value = fake_resp
            mock_client.return_value = mock_instance

            results = scraper.search_videos("test")

            params = mock_instance.get.call_args[1]["params"]
            assert params["key"] == "config-key"
            assert len(results) == 1


# ═══════════════════════════════════════════════════════════════
#  search_videos — date filter
# ═══════════════════════════════════════════════════════════════


class TestSearchVideosDateFilter:
    def test_published_after_param_present(self):
        scraper = YouTubeScraper()
        api_resp = _make_api_response([])
        fake_resp = _make_httpx_response(api_resp)

        with patch("reviewmind.scrapers.youtube.httpx.Client") as mock_client:
            mock_instance = MagicMock()
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=False)
            mock_instance.get.return_value = fake_resp
            mock_client.return_value = mock_instance

            scraper.search_videos("test", api_key="k")

            params = mock_instance.get.call_args[1]["params"]
            assert "publishedAfter" in params
            assert params["publishedAfter"].endswith("Z")

    def test_niche_product_returns_fewer_results(self):
        """A niche product may return < max_results — that's not an error."""
        scraper = YouTubeScraper()
        api_resp = _make_api_response([_make_api_item("v1")])
        fake_resp = _make_httpx_response(api_resp)

        with patch("reviewmind.scrapers.youtube.httpx.Client") as mock_client:
            mock_instance = MagicMock()
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=False)
            mock_instance.get.return_value = fake_resp
            mock_client.return_value = mock_instance

            results = scraper.search_videos("obscure niche product", api_key="k", max_results=10)

        assert len(results) == 1
        assert results[0].video_id == "v1"


# ═══════════════════════════════════════════════════════════════
#  Exports
# ═══════════════════════════════════════════════════════════════


class TestYouTubeSearchExports:
    def test_video_info_from_init(self):
        from reviewmind.scrapers import VideoInfo as VI

        assert VI is VideoInfo

    def test_constants_from_init(self):
        from reviewmind.scrapers import DEFAULT_SEARCH_MAX_RESULTS, MAX_AGE_DAYS, YOUTUBE_API_BASE_URL

        assert DEFAULT_SEARCH_MAX_RESULTS == 10
        assert MAX_AGE_DAYS == 365
        assert isinstance(YOUTUBE_API_BASE_URL, str)


# ═══════════════════════════════════════════════════════════════
#  Integration scenarios (all mocked, no real API calls)
# ═══════════════════════════════════════════════════════════════


class TestYouTubeSearchIntegration:
    def test_search_then_fetch_transcript_flow(self):
        """Simulate auto-mode: search → get URLs → fetch transcript."""
        scraper = YouTubeScraper(min_word_count=5)
        api_resp = _make_api_response([_make_api_item("dQw4w9WgXcQ", "Review")])
        fake_resp = _make_httpx_response(api_resp)

        with patch("reviewmind.scrapers.youtube.httpx.Client") as mock_client:
            mock_instance = MagicMock()
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=False)
            mock_instance.get.return_value = fake_resp
            mock_client.return_value = mock_instance

            videos = scraper.search_videos("Sony XM5", api_key="k")

        assert len(videos) == 1
        url = videos[0].url
        assert scraper.is_youtube_url(url)
        assert scraper.extract_video_id(url) == "dQw4w9WgXcQ"

    def test_search_result_urls_are_parseable(self):
        items = [_make_api_item(f"vid{i:08d}ab") for i in range(5)]
        api_resp = _make_api_response(items)
        fake_resp = _make_httpx_response(api_resp)
        scraper = YouTubeScraper()

        with patch("reviewmind.scrapers.youtube.httpx.Client") as mock_client:
            mock_instance = MagicMock()
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=False)
            mock_instance.get.return_value = fake_resp
            mock_client.return_value = mock_instance

            results = scraper.search_videos("headphones", api_key="k")

        for v in results:
            assert YouTubeScraper.is_youtube_url(v.url)

    def test_empty_api_response(self):
        scraper = YouTubeScraper()
        api_resp = {"kind": "youtube#searchListResponse", "items": []}
        fake_resp = _make_httpx_response(api_resp)

        with patch("reviewmind.scrapers.youtube.httpx.Client") as mock_client:
            mock_instance = MagicMock()
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=False)
            mock_instance.get.return_value = fake_resp
            mock_client.return_value = mock_instance

            results = scraper.search_videos("nothing here", api_key="k")

        assert results == []
