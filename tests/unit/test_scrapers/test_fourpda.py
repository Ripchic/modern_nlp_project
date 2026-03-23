"""Tests for the 4PDA forum scraper — pinned post, specs, and link extraction."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reviewmind.scrapers.fourpda import (
    ExtractedLink,
    Forum4PDAScraper,
    ForumPost4PDA,
    ForumTopic4PDA,
    PinnedPostData,
    _clean_post_text,
)


# ── Fixtures ──────────────────────────────────────────────────


def _build_topic_html(
    *,
    title: str = "Samsung Galaxy S24 Ultra",
    pinned_body: str = "",
    pinned_links: list[tuple[str, str]] | None = None,
    extra_posts: list[tuple[str, str]] | None = None,
    specs_table: bool = False,
    specs_list: bool = False,
    specs_kv_lines: bool = False,
) -> str:
    """Build a minimal 4PDA topic page HTML for testing."""
    links_html = ""
    if pinned_links:
        for href, text in pinned_links:
            links_html += f'<a href="{href}">{text}</a>\n'

    table_html = ""
    if specs_table:
        table_html = """
        <table>
          <tr><td>Процессор</td><td>Snapdragon 8 Gen 3</td></tr>
          <tr><td>Экран</td><td>6.8" Dynamic AMOLED 2X</td></tr>
          <tr><td>Память</td><td>12 ГБ / 256 ГБ</td></tr>
          <tr><td>Батарея</td><td>5000 мАч</td></tr>
        </table>
        """

    list_html = ""
    if specs_list:
        list_html = """
        <ul>
          <li>Процессор: Snapdragon 8 Gen 3</li>
          <li>Экран: AMOLED 6.8"</li>
          <li>Камера: 200 Мп</li>
        </ul>
        """

    kv_html = ""
    if specs_kv_lines:
        kv_html = """
        <p>Процессор: Snapdragon 8 Gen 3</p>
        <p>Экран: 6.8" AMOLED</p>
        <p>Батарея: 5000 мАч</p>
        """

    posts_html = ""
    if extra_posts:
        for i, (author, body) in enumerate(extra_posts, start=2):
            posts_html += f"""
            <div id="post-{1000 + i}" class="post">
              <span class="post_author">{author}</span>
              <div class="post_body">{body}</div>
            </div>
            """

    return f"""
    <html>
    <head><title>{title} - 4PDA</title></head>
    <body>
      <h1 class="topic-title">{title}</h1>
      <div id="post-100001" class="post">
        <span class="post_author">TopicStarter</span>
        <div class="post_body">
          {pinned_body}
          {table_html}
          {list_html}
          {kv_html}
          {links_html}
        </div>
      </div>
      {posts_html}
    </body>
    </html>
    """


# ── Basic parsing ─────────────────────────────────────────────


class TestParseTopicHTML:
    """Tests for _parse_topic_html including pinned post extraction."""

    def test_basic_topic_parsing(self) -> None:
        scraper = Forum4PDAScraper(follow_pinned_links=False)
        html = _build_topic_html(
            pinned_body="Это описание устройства Samsung Galaxy S24 Ultra с подробной информацией.",
        )
        topic = scraper._parse_topic_html(html, "https://4pda.to/forum/index.php?showtopic=123")
        assert topic.title == "Samsung Galaxy S24 Ultra"
        assert topic.topic_id == "123"
        assert len(topic.posts) >= 1
        assert topic.pinned is not None

    def test_pinned_post_has_post_id(self) -> None:
        scraper = Forum4PDAScraper(follow_pinned_links=False)
        html = _build_topic_html(pinned_body="Описание устройства для тестирования парсера.")
        topic = scraper._parse_topic_html(html, "https://4pda.to/forum/index.php?showtopic=1")
        assert topic.pinned is not None
        assert topic.pinned.post_id == "100001"


# ── Specs extraction ─────────────────────────────────────────


class TestSpecsExtraction:
    """Tests for device specifications extraction from pinned post."""

    def test_specs_from_table(self) -> None:
        scraper = Forum4PDAScraper(follow_pinned_links=False)
        html = _build_topic_html(
            pinned_body="Характеристики устройства:",
            specs_table=True,
        )
        topic = scraper._parse_topic_html(html, "https://4pda.to/forum/index.php?showtopic=1")
        assert topic.pinned is not None
        assert "Snapdragon 8 Gen 3" in topic.pinned.specs_text
        assert "5000" in topic.pinned.specs_text

    def test_specs_from_list(self) -> None:
        scraper = Forum4PDAScraper(follow_pinned_links=False)
        html = _build_topic_html(
            pinned_body="Основные характеристики:",
            specs_list=True,
        )
        topic = scraper._parse_topic_html(html, "https://4pda.to/forum/index.php?showtopic=1")
        assert topic.pinned is not None
        assert "200 Мп" in topic.pinned.specs_text

    def test_specs_from_kv_lines(self) -> None:
        scraper = Forum4PDAScraper(follow_pinned_links=False)
        html = _build_topic_html(
            pinned_body="Общая информация.",
            specs_kv_lines=True,
        )
        topic = scraper._parse_topic_html(html, "https://4pda.to/forum/index.php?showtopic=1")
        assert topic.pinned is not None
        assert "Процессор" in topic.pinned.specs_text
        assert "Snapdragon 8 Gen 3" in topic.pinned.specs_text

    def test_specs_in_full_text(self) -> None:
        scraper = Forum4PDAScraper(follow_pinned_links=False)
        html = _build_topic_html(specs_table=True, pinned_body="Описание устройства.")
        topic = scraper._parse_topic_html(html, "https://4pda.to/forum/index.php?showtopic=1")
        assert "[Характеристики]" in topic.full_text
        assert "Snapdragon 8 Gen 3" in topic.full_text


# ── Link extraction & classification ─────────────────────────


class TestLinkExtraction:
    """Tests for link extraction and classification from pinned post."""

    def test_review_link_classified(self) -> None:
        scraper = Forum4PDAScraper(follow_pinned_links=False)
        html = _build_topic_html(
            pinned_body="Полезные ссылки:",
            pinned_links=[
                ("https://4pda.to/2024/01/15/review-s24/", "Обзор Samsung Galaxy S24 Ultra"),
                ("https://example.com/review", "Full Review"),
            ],
        )
        topic = scraper._parse_topic_html(html, "https://4pda.to/forum/index.php?showtopic=1")
        assert topic.pinned is not None
        review_links = topic.pinned.review_links
        assert len(review_links) >= 2
        for link in review_links:
            assert link.category == "review"

    def test_specs_link_classified(self) -> None:
        scraper = Forum4PDAScraper(follow_pinned_links=False)
        html = _build_topic_html(
            pinned_body="Ссылки:",
            pinned_links=[
                ("https://example.com/specs", "Характеристики"),
            ],
        )
        topic = scraper._parse_topic_html(html, "https://4pda.to/forum/index.php?showtopic=1")
        assert topic.pinned is not None
        assert any(ln.category == "specs" for ln in topic.pinned.links)

    def test_firmware_link_classified(self) -> None:
        scraper = Forum4PDAScraper(follow_pinned_links=False)
        html = _build_topic_html(
            pinned_body="Прошивки:",
            pinned_links=[
                ("https://4pda.to/forum/index.php?showtopic=999", "Прошивки и root"),
            ],
        )
        topic = scraper._parse_topic_html(html, "https://4pda.to/forum/index.php?showtopic=1")
        assert topic.pinned is not None
        assert any(ln.category == "firmware" for ln in topic.pinned.links)

    def test_discussion_link_classified(self) -> None:
        scraper = Forum4PDAScraper(follow_pinned_links=False)
        html = _build_topic_html(
            pinned_body="Обсуждения:",
            pinned_links=[
                ("https://4pda.to/forum/index.php?showtopic=555", "Тема обсуждения"),
            ],
        )
        topic = scraper._parse_topic_html(html, "https://4pda.to/forum/index.php?showtopic=1")
        assert topic.pinned is not None
        assert any(ln.category == "discussion" for ln in topic.pinned.links)

    def test_duplicate_links_deduplicated(self) -> None:
        scraper = Forum4PDAScraper(follow_pinned_links=False)
        html = _build_topic_html(
            pinned_body="Ссылки:",
            pinned_links=[
                ("https://example.com/review", "Обзор 1"),
                ("https://example.com/review", "Обзор 2"),
            ],
        )
        topic = scraper._parse_topic_html(html, "https://4pda.to/forum/index.php?showtopic=1")
        assert topic.pinned is not None
        assert len(topic.pinned.links) == 1

    def test_js_and_anchor_links_skipped(self) -> None:
        scraper = Forum4PDAScraper(follow_pinned_links=False)
        html = _build_topic_html(
            pinned_body="Links:",
            pinned_links=[
                ("#top", "Наверх"),
                ("javascript:void(0)", "Click"),
                ("https://example.com/review", "Обзор"),
            ],
        )
        topic = scraper._parse_topic_html(html, "https://4pda.to/forum/index.php?showtopic=1")
        assert topic.pinned is not None
        assert len(topic.pinned.links) == 1


# ── Link classification (unit) ───────────────────────────────


class TestClassifyLink:
    def test_review_by_text(self) -> None:
        assert Forum4PDAScraper._classify_link("https://x.com/page", "Обзор S24") == "review"

    def test_review_by_url(self) -> None:
        assert Forum4PDAScraper._classify_link("https://x.com/review/123", "") == "review"

    def test_specs(self) -> None:
        assert Forum4PDAScraper._classify_link("https://x.com", "Характеристики") == "specs"

    def test_firmware(self) -> None:
        assert Forum4PDAScraper._classify_link("https://x.com", "Прошивка и root") == "firmware"

    def test_discussion(self) -> None:
        assert (
            Forum4PDAScraper._classify_link(
                "https://4pda.to/forum/index.php?showtopic=999", "Общая тема"
            )
            == "discussion"
        )

    def test_other(self) -> None:
        assert Forum4PDAScraper._classify_link("https://example.com", "Купить") == "other"


# ── Review link following ─────────────────────────────────────


class TestFollowReviewLinks:
    """Tests for following review links from pinned post."""

    def test_sync_follows_review_links(self) -> None:
        scraper = Forum4PDAScraper(follow_pinned_links=True, max_followed_links=3)
        html = _build_topic_html(
            pinned_body="Шапка темы с описанием устройства.",
            pinned_links=[
                ("https://example.com/review-1", "Обзор #1"),
                ("https://example.com/review-2", "Обзор #2"),
                ("https://example.com/review-3", "Обзор #3"),  # should be skipped (limit=2)
            ],
        )
        # Mock _get_sync for the topic page
        with patch.object(scraper, "_get_sync", return_value=html) as mock_get:
            # Mock _fetch_review_text_sync for review URLs
            with patch.object(
                scraper,
                "_fetch_review_text_sync",
                return_value="This is a detailed review of the device with many useful insights and comparisons.",
            ) as mock_fetch:
                topic = scraper.parse_topic_sync("https://4pda.to/forum/index.php?showtopic=1")

        assert topic is not None
        assert topic.pinned is not None
        # Only 2 reviews should be followed (max_followed_links=2)
        assert len(topic.pinned.review_texts) == 3
        assert mock_fetch.call_count == 3

    def test_sync_skips_short_review_text(self) -> None:
        scraper = Forum4PDAScraper(follow_pinned_links=True)
        html = _build_topic_html(
            pinned_body="Шапка темы с описанием устройства.",
            pinned_links=[("https://example.com/review", "Обзор")],
        )
        with patch.object(scraper, "_get_sync", return_value=html):
            with patch.object(scraper, "_fetch_review_text_sync", return_value="Short"):
                topic = scraper.parse_topic_sync("https://4pda.to/forum/index.php?showtopic=1")

        assert topic is not None
        assert topic.pinned is not None
        assert len(topic.pinned.review_texts) == 0

    def test_review_text_in_full_text(self) -> None:
        scraper = Forum4PDAScraper(follow_pinned_links=True)
        html = _build_topic_html(
            pinned_body="Шапка темы с описанием устройства.",
            pinned_links=[("https://example.com/review", "Обзор")],
        )
        review_content = "Подробный обзор устройства с анализом производительности, камеры и батареи."
        with patch.object(scraper, "_get_sync", return_value=html):
            with patch.object(scraper, "_fetch_review_text_sync", return_value=review_content):
                topic = scraper.parse_topic_sync("https://4pda.to/forum/index.php?showtopic=1")

        assert topic is not None
        assert "[Обзор]" in topic.full_text
        assert review_content in topic.full_text

    @pytest.mark.asyncio
    async def test_async_follows_review_links(self) -> None:
        scraper = Forum4PDAScraper(follow_pinned_links=True, max_followed_links=1)
        html = _build_topic_html(
            pinned_body="Шапка темы с описанием устройства.",
            pinned_links=[("https://example.com/review", "Обзор")],
        )
        review_content = "Detailed review content that is long enough to pass the minimum length filter threshold."
        with patch.object(scraper, "_get_async", new_callable=AsyncMock, return_value=html):
            with patch.object(
                scraper,
                "_fetch_review_text_async",
                new_callable=AsyncMock,
                return_value=review_content,
            ):
                topic = await scraper.parse_topic("https://4pda.to/forum/index.php?showtopic=1")

        assert topic is not None
        assert topic.pinned is not None
        assert len(topic.pinned.review_texts) == 1

    def test_follow_disabled(self) -> None:
        scraper = Forum4PDAScraper(follow_pinned_links=False)
        html = _build_topic_html(
            pinned_body="Шапка темы с описанием устройства.",
            pinned_links=[("https://example.com/review", "Обзор")],
        )
        with patch.object(scraper, "_get_sync", return_value=html):
            topic = scraper.parse_topic_sync("https://4pda.to/forum/index.php?showtopic=1")

        assert topic is not None
        assert topic.pinned is not None
        assert len(topic.pinned.review_texts) == 0


# ── Fetch review text helpers ─────────────────────────────────


class TestFetchReviewText:
    """Tests for _fetch_review_text_sync / _fetch_review_text_async."""

    def test_fetch_4pda_post_by_id(self) -> None:
        """When a 4PDA link has ?p=NNNNN, extract just that post."""
        scraper = Forum4PDAScraper(follow_pinned_links=False)
        post_html = """
        <html><body>
          <div id="post-12345" class="post">
            <div class="post_body">Подробный обзор — Samsung Galaxy S24 Ultra показывает
            отличную производительность в синтетических тестах.</div>
          </div>
        </body></html>
        """
        with patch.object(scraper, "_get_sync", return_value=post_html):
            text = scraper._fetch_review_text_sync(
                "https://4pda.to/forum/index.php?showtopic=1&view=findpost&p=12345"
            )
        assert text is not None
        assert "Samsung Galaxy S24 Ultra" in text

    def test_fetch_external_url(self) -> None:
        """External URLs use the WebScraper."""
        scraper = Forum4PDAScraper(follow_pinned_links=False)
        mock_page = MagicMock()
        mock_page.text = "External review content about the device."
        with patch.object(scraper._web_scraper, "parse_url", return_value=mock_page):
            text = scraper._fetch_review_text_sync("https://ixbt.com/review/123")
        assert text == "External review content about the device."

    def test_fetch_external_url_fails(self) -> None:
        scraper = Forum4PDAScraper(follow_pinned_links=False)
        with patch.object(scraper._web_scraper, "parse_url", return_value=None):
            text = scraper._fetch_review_text_sync("https://ixbt.com/broken")
        assert text is None


# ── Dataclass properties ──────────────────────────────────────


class TestPinnedPostData:
    def test_review_links_filter(self) -> None:
        pinned = PinnedPostData(
            links=[
                ExtractedLink(url="https://a.com", text="Обзор", category="review"),
                ExtractedLink(url="https://b.com", text="Прошивка", category="firmware"),
                ExtractedLink(url="https://c.com", text="Review 2", category="review"),
            ]
        )
        assert len(pinned.review_links) == 2


class TestCleanPostText:
    def test_removes_invisible_chars(self) -> None:
        assert _clean_post_text("hello\u200bworld") == "helloworld"

    def test_collapses_whitespace(self) -> None:
        assert _clean_post_text("a   b") == "a b"

    def test_collapses_newlines(self) -> None:
        assert _clean_post_text("a\n\n\n\nb") == "a\n\nb"
