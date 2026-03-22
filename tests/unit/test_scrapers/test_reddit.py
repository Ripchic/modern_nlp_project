"""Unit tests for reviewmind.scrapers.reddit — RedditScraper."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from reviewmind.scrapers.reddit import (
    _DELETED_MARKERS,
    _REDDIT_URL_PATTERNS,
    DEFAULT_COMMENT_LIMIT,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_SUBREDDIT,
    RedditComment,
    RedditPost,
    RedditScraper,
    SearchResult,
)

# ═══════════════════════════════════════════════════════════════
#  Helpers — Fake PRAW objects
# ═══════════════════════════════════════════════════════════════


def _fake_author(name: str) -> object:
    """Create a fake PRAW author-like object whose str() returns *name*."""
    return type("FakeAuthor", (), {"__str__": lambda self: name, "__repr__": lambda self: name})()


def _make_fake_comment(
    body: str = "Great comment",
    author: str = "user1",
    score: int = 10,
    replies: list | None = None,
) -> SimpleNamespace:
    """Create a fake PRAW comment-like object."""
    return SimpleNamespace(
        body=body,
        author=_fake_author(author) if author else None,
        score=score,
        replies=replies or [],
    )


def _make_fake_submission(
    post_id: str = "abc123",
    title: str = "Best headphones 2026",
    selftext: str = "Looking for recommendations on noise-cancelling headphones.",
    author: str = "testuser",
    score: int = 42,
    upvote_ratio: float = 0.95,
    num_comments: int = 15,
    subreddit: str = "headphones",
    url: str = "https://www.reddit.com/r/headphones/comments/abc123/best_headphones/",
    permalink: str = "/r/headphones/comments/abc123/best_headphones/",
    created_utc: float = 1709300000.0,
    comments: list | None = None,
) -> SimpleNamespace:
    """Create a fake PRAW submission-like object."""
    if author:
        author_obj = _fake_author(author)
    else:
        author_obj = None

    comment_list = comments if comments is not None else []

    # Create a mock comment forest
    comment_forest = MagicMock()
    comment_forest.replace_more = MagicMock()
    comment_forest.__iter__ = lambda self: iter(comment_list)

    subreddit_obj = _fake_author(subreddit)  # reuse helper for str() behavior

    return SimpleNamespace(
        id=post_id,
        title=title,
        selftext=selftext,
        author=author_obj,
        score=score,
        upvote_ratio=upvote_ratio,
        num_comments=num_comments,
        subreddit=subreddit_obj,
        url=url,
        permalink=permalink,
        created_utc=created_utc,
        comments=comment_forest,
    )


# ═══════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════


class TestConstants:
    def test_default_comment_limit(self):
        assert DEFAULT_COMMENT_LIMIT == 20

    def test_default_search_limit(self):
        assert DEFAULT_SEARCH_LIMIT == 10

    def test_default_subreddit(self):
        assert DEFAULT_SUBREDDIT == "all"

    def test_deleted_markers(self):
        assert "[deleted]" in _DELETED_MARKERS
        assert "[removed]" in _DELETED_MARKERS
        assert "[удалено]" in _DELETED_MARKERS

    def test_deleted_markers_is_frozenset(self):
        assert isinstance(_DELETED_MARKERS, frozenset)

    def test_url_patterns_count(self):
        assert len(_REDDIT_URL_PATTERNS) >= 3


# ═══════════════════════════════════════════════════════════════
#  RedditComment dataclass
# ═══════════════════════════════════════════════════════════════


class TestRedditComment:
    def test_basic_creation(self):
        c = RedditComment(author="user1", body="Good stuff", score=5, is_top_level=True)
        assert c.author == "user1"
        assert c.body == "Good stuff"
        assert c.score == 5
        assert c.is_top_level is True

    def test_reply_comment(self):
        c = RedditComment(author="user2", body="Reply", score=2, is_top_level=False)
        assert c.is_top_level is False

    def test_negative_score(self):
        c = RedditComment(author="troll", body="bad take", score=-10, is_top_level=True)
        assert c.score == -10


# ═══════════════════════════════════════════════════════════════
#  RedditPost dataclass
# ═══════════════════════════════════════════════════════════════


class TestRedditPost:
    def test_basic_creation(self):
        p = RedditPost(
            post_id="abc123",
            title="Title",
            selftext="Body text",
            author="user",
            score=100,
            upvote_ratio=0.9,
            num_comments=10,
            subreddit="gadgets",
            url="https://...",
            permalink="https://...",
            created_utc=1709300000.0,
        )
        assert p.post_id == "abc123"
        assert p.title == "Title"
        assert p.selftext == "Body text"
        assert p.score == 100

    def test_default_comments_empty(self):
        p = RedditPost(
            post_id="x",
            title="T",
            selftext="B",
            author="u",
            score=0,
            upvote_ratio=0.5,
            num_comments=0,
            subreddit="s",
            url="u",
            permalink="p",
            created_utc=0.0,
        )
        assert p.comments == []

    def test_default_source_url(self):
        p = RedditPost(
            post_id="x",
            title="T",
            selftext="B",
            author="u",
            score=0,
            upvote_ratio=0.5,
            num_comments=0,
            subreddit="s",
            url="u",
            permalink="p",
            created_utc=0.0,
        )
        assert p.source_url == ""

    def test_default_extra_metadata(self):
        p = RedditPost(
            post_id="x",
            title="T",
            selftext="B",
            author="u",
            score=0,
            upvote_ratio=0.5,
            num_comments=0,
            subreddit="s",
            url="u",
            permalink="p",
            created_utc=0.0,
        )
        assert p.extra_metadata == {}

    def test_full_text_with_title_and_selftext(self):
        p = RedditPost(
            post_id="x",
            title="My Title",
            selftext="My body text",
            author="u",
            score=0,
            upvote_ratio=0.5,
            num_comments=0,
            subreddit="s",
            url="u",
            permalink="p",
            created_utc=0.0,
        )
        assert "My Title" in p.full_text
        assert "My body text" in p.full_text

    def test_full_text_includes_comments(self):
        comments = [
            RedditComment(author="a", body="Comment one", score=1, is_top_level=True),
            RedditComment(author="b", body="Comment two", score=2, is_top_level=False),
        ]
        p = RedditPost(
            post_id="x",
            title="Title",
            selftext="Body",
            author="u",
            score=0,
            upvote_ratio=0.5,
            num_comments=2,
            subreddit="s",
            url="u",
            permalink="p",
            created_utc=0.0,
            comments=comments,
        )
        assert "Comment one" in p.full_text
        assert "Comment two" in p.full_text

    def test_full_text_skips_deleted_selftext(self):
        p = RedditPost(
            post_id="x",
            title="Title",
            selftext="[deleted]",
            author="u",
            score=0,
            upvote_ratio=0.5,
            num_comments=0,
            subreddit="s",
            url="u",
            permalink="p",
            created_utc=0.0,
        )
        assert "[deleted]" not in p.full_text
        assert "Title" in p.full_text

    def test_full_text_skips_deleted_comments(self):
        comments = [
            RedditComment(author="a", body="Good", score=1, is_top_level=True),
            RedditComment(author="b", body="[deleted]", score=0, is_top_level=True),
            RedditComment(author="c", body="[removed]", score=0, is_top_level=True),
        ]
        p = RedditPost(
            post_id="x",
            title="T",
            selftext="B",
            author="u",
            score=0,
            upvote_ratio=0.5,
            num_comments=3,
            subreddit="s",
            url="u",
            permalink="p",
            created_utc=0.0,
            comments=comments,
        )
        assert "Good" in p.full_text
        assert "[deleted]" not in p.full_text
        assert "[removed]" not in p.full_text

    def test_full_text_empty_selftext(self):
        p = RedditPost(
            post_id="x",
            title="Title Only",
            selftext="",
            author="u",
            score=0,
            upvote_ratio=0.5,
            num_comments=0,
            subreddit="s",
            url="u",
            permalink="p",
            created_utc=0.0,
        )
        assert p.full_text == "Title Only"


# ═══════════════════════════════════════════════════════════════
#  SearchResult dataclass
# ═══════════════════════════════════════════════════════════════


class TestSearchResult:
    def test_basic_creation(self):
        r = SearchResult(
            post_id="xyz",
            title="Review",
            selftext="Content",
            author="reviewer",
            score=50,
            num_comments=5,
            subreddit="tech",
            url="https://...",
            permalink="https://...",
            created_utc=1709300000.0,
        )
        assert r.post_id == "xyz"
        assert r.title == "Review"
        assert r.score == 50

    def test_all_fields(self):
        r = SearchResult(
            post_id="id1",
            title="T",
            selftext="S",
            author="A",
            score=0,
            num_comments=0,
            subreddit="sub",
            url="url",
            permalink="perm",
            created_utc=0.0,
        )
        assert r.subreddit == "sub"
        assert r.permalink == "perm"


# ═══════════════════════════════════════════════════════════════
#  extract_post_id
# ═══════════════════════════════════════════════════════════════


class TestExtractPostId:
    """Test all supported Reddit URL formats."""

    @pytest.mark.parametrize(
        "url, expected_id",
        [
            # Standard reddit.com URL
            ("https://www.reddit.com/r/headphones/comments/abc123/best_headphones/", "abc123"),
            ("http://www.reddit.com/r/gadgets/comments/xyz789/cool_gadget/", "xyz789"),
            ("https://reddit.com/r/tech/comments/def456/review/", "def456"),
            # Without trailing slash
            ("https://www.reddit.com/r/headphones/comments/abc123/best_headphones", "abc123"),
            # With query params
            ("https://www.reddit.com/r/headphones/comments/abc123/best_headphones/?sort=top", "abc123"),
            # Old reddit
            ("https://old.reddit.com/r/headphones/comments/abc123/best_headphones/", "abc123"),
            ("http://old.reddit.com/r/gadgets/comments/def456/review/", "def456"),
            # Short URL
            ("https://redd.it/abc123", "abc123"),
            ("http://redd.it/xyz789", "xyz789"),
            # With whitespace
            ("  https://www.reddit.com/r/tech/comments/abc123/test/  ", "abc123"),
        ],
    )
    def test_valid_urls(self, url: str, expected_id: str):
        assert RedditScraper.extract_post_id(url) == expected_id

    @pytest.mark.parametrize(
        "url",
        [
            "",
            "not a url",
            "https://google.com",
            "https://youtube.com/watch?v=abc",
            "https://www.reddit.com/r/headphones/",  # no /comments/
            "https://www.reddit.com/r/headphones/top/",  # subreddit page, not post
        ],
    )
    def test_invalid_urls_raise_value_error(self, url: str):
        with pytest.raises(ValueError):
            RedditScraper.extract_post_id(url)

    def test_none_url_raises(self):
        with pytest.raises(ValueError):
            RedditScraper.extract_post_id(None)  # type: ignore[arg-type]

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            RedditScraper.extract_post_id("")

    def test_non_string_raises(self):
        with pytest.raises(ValueError):
            RedditScraper.extract_post_id(12345)  # type: ignore[arg-type]


# ═══════════════════════════════════════════════════════════════
#  is_reddit_url
# ═══════════════════════════════════════════════════════════════


class TestIsRedditUrl:
    @pytest.mark.parametrize(
        "url",
        [
            "https://www.reddit.com/r/headphones/comments/abc/post/",
            "https://reddit.com/r/tech/",
            "https://old.reddit.com/r/gadgets/",
            "https://redd.it/abc123",
            "http://www.reddit.com/r/test/",
        ],
    )
    def test_valid_reddit_urls(self, url: str):
        assert RedditScraper.is_reddit_url(url) is True

    @pytest.mark.parametrize(
        "url",
        [
            "",
            None,
            123,
            "https://google.com",
            "https://youtube.com/watch?v=abc",
            "not a url",
        ],
    )
    def test_non_reddit_urls(self, url):
        assert RedditScraper.is_reddit_url(url) is False


# ═══════════════════════════════════════════════════════════════
#  RedditScraper init
# ═══════════════════════════════════════════════════════════════


class TestRedditScraperInit:
    def test_explicit_credentials(self):
        scraper = RedditScraper(
            client_id="id",
            client_secret="secret",
            user_agent="ua/1.0",
        )
        assert scraper._client_id == "id"
        assert scraper._client_secret == "secret"
        assert scraper._user_agent == "ua/1.0"

    def test_default_comment_limit(self):
        scraper = RedditScraper(
            client_id="id",
            client_secret="secret",
            user_agent="ua",
        )
        assert scraper._comment_limit == 20

    def test_custom_comment_limit(self):
        scraper = RedditScraper(
            client_id="id",
            client_secret="secret",
            user_agent="ua",
            comment_limit=5,
        )
        assert scraper._comment_limit == 5

    def test_reddit_lazy_not_created(self):
        scraper = RedditScraper(
            client_id="id",
            client_secret="secret",
            user_agent="ua",
        )
        assert scraper._reddit is None

    def test_fallback_to_config(self):
        mock_settings = SimpleNamespace(
            reddit_client_id="cfg_id",
            reddit_client_secret="cfg_secret",
            reddit_user_agent="cfg_ua",
        )
        with patch("reviewmind.scrapers.reddit.settings", mock_settings, create=True):
            # Import path for lazy settings
            with patch("reviewmind.config.settings", mock_settings):
                scraper = RedditScraper()
        assert scraper._client_id == "cfg_id"
        assert scraper._client_secret == "cfg_secret"
        assert scraper._user_agent == "cfg_ua"

    def test_reddit_property_creates_instance(self):
        scraper = RedditScraper(
            client_id="id",
            client_secret="secret",
            user_agent="ua",
        )
        with patch("praw.Reddit") as mock_reddit_cls:
            mock_reddit_cls.return_value = MagicMock()
            reddit = scraper.reddit
            mock_reddit_cls.assert_called_once_with(
                client_id="id",
                client_secret="secret",
                user_agent="ua",
            )
            assert reddit is mock_reddit_cls.return_value

    def test_reddit_property_cached(self):
        scraper = RedditScraper(
            client_id="id",
            client_secret="secret",
            user_agent="ua",
        )
        with patch("praw.Reddit") as mock_reddit_cls:
            mock_instance = MagicMock()
            mock_reddit_cls.return_value = mock_instance
            r1 = scraper.reddit
            r2 = scraper.reddit
            # Only created once
            mock_reddit_cls.assert_called_once()
            assert r1 is r2


# ═══════════════════════════════════════════════════════════════
#  _is_deleted_submission
# ═══════════════════════════════════════════════════════════════


class TestIsDeletedSubmission:
    def test_deleted_author_none(self):
        sub = SimpleNamespace(author=None, selftext="text", title="Title")
        assert RedditScraper._is_deleted_submission(sub) is True

    def test_normal_post_not_deleted(self):
        sub = SimpleNamespace(
            author=_fake_author("user"),
            selftext="normal text",
            title="Title",
        )
        assert RedditScraper._is_deleted_submission(sub) is False

    def test_deleted_selftext_with_title_not_deleted(self):
        """Post with [deleted] body but valid title is still usable."""
        sub = SimpleNamespace(
            author=_fake_author("user"),
            selftext="[deleted]",
            title="Has a Title",
        )
        assert RedditScraper._is_deleted_submission(sub) is False

    def test_deleted_selftext_and_empty_title_is_deleted(self):
        sub = SimpleNamespace(
            author=_fake_author("user"),
            selftext="[deleted]",
            title="",
        )
        assert RedditScraper._is_deleted_submission(sub) is True

    def test_removed_selftext_and_empty_title_is_deleted(self):
        sub = SimpleNamespace(
            author=_fake_author("user"),
            selftext="[removed]",
            title="  ",
        )
        assert RedditScraper._is_deleted_submission(sub) is True

    def test_russian_deleted_marker(self):
        sub = SimpleNamespace(
            author=_fake_author("user"),
            selftext="[удалено]",
            title="",
        )
        assert RedditScraper._is_deleted_submission(sub) is True


# ═══════════════════════════════════════════════════════════════
#  _collect_comments
# ═══════════════════════════════════════════════════════════════


class TestCollectComments:
    def _make_scraper(self, comment_limit: int = 20) -> RedditScraper:
        return RedditScraper(
            client_id="id",
            client_secret="secret",
            user_agent="ua",
            comment_limit=comment_limit,
        )

    def test_collects_top_level_comments(self):
        scraper = self._make_scraper()
        comments = [
            SimpleNamespace(body="Comment 1", author=_fake_author("u1"), score=10, replies=[]),
            SimpleNamespace(body="Comment 2", author=_fake_author("u2"), score=20, replies=[]),
        ]
        sub = _make_fake_submission(comments=comments)
        result = scraper._collect_comments(sub)
        assert len(result) == 2
        assert result[0].body == "Comment 1"
        assert result[0].is_top_level is True
        assert result[1].body == "Comment 2"

    def test_collects_first_level_replies(self):
        scraper = self._make_scraper()
        reply = SimpleNamespace(
            body="Reply to comment",
            author=_fake_author("r1"),
            score=5,
            replies=[],
        )
        top_comment = SimpleNamespace(
            body="Top comment",
            author=_fake_author("u1"),
            score=10,
            replies=[reply],
        )
        sub = _make_fake_submission(comments=[top_comment])
        result = scraper._collect_comments(sub)
        assert len(result) == 2
        assert result[0].is_top_level is True
        assert result[0].body == "Top comment"
        assert result[1].is_top_level is False
        assert result[1].body == "Reply to comment"

    def test_respects_comment_limit(self):
        scraper = self._make_scraper(comment_limit=3)
        comments = [
            SimpleNamespace(body=f"Comment {i}", author=_fake_author(f"u{i}"), score=i, replies=[]) for i in range(10)
        ]
        sub = _make_fake_submission(comments=comments)
        result = scraper._collect_comments(sub)
        assert len(result) == 3

    def test_skips_deleted_comments(self):
        scraper = self._make_scraper()
        comments = [
            SimpleNamespace(body="Good comment", author=_fake_author("u1"), score=10, replies=[]),
            SimpleNamespace(body="[deleted]", author=_fake_author("u2"), score=0, replies=[]),
            SimpleNamespace(body="[removed]", author=_fake_author("u3"), score=0, replies=[]),
            SimpleNamespace(
                body="Another good one",
                author=_fake_author("u4"),
                score=5,
                replies=[],
            ),
        ]
        sub = _make_fake_submission(comments=comments)
        result = scraper._collect_comments(sub)
        assert len(result) == 2
        assert result[0].body == "Good comment"
        assert result[1].body == "Another good one"

    def test_skips_empty_comments(self):
        scraper = self._make_scraper()
        comments = [
            SimpleNamespace(body="", author=_fake_author("u1"), score=0, replies=[]),
            SimpleNamespace(body="  ", author=_fake_author("u2"), score=0, replies=[]),
            SimpleNamespace(body="Real comment", author=_fake_author("u3"), score=5, replies=[]),
        ]
        sub = _make_fake_submission(comments=comments)
        result = scraper._collect_comments(sub)
        assert len(result) == 1
        assert result[0].body == "Real comment"

    def test_deleted_author_in_comment(self):
        scraper = self._make_scraper()
        comments = [
            SimpleNamespace(body="No author comment", author=None, score=3, replies=[]),
        ]
        sub = _make_fake_submission(comments=comments)
        result = scraper._collect_comments(sub)
        assert len(result) == 1
        assert result[0].author == "[deleted]"

    def test_empty_comments_list(self):
        scraper = self._make_scraper()
        sub = _make_fake_submission(comments=[])
        result = scraper._collect_comments(sub)
        assert result == []

    def test_replace_more_failure_graceful(self):
        scraper = self._make_scraper()
        comments = [
            SimpleNamespace(body="Comment", author=_fake_author("u1"), score=5, replies=[]),
        ]
        sub = _make_fake_submission(comments=comments)
        sub.comments.replace_more.side_effect = RuntimeError("network error")
        result = scraper._collect_comments(sub)
        # Should still collect available comments
        assert len(result) == 1

    def test_comment_limit_includes_replies(self):
        """comment_limit counts both top-level AND reply comments."""
        scraper = self._make_scraper(comment_limit=2)
        reply = SimpleNamespace(
            body="Reply",
            author=_fake_author("r1"),
            score=1,
            replies=[],
        )
        top1 = SimpleNamespace(
            body="Top 1",
            author=_fake_author("u1"),
            score=10,
            replies=[reply],
        )
        top2 = SimpleNamespace(
            body="Top 2",
            author=_fake_author("u2"),
            score=5,
            replies=[],
        )
        sub = _make_fake_submission(comments=[top1, top2])
        result = scraper._collect_comments(sub)
        # Limit is 2: top1 + reply = 2, top2 skipped
        assert len(result) == 2
        assert result[0].body == "Top 1"
        assert result[1].body == "Reply"


# ═══════════════════════════════════════════════════════════════
#  get_post
# ═══════════════════════════════════════════════════════════════


class TestGetPost:
    def _make_scraper_with_mock(self):
        scraper = RedditScraper(
            client_id="id",
            client_secret="secret",
            user_agent="ua",
        )
        mock_reddit = MagicMock()
        scraper._reddit = mock_reddit
        return scraper, mock_reddit

    def test_returns_post(self):
        scraper, mock_reddit = self._make_scraper_with_mock()
        sub = _make_fake_submission(
            post_id="abc123",
            title="Test Post",
            selftext="Post body",
            comments=[
                SimpleNamespace(body="Comment", author=_fake_author("u1"), score=5, replies=[]),
            ],
        )
        mock_reddit.submission.return_value = sub

        result = scraper.get_post("abc123")
        assert result is not None
        assert isinstance(result, RedditPost)
        assert result.post_id == "abc123"
        assert result.title == "Test Post"
        assert result.selftext == "Post body"
        assert len(result.comments) == 1

    def test_sets_source_url(self):
        scraper, mock_reddit = self._make_scraper_with_mock()
        sub = _make_fake_submission()
        mock_reddit.submission.return_value = sub

        result = scraper.get_post("abc123", source_url="https://example.com/post")
        assert result is not None
        assert result.source_url == "https://example.com/post"

    def test_default_source_url_is_permalink(self):
        scraper, mock_reddit = self._make_scraper_with_mock()
        sub = _make_fake_submission(permalink="/r/test/comments/abc123/slug/")
        mock_reddit.submission.return_value = sub

        result = scraper.get_post("abc123")
        assert result is not None
        assert result.source_url == "https://www.reddit.com/r/test/comments/abc123/slug/"

    def test_deleted_post_returns_none(self):
        scraper, mock_reddit = self._make_scraper_with_mock()
        sub = _make_fake_submission(author=None)
        # We need author to be None on the namespace
        sub.author = None
        mock_reddit.submission.return_value = sub

        result = scraper.get_post("abc123")
        assert result is None

    def test_fetch_error_returns_none(self):
        scraper, mock_reddit = self._make_scraper_with_mock()
        mock_reddit.submission.side_effect = RuntimeError("API error")

        result = scraper.get_post("abc123")
        assert result is None

    def test_invalid_post_id_none_returns_none(self):
        scraper, _ = self._make_scraper_with_mock()
        assert scraper.get_post(None) is None  # type: ignore[arg-type]

    def test_empty_post_id_returns_none(self):
        scraper, _ = self._make_scraper_with_mock()
        assert scraper.get_post("") is None

    def test_non_string_post_id_returns_none(self):
        scraper, _ = self._make_scraper_with_mock()
        assert scraper.get_post(12345) is None  # type: ignore[arg-type]

    def test_permalink_format(self):
        scraper, mock_reddit = self._make_scraper_with_mock()
        sub = _make_fake_submission(permalink="/r/tech/comments/xyz/nice/")
        mock_reddit.submission.return_value = sub

        result = scraper.get_post("xyz")
        assert result is not None
        assert result.permalink == "https://www.reddit.com/r/tech/comments/xyz/nice/"

    def test_post_metadata_fields(self):
        scraper, mock_reddit = self._make_scraper_with_mock()
        sub = _make_fake_submission(
            score=42,
            upvote_ratio=0.95,
            num_comments=15,
            subreddit="headphones",
            created_utc=1709300000.0,
        )
        mock_reddit.submission.return_value = sub

        result = scraper.get_post("abc123")
        assert result is not None
        assert result.score == 42
        assert result.upvote_ratio == 0.95
        assert result.num_comments == 15
        assert result.subreddit == "headphones"
        assert result.created_utc == 1709300000.0


# ═══════════════════════════════════════════════════════════════
#  parse_url
# ═══════════════════════════════════════════════════════════════


class TestParseUrl:
    def test_valid_url_calls_get_post(self):
        scraper = RedditScraper(
            client_id="id",
            client_secret="secret",
            user_agent="ua",
        )
        mock_reddit = MagicMock()
        scraper._reddit = mock_reddit
        sub = _make_fake_submission(post_id="abc123")
        mock_reddit.submission.return_value = sub

        result = scraper.parse_url("https://www.reddit.com/r/test/comments/abc123/slug/")
        assert result is not None
        assert result.post_id == "abc123"

    def test_invalid_url_returns_none(self):
        scraper = RedditScraper(
            client_id="id",
            client_secret="secret",
            user_agent="ua",
        )
        result = scraper.parse_url("https://google.com")
        assert result is None

    def test_sets_source_url_from_input(self):
        scraper = RedditScraper(
            client_id="id",
            client_secret="secret",
            user_agent="ua",
        )
        mock_reddit = MagicMock()
        scraper._reddit = mock_reddit
        sub = _make_fake_submission(post_id="abc123")
        mock_reddit.submission.return_value = sub

        url = "https://www.reddit.com/r/test/comments/abc123/slug/"
        result = scraper.parse_url(url)
        assert result is not None
        assert result.source_url == url

    def test_redd_it_url(self):
        scraper = RedditScraper(
            client_id="id",
            client_secret="secret",
            user_agent="ua",
        )
        mock_reddit = MagicMock()
        scraper._reddit = mock_reddit
        sub = _make_fake_submission(post_id="xyz789")
        mock_reddit.submission.return_value = sub

        result = scraper.parse_url("https://redd.it/xyz789")
        assert result is not None
        assert result.post_id == "xyz789"


# ═══════════════════════════════════════════════════════════════
#  search_subreddit
# ═══════════════════════════════════════════════════════════════


class TestSearchSubreddit:
    def _make_scraper_with_mock(self):
        scraper = RedditScraper(
            client_id="id",
            client_secret="secret",
            user_agent="ua",
        )
        mock_reddit = MagicMock()
        scraper._reddit = mock_reddit
        return scraper, mock_reddit

    def test_returns_results(self):
        scraper, mock_reddit = self._make_scraper_with_mock()
        sub1 = _make_fake_submission(post_id="p1", title="Post 1", score=100)
        sub2 = _make_fake_submission(post_id="p2", title="Post 2", score=50)

        mock_subreddit = MagicMock()
        mock_subreddit.search.return_value = [sub1, sub2]
        mock_reddit.subreddit.return_value = mock_subreddit

        results = scraper.search_subreddit("headphones review")
        assert len(results) == 2
        assert isinstance(results[0], SearchResult)
        assert results[0].post_id == "p1"
        assert results[0].title == "Post 1"
        assert results[1].post_id == "p2"

    def test_custom_subreddit(self):
        scraper, mock_reddit = self._make_scraper_with_mock()
        mock_subreddit = MagicMock()
        mock_subreddit.search.return_value = []
        mock_reddit.subreddit.return_value = mock_subreddit

        scraper.search_subreddit("query", subreddit="headphones")
        mock_reddit.subreddit.assert_called_once_with("headphones")

    def test_default_subreddit_all(self):
        scraper, mock_reddit = self._make_scraper_with_mock()
        mock_subreddit = MagicMock()
        mock_subreddit.search.return_value = []
        mock_reddit.subreddit.return_value = mock_subreddit

        scraper.search_subreddit("query")
        mock_reddit.subreddit.assert_called_once_with("all")

    def test_custom_limit(self):
        scraper, mock_reddit = self._make_scraper_with_mock()
        mock_subreddit = MagicMock()
        mock_subreddit.search.return_value = []
        mock_reddit.subreddit.return_value = mock_subreddit

        scraper.search_subreddit("query", limit=5)
        mock_subreddit.search.assert_called_once_with(
            "query",
            sort="relevance",
            time_filter="year",
            limit=5,
        )

    def test_filters_deleted_posts(self):
        scraper, mock_reddit = self._make_scraper_with_mock()
        sub_ok = _make_fake_submission(post_id="ok", title="Good post")
        sub_deleted = _make_fake_submission(post_id="del", title="Deleted", author=None)
        sub_deleted.author = None

        mock_subreddit = MagicMock()
        mock_subreddit.search.return_value = [sub_ok, sub_deleted]
        mock_reddit.subreddit.return_value = mock_subreddit

        results = scraper.search_subreddit("query")
        assert len(results) == 1
        assert results[0].post_id == "ok"

    def test_empty_query_returns_empty(self):
        scraper, _ = self._make_scraper_with_mock()
        assert scraper.search_subreddit("") == []

    def test_none_query_returns_empty(self):
        scraper, _ = self._make_scraper_with_mock()
        assert scraper.search_subreddit(None) == []  # type: ignore[arg-type]

    def test_whitespace_query_returns_empty(self):
        scraper, _ = self._make_scraper_with_mock()
        assert scraper.search_subreddit("   ") == []

    def test_api_error_returns_empty(self):
        scraper, mock_reddit = self._make_scraper_with_mock()
        mock_subreddit = MagicMock()
        mock_subreddit.search.side_effect = RuntimeError("API error")
        mock_reddit.subreddit.return_value = mock_subreddit

        results = scraper.search_subreddit("query")
        assert results == []

    def test_search_params_passed(self):
        scraper, mock_reddit = self._make_scraper_with_mock()
        mock_subreddit = MagicMock()
        mock_subreddit.search.return_value = []
        mock_reddit.subreddit.return_value = mock_subreddit

        scraper.search_subreddit(
            "query",
            sort="top",
            time_filter="month",
            limit=20,
        )
        mock_subreddit.search.assert_called_once_with(
            "query",
            sort="top",
            time_filter="month",
            limit=20,
        )

    def test_permalink_format_in_results(self):
        scraper, mock_reddit = self._make_scraper_with_mock()
        sub = _make_fake_submission(
            post_id="p1",
            permalink="/r/test/comments/p1/slug/",
        )
        mock_subreddit = MagicMock()
        mock_subreddit.search.return_value = [sub]
        mock_reddit.subreddit.return_value = mock_subreddit

        results = scraper.search_subreddit("query")
        assert results[0].permalink == "https://www.reddit.com/r/test/comments/p1/slug/"

    def test_deleted_author_in_search_result(self):
        scraper, mock_reddit = self._make_scraper_with_mock()
        sub = _make_fake_submission(post_id="p1", title="Anon post")
        # Author exists but will show as string
        mock_subreddit = MagicMock()
        mock_subreddit.search.return_value = [sub]
        mock_reddit.subreddit.return_value = mock_subreddit

        results = scraper.search_subreddit("query")
        assert len(results) == 1
        # Author present (not None), so post is kept


# ═══════════════════════════════════════════════════════════════
#  Exports from __init__.py
# ═══════════════════════════════════════════════════════════════


class TestScrapersExports:
    def test_reddit_scraper_exported(self):
        from reviewmind.scrapers import RedditScraper as RS

        assert RS is RedditScraper

    def test_reddit_post_exported(self):
        from reviewmind.scrapers import RedditPost as RP

        assert RP is RedditPost

    def test_reddit_comment_exported(self):
        from reviewmind.scrapers import RedditComment as RC

        assert RC is RedditComment

    def test_search_result_exported(self):
        from reviewmind.scrapers import SearchResult as SR

        assert SR is SearchResult

    def test_constants_exported(self):
        from reviewmind.scrapers import DEFAULT_COMMENT_LIMIT as DCL
        from reviewmind.scrapers import DEFAULT_SEARCH_LIMIT as DSL
        from reviewmind.scrapers import DEFAULT_SUBREDDIT as DS

        assert DCL == 20
        assert DSL == 10
        assert DS == "all"


# ═══════════════════════════════════════════════════════════════
#  Integration scenarios (all mocked)
# ═══════════════════════════════════════════════════════════════


class TestIntegrationScenarios:
    def test_full_parse_url_flow(self):
        """Complete flow: URL → extract_post_id → get_post → RedditPost with comments."""
        scraper = RedditScraper(
            client_id="id",
            client_secret="secret",
            user_agent="ua",
        )
        mock_reddit = MagicMock()
        scraper._reddit = mock_reddit

        reply = SimpleNamespace(
            body="I agree!",
            author=_fake_author("replier"),
            score=3,
            replies=[],
        )
        comment = SimpleNamespace(
            body="Great headphones!",
            author=_fake_author("commenter"),
            score=15,
            replies=[reply],
        )
        sub = _make_fake_submission(
            post_id="abc123",
            title="Sony WH-1000XM5 Review",
            selftext="These are the best headphones I've tried.",
            subreddit="headphones",
            comments=[comment],
        )
        mock_reddit.submission.return_value = sub

        url = "https://www.reddit.com/r/headphones/comments/abc123/sony_review/"
        result = scraper.parse_url(url)

        assert result is not None
        assert result.post_id == "abc123"
        assert result.title == "Sony WH-1000XM5 Review"
        assert len(result.comments) == 2  # top + reply
        assert "Sony WH-1000XM5 Review" in result.full_text
        assert "Great headphones!" in result.full_text
        assert "I agree!" in result.full_text

    def test_search_and_parse_flow(self):
        """Search → get results → parse first result."""
        scraper = RedditScraper(
            client_id="id",
            client_secret="secret",
            user_agent="ua",
        )
        mock_reddit = MagicMock()
        scraper._reddit = mock_reddit

        search_sub = _make_fake_submission(
            post_id="s1",
            title="Best headphones",
            permalink="/r/headphones/comments/s1/best/",
        )
        mock_subreddit = MagicMock()
        mock_subreddit.search.return_value = [search_sub]
        mock_reddit.subreddit.return_value = mock_subreddit

        results = scraper.search_subreddit("best headphones")
        assert len(results) == 1
        assert results[0].post_id == "s1"

    def test_mixed_deleted_and_valid_comments(self):
        """Ensure deleted comments are filtered while valid ones are kept."""
        scraper = RedditScraper(
            client_id="id",
            client_secret="secret",
            user_agent="ua",
        )
        mock_reddit = MagicMock()
        scraper._reddit = mock_reddit

        comments = [
            SimpleNamespace(body="Valid comment", author=_fake_author("u1"), score=10, replies=[]),
            SimpleNamespace(body="[deleted]", author=None, score=0, replies=[]),
            SimpleNamespace(body="[removed]", author=_fake_author("u3"), score=-5, replies=[]),
            SimpleNamespace(body="Another valid", author=_fake_author("u4"), score=5, replies=[]),
        ]
        sub = _make_fake_submission(post_id="mix", comments=comments)
        mock_reddit.submission.return_value = sub

        result = scraper.get_post("mix")
        assert result is not None
        assert len(result.comments) == 2
        assert result.comments[0].body == "Valid comment"
        assert result.comments[1].body == "Another valid"

    def test_post_with_only_title_no_selftext(self):
        """Link posts often have empty selftext."""
        scraper = RedditScraper(
            client_id="id",
            client_secret="secret",
            user_agent="ua",
        )
        mock_reddit = MagicMock()
        scraper._reddit = mock_reddit

        sub = _make_fake_submission(
            post_id="link",
            title="Check out this review",
            selftext="",
            url="https://example.com/review",
        )
        mock_reddit.submission.return_value = sub

        result = scraper.get_post("link")
        assert result is not None
        assert result.title == "Check out this review"
        assert result.selftext == ""
        assert "Check out this review" in result.full_text

    def test_large_comment_thread_limited(self):
        """With many comments, only comment_limit are collected."""
        scraper = RedditScraper(
            client_id="id",
            client_secret="secret",
            user_agent="ua",
            comment_limit=5,
        )
        mock_reddit = MagicMock()
        scraper._reddit = mock_reddit

        comments = [
            SimpleNamespace(
                body=f"Comment {i}",
                author=_fake_author(f"user{i}"),
                score=i,
                replies=[],
            )
            for i in range(50)
        ]
        sub = _make_fake_submission(post_id="big", comments=comments)
        mock_reddit.submission.return_value = sub

        result = scraper.get_post("big")
        assert result is not None
        assert len(result.comments) == 5
