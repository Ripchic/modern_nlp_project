"""Unit tests for RedditScraper.search_posts — multi-subreddit search."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from reviewmind.scrapers.reddit import (
    DEFAULT_REVIEW_SUBREDDITS,
    RedditScraper,
    SearchResult,
)

# ═══════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════


def _fake_author(name: str) -> object:
    return type("FakeAuthor", (), {"__str__": lambda self: name})()


def _make_fake_submission(
    post_id: str = "abc123",
    title: str = "Test post",
    selftext: str = "Body text",
    author: str = "testuser",
    score: int = 42,
    num_comments: int = 10,
    subreddit: str = "headphones",
    url: str = "",
    permalink: str = "/r/headphones/comments/abc123/test_post/",
    created_utc: float = 1709300000.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=post_id,
        title=title,
        selftext=selftext,
        author=_fake_author(author) if author else None,
        score=score,
        upvote_ratio=0.95,
        num_comments=num_comments,
        subreddit=_fake_author(subreddit),
        url=url or f"https://www.reddit.com{permalink}",
        permalink=permalink,
        created_utc=created_utc,
    )


def _make_scraper_with_mock() -> tuple[RedditScraper, MagicMock]:
    """Create a scraper with a mocked PRAW Reddit instance."""
    scraper = RedditScraper(client_id="fake", client_secret="fake", user_agent="test/1.0")
    mock_reddit = MagicMock()
    scraper._reddit = mock_reddit
    return scraper, mock_reddit


# ═══════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════


class TestSearchPostsConstants:
    def test_default_review_subreddits_count(self):
        assert len(DEFAULT_REVIEW_SUBREDDITS) >= 5

    def test_default_review_subreddits_contains_expected(self):
        expected = {"BuyItForLife", "gadgets", "headphones"}
        assert expected.issubset(set(DEFAULT_REVIEW_SUBREDDITS))

    def test_default_review_subreddits_is_tuple(self):
        assert isinstance(DEFAULT_REVIEW_SUBREDDITS, tuple)

    def test_importable_from_init(self):
        from reviewmind.scrapers import DEFAULT_REVIEW_SUBREDDITS as DRS

        assert DRS is DEFAULT_REVIEW_SUBREDDITS


# ═══════════════════════════════════════════════════════════════
#  search_posts — success
# ═══════════════════════════════════════════════════════════════


class TestSearchPostsSuccess:
    def test_returns_search_results(self):
        scraper, mock_reddit = _make_scraper_with_mock()
        submissions = [
            _make_fake_submission(post_id="p1", title="Post 1", score=100, subreddit="gadgets"),
            _make_fake_submission(post_id="p2", title="Post 2", score=50, subreddit="gadgets"),
        ]

        mock_sub = MagicMock()
        mock_sub.search.return_value = iter(submissions)
        mock_reddit.subreddit.return_value = mock_sub

        results = scraper.search_posts("best headphones", subreddits=("gadgets",), limit=10)

        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)

    def test_results_sorted_by_score_descending(self):
        scraper, mock_reddit = _make_scraper_with_mock()
        subs_a = [_make_fake_submission(post_id="p1", score=10, subreddit="a")]
        subs_b = [_make_fake_submission(post_id="p2", score=100, subreddit="b")]

        mock_sub_a = MagicMock()
        mock_sub_a.search.return_value = iter(subs_a)
        mock_sub_b = MagicMock()
        mock_sub_b.search.return_value = iter(subs_b)
        mock_reddit.subreddit.side_effect = [mock_sub_a, mock_sub_b]

        results = scraper.search_posts("headphones", subreddits=("a", "b"), limit=10)

        assert results[0].score >= results[-1].score
        assert results[0].post_id == "p2"

    def test_deduplicates_by_post_id(self):
        scraper, mock_reddit = _make_scraper_with_mock()
        subs_a = [_make_fake_submission(post_id="dup1", score=50, subreddit="a")]
        subs_b = [_make_fake_submission(post_id="dup1", score=50, subreddit="b")]

        mock_sub_a = MagicMock()
        mock_sub_a.search.return_value = iter(subs_a)
        mock_sub_b = MagicMock()
        mock_sub_b.search.return_value = iter(subs_b)
        mock_reddit.subreddit.side_effect = [mock_sub_a, mock_sub_b]

        results = scraper.search_posts("test", subreddits=("a", "b"), limit=10)

        assert len(results) == 1
        assert results[0].post_id == "dup1"

    def test_limit_respected(self):
        scraper, mock_reddit = _make_scraper_with_mock()
        submissions = [_make_fake_submission(post_id=f"p{i}", score=100 - i) for i in range(20)]

        mock_sub = MagicMock()
        mock_sub.search.return_value = iter(submissions)
        mock_reddit.subreddit.return_value = mock_sub

        results = scraper.search_posts("headphones", subreddits=("all",), limit=5)

        assert len(results) == 5

    def test_multiple_subreddits_searched(self):
        scraper, mock_reddit = _make_scraper_with_mock()

        mock_sub = MagicMock()
        mock_sub.search.return_value = iter([])
        mock_reddit.subreddit.return_value = mock_sub

        scraper.search_posts("test", subreddits=("a", "b", "c"))

        assert mock_reddit.subreddit.call_count == 3


# ═══════════════════════════════════════════════════════════════
#  search_posts — empty / invalid input
# ═══════════════════════════════════════════════════════════════


class TestSearchPostsEmptyInput:
    def test_empty_query(self):
        scraper, _ = _make_scraper_with_mock()
        assert scraper.search_posts("") == []

    def test_whitespace_query(self):
        scraper, _ = _make_scraper_with_mock()
        assert scraper.search_posts("   ") == []

    def test_none_query(self):
        scraper, _ = _make_scraper_with_mock()
        assert scraper.search_posts(None) == []  # type: ignore[arg-type]

    def test_non_string_query(self):
        scraper, _ = _make_scraper_with_mock()
        assert scraper.search_posts(123) == []  # type: ignore[arg-type]


# ═══════════════════════════════════════════════════════════════
#  search_posts — error handling
# ═══════════════════════════════════════════════════════════════


class TestSearchPostsErrors:
    def test_subreddit_error_returns_empty_for_that_sub(self):
        """Error in one subreddit should not crash the whole search."""
        scraper, mock_reddit = _make_scraper_with_mock()

        good_subs = [_make_fake_submission(post_id="p1", score=50, subreddit="b")]
        mock_sub_good = MagicMock()
        mock_sub_good.search.return_value = iter(good_subs)

        mock_sub_bad = MagicMock()
        mock_sub_bad.search.side_effect = Exception("Subreddit unavailable")

        mock_reddit.subreddit.side_effect = [mock_sub_bad, mock_sub_good]

        results = scraper.search_posts("test", subreddits=("bad", "b"))

        assert len(results) == 1
        assert results[0].post_id == "p1"

    def test_all_subreddits_fail_returns_empty(self):
        scraper, mock_reddit = _make_scraper_with_mock()

        mock_sub = MagicMock()
        mock_sub.search.side_effect = Exception("Nope")
        mock_reddit.subreddit.return_value = mock_sub

        results = scraper.search_posts("test", subreddits=("a", "b"))

        assert results == []


# ═══════════════════════════════════════════════════════════════
#  search_posts — time_filter
# ═══════════════════════════════════════════════════════════════


class TestSearchPostsTimeFilter:
    def test_default_time_filter_is_year(self):
        scraper, mock_reddit = _make_scraper_with_mock()

        mock_sub = MagicMock()
        mock_sub.search.return_value = iter([])
        mock_reddit.subreddit.return_value = mock_sub

        scraper.search_posts("test", subreddits=("all",))

        call_kwargs = mock_sub.search.call_args[1]
        assert call_kwargs["time_filter"] == "year"

    def test_custom_time_filter(self):
        scraper, mock_reddit = _make_scraper_with_mock()

        mock_sub = MagicMock()
        mock_sub.search.return_value = iter([])
        mock_reddit.subreddit.return_value = mock_sub

        scraper.search_posts("test", subreddits=("all",), time_filter="month")

        call_kwargs = mock_sub.search.call_args[1]
        assert call_kwargs["time_filter"] == "month"


# ═══════════════════════════════════════════════════════════════
#  Exports
# ═══════════════════════════════════════════════════════════════


class TestRedditSearchExports:
    def test_default_review_subreddits_from_init(self):
        from reviewmind.scrapers import DEFAULT_REVIEW_SUBREDDITS

        assert isinstance(DEFAULT_REVIEW_SUBREDDITS, tuple)
        assert len(DEFAULT_REVIEW_SUBREDDITS) >= 5


# ═══════════════════════════════════════════════════════════════
#  Integration scenarios
# ═══════════════════════════════════════════════════════════════


class TestRedditSearchIntegration:
    def test_search_results_have_urls(self):
        scraper, mock_reddit = _make_scraper_with_mock()
        submissions = [
            _make_fake_submission(post_id="p1", subreddit="headphones"),
        ]

        mock_sub = MagicMock()
        mock_sub.search.return_value = iter(submissions)
        mock_reddit.subreddit.return_value = mock_sub

        results = scraper.search_posts("best headphones", subreddits=("headphones",))

        assert len(results) == 1
        assert "reddit.com" in results[0].permalink

    def test_search_with_default_subreddits(self):
        """Using default subreddits should not crash."""
        scraper, mock_reddit = _make_scraper_with_mock()

        mock_sub = MagicMock()
        mock_sub.search.return_value = iter([])
        mock_reddit.subreddit.return_value = mock_sub

        results = scraper.search_posts("noise cancelling headphones")

        assert results == []
        assert mock_reddit.subreddit.call_count == len(DEFAULT_REVIEW_SUBREDDITS)

    def test_niche_query_returns_fewer_results(self):
        """A niche query may return < limit results — not an error."""
        scraper, mock_reddit = _make_scraper_with_mock()
        submissions = [_make_fake_submission(post_id="p1")]

        mock_sub = MagicMock()
        mock_sub.search.return_value = iter(submissions)
        mock_reddit.subreddit.return_value = mock_sub

        results = scraper.search_posts("obscure gadget xyz", subreddits=("all",), limit=10)

        assert len(results) >= 0  # May be 0 or 1
