"""reviewmind/scrapers/reddit.py — Reddit parser (PRAW).

Fetches posts and comments from Reddit using PRAW (Python Reddit API Wrapper).
Supports parsing direct Reddit URLs and searching subreddits by query.
Filters out deleted/removed posts and comments.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import praw
import structlog

logger = structlog.get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────

DEFAULT_COMMENT_LIMIT: int = 20
"""Maximum number of top-level + first-level comments to collect per post."""

DEFAULT_SEARCH_LIMIT: int = 10
"""Default number of posts returned by subreddit search."""

DEFAULT_SUBREDDIT: str = "all"
"""Default subreddit for search (r/all)."""

DEFAULT_REVIEW_SUBREDDITS: tuple[str, ...] = (
    "BuyItForLife",
    "gadgets",
    "headphones",
    "audiophile",
    "buildapc",
    "smarthome",
    "apple",
    "Android",
)
"""Subreddits commonly used for product reviews and discussions."""

_DELETED_MARKERS: frozenset[str] = frozenset({"[deleted]", "[removed]", "[удалено]"})
"""Markers indicating a deleted or removed post/comment body."""

# ── URL patterns ──────────────────────────────────────────────

_REDDIT_URL_PATTERNS: list[re.Pattern[str]] = [
    # Standard: https://www.reddit.com/r/subreddit/comments/post_id/slug/
    re.compile(r"(?:https?://)?(?:www\.)?reddit\.com/r/(?P<subreddit>[^/]+)/comments/(?P<post_id>[a-z0-9]+)"),
    # Old reddit: https://old.reddit.com/r/subreddit/comments/post_id/slug/
    re.compile(r"(?:https?://)?old\.reddit\.com/r/(?P<subreddit>[^/]+)/comments/(?P<post_id>[a-z0-9]+)"),
    # Short links: https://redd.it/post_id
    re.compile(r"(?:https?://)?redd\.it/(?P<post_id>[a-z0-9]+)"),
]

_REDDIT_DOMAIN_PATTERN: re.Pattern[str] = re.compile(
    r"(?:https?://)?(?:www\.|old\.)?reddit\.com|(?:https?://)?redd\.it",
    re.IGNORECASE,
)


# ── Result dataclasses ────────────────────────────────────────


@dataclass
class RedditComment:
    """A single Reddit comment."""

    author: str
    body: str
    score: int
    is_top_level: bool


@dataclass
class RedditPost:
    """Result of parsing a Reddit post."""

    post_id: str
    title: str
    selftext: str
    author: str
    score: int
    upvote_ratio: float
    num_comments: int
    subreddit: str
    url: str
    permalink: str
    created_utc: float
    comments: list[RedditComment] = field(default_factory=list)
    source_url: str = ""
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        """Return combined post text + comments as a single string for ingestion."""
        parts: list[str] = []

        # Post title and body
        if self.title:
            parts.append(self.title)
        if self.selftext and self.selftext not in _DELETED_MARKERS:
            parts.append(self.selftext)

        # Comments
        for comment in self.comments:
            if comment.body and comment.body not in _DELETED_MARKERS:
                parts.append(comment.body)

        return "\n\n".join(parts)


@dataclass
class SearchResult:
    """Summary of a Reddit post found through search."""

    post_id: str
    title: str
    selftext: str
    author: str
    score: int
    num_comments: int
    subreddit: str
    url: str
    permalink: str
    created_utc: float


# ── Scraper class ─────────────────────────────────────────────


class RedditScraper:
    """Fetches posts and comments from Reddit via PRAW.

    Usage::

        scraper = RedditScraper(
            client_id="...",
            client_secret="...",
            user_agent="ReviewMind/1.0",
        )
        post = scraper.parse_url("https://reddit.com/r/headphones/comments/abc123/")
        if post:
            print(post.title, len(post.comments))

    For searching::

        results = scraper.search_subreddit("best headphones 2026", limit=10)
        for r in results:
            print(r.title, r.score)
    """

    def __init__(
        self,
        *,
        client_id: str | None = None,
        client_secret: str | None = None,
        user_agent: str | None = None,
        comment_limit: int = DEFAULT_COMMENT_LIMIT,
    ) -> None:
        """Create a RedditScraper instance.

        Args:
            client_id: Reddit API client ID. Falls back to config if ``None``.
            client_secret: Reddit API client secret. Falls back to config if ``None``.
            user_agent: User agent string. Falls back to config if ``None``.
            comment_limit: Max comments to collect per post (top-level + first-level).
        """
        if client_id is None or client_secret is None or user_agent is None:
            from reviewmind.config import settings

            client_id = client_id or settings.reddit_client_id
            client_secret = client_secret or settings.reddit_client_secret
            user_agent = user_agent or settings.reddit_user_agent

        self._client_id = client_id
        self._client_secret = client_secret
        self._user_agent = user_agent
        self._comment_limit = comment_limit

        self._reddit: praw.Reddit | None = None

    @property
    def reddit(self) -> praw.Reddit:
        """Lazy-create and return the PRAW Reddit instance."""
        if self._reddit is None:
            self._reddit = praw.Reddit(
                client_id=self._client_id,
                client_secret=self._client_secret,
                user_agent=self._user_agent,
            )
        return self._reddit

    # ── Public API ────────────────────────────────────────────

    @staticmethod
    def extract_post_id(url: str) -> str:
        """Extract the post ID from a Reddit URL.

        Supports ``reddit.com/r/.../comments/ID/...``, ``old.reddit.com/...``,
        and ``redd.it/ID`` formats.

        Args:
            url: A Reddit post URL.

        Returns:
            The post ID string.

        Raises:
            ValueError: If the URL does not match any known Reddit format.
        """
        if not url or not isinstance(url, str):
            raise ValueError(f"Invalid Reddit URL: {url!r}")

        url = url.strip()
        for pattern in _REDDIT_URL_PATTERNS:
            match = pattern.search(url)
            if match:
                return match.group("post_id")

        raise ValueError(f"Could not extract post ID from URL: {url!r}")

    @staticmethod
    def is_reddit_url(url: str) -> bool:
        """Return ``True`` if *url* looks like a Reddit URL."""
        if not url or not isinstance(url, str):
            return False
        return bool(_REDDIT_DOMAIN_PATTERN.search(url.strip()))

    def parse_url(self, url: str) -> RedditPost | None:
        """Fetch a Reddit post and its top comments by URL.

        Args:
            url: A Reddit post URL.

        Returns:
            A :class:`RedditPost` on success, or ``None`` if the post is
            deleted, removed, or inaccessible.
        """
        try:
            post_id = self.extract_post_id(url)
        except ValueError:
            logger.warning("reddit.invalid_url", url=url)
            return None

        return self.get_post(post_id, source_url=url.strip())

    def get_post(self, post_id: str, *, source_url: str = "") -> RedditPost | None:
        """Fetch a Reddit post by its ID.

        Args:
            post_id: The Reddit post ID (e.g. ``"abc123"``).
            source_url: Original URL for metadata tracking.

        Returns:
            A :class:`RedditPost` on success, or ``None`` if deleted/removed
            or an error occurs.
        """
        if not post_id or not isinstance(post_id, str):
            logger.warning("reddit.invalid_post_id", post_id=post_id)
            return None

        post_id = post_id.strip()

        try:
            submission = self.reddit.submission(id=post_id)
            # Force-load the submission attributes
            _ = submission.title
        except Exception:
            logger.exception("reddit.fetch_error", post_id=post_id)
            return None

        # Check for deleted/removed posts
        if self._is_deleted_submission(submission):
            logger.info("reddit.post_deleted", post_id=post_id)
            return None

        comments = self._collect_comments(submission)

        permalink = f"https://www.reddit.com{submission.permalink}"

        return RedditPost(
            post_id=post_id,
            title=submission.title or "",
            selftext=submission.selftext or "",
            author=str(submission.author) if submission.author else "[deleted]",
            score=submission.score,
            upvote_ratio=submission.upvote_ratio,
            num_comments=submission.num_comments,
            subreddit=str(submission.subreddit),
            url=submission.url or "",
            permalink=permalink,
            created_utc=submission.created_utc,
            comments=comments,
            source_url=source_url or permalink,
        )

    def search_subreddit(
        self,
        query: str,
        *,
        subreddit: str = DEFAULT_SUBREDDIT,
        limit: int = DEFAULT_SEARCH_LIMIT,
        sort: str = "relevance",
        time_filter: str = "year",
    ) -> list[SearchResult]:
        """Search a subreddit for posts matching *query*.

        Args:
            query: Search query string.
            subreddit: Subreddit name (default ``"all"``).
            limit: Maximum number of results (default 10).
            sort: Sort order: ``"relevance"``, ``"hot"``, ``"top"``, ``"new"``, ``"comments"``.
            time_filter: Time filter: ``"all"``, ``"year"``, ``"month"``, ``"week"``, ``"day"``, ``"hour"``.

        Returns:
            List of :class:`SearchResult` objects. May be shorter than *limit*
            if fewer matching posts exist or deleted posts are filtered out.
        """
        if not query or not isinstance(query, str):
            logger.warning("reddit.empty_search_query")
            return []

        query = query.strip()
        if not query:
            return []

        try:
            sub = self.reddit.subreddit(subreddit)
            results: list[SearchResult] = []

            for submission in sub.search(query, sort=sort, time_filter=time_filter, limit=limit):
                # Skip deleted/removed submissions
                if self._is_deleted_submission(submission):
                    continue

                permalink = f"https://www.reddit.com{submission.permalink}"
                results.append(
                    SearchResult(
                        post_id=submission.id,
                        title=submission.title or "",
                        selftext=submission.selftext or "",
                        author=str(submission.author) if submission.author else "[deleted]",
                        score=submission.score,
                        num_comments=submission.num_comments,
                        subreddit=str(submission.subreddit),
                        url=submission.url or "",
                        permalink=permalink,
                        created_utc=submission.created_utc,
                    )
                )

            logger.info(
                "reddit.search_completed",
                query=query,
                subreddit=subreddit,
                results_count=len(results),
            )
            return results

        except Exception:
            logger.exception(
                "reddit.search_error",
                query=query,
                subreddit=subreddit,
            )
            return []

    def search_posts(
        self,
        query: str,
        *,
        subreddits: tuple[str, ...] | list[str] = DEFAULT_REVIEW_SUBREDDITS,
        limit: int = DEFAULT_SEARCH_LIMIT,
        sort: str = "relevance",
        time_filter: str = "year",
    ) -> list[SearchResult]:
        """Search multiple subreddits and return deduplicated, merged results.

        Searches each subreddit individually and merges results, deduplicating
        by ``post_id`` and sorting by score descending. Returns at most *limit*
        unique results.

        Args:
            query: Search query string.
            subreddits: Subreddits to search (default :data:`DEFAULT_REVIEW_SUBREDDITS`).
            limit: Maximum total results to return (default 10).
            sort: Sort order per subreddit search.
            time_filter: Time filter: ``"year"``, ``"month"``, ``"week"``, etc.

        Returns:
            Deduplicated list of :class:`SearchResult` sorted by score descending.
        """
        if not query or not isinstance(query, str):
            logger.warning("reddit.empty_search_posts_query")
            return []

        query = query.strip()
        if not query:
            return []

        seen_ids: set[str] = set()
        merged: list[SearchResult] = []

        for sub_name in subreddits:
            per_sub = self.search_subreddit(
                query,
                subreddit=sub_name,
                limit=max(3, limit // max(len(subreddits), 1)),
                sort=sort,
                time_filter=time_filter,
            )
            for result in per_sub:
                if result.post_id not in seen_ids:
                    seen_ids.add(result.post_id)
                    merged.append(result)

        # Sort by score descending, then trim
        merged.sort(key=lambda r: r.score, reverse=True)
        final = merged[:limit]

        logger.info(
            "reddit.search_posts_completed",
            query=query,
            subreddits_searched=len(subreddits),
            results_count=len(final),
        )
        return final

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _is_deleted_submission(submission: Any) -> bool:
        """Check if a submission is deleted or removed."""
        # Author is None for deleted posts
        if submission.author is None:
            return True

        # Check selftext for deletion markers
        selftext = getattr(submission, "selftext", "") or ""
        if selftext.strip() in _DELETED_MARKERS:
            # Title-only posts with [deleted] body are OK, but body being
            # [removed] means mod removal — still return for title content.
            # We only skip if BOTH title is empty and selftext is deleted.
            title = getattr(submission, "title", "") or ""
            if not title.strip():
                return True

        return False

    def _collect_comments(self, submission: Any) -> list[RedditComment]:
        """Collect top-level and first-level reply comments.

        Replaces ``MoreComments`` objects inline, collects up to
        ``self._comment_limit`` comments total.

        Args:
            submission: A PRAW Submission object.

        Returns:
            List of :class:`RedditComment` objects.
        """
        comments: list[RedditComment] = []
        collected = 0

        try:
            # Replace MoreComments to load actual comments (limit=0 means
            # replace all MoreComments objects).
            submission.comments.replace_more(limit=0)
        except Exception:
            logger.warning(
                "reddit.replace_more_failed",
                post_id=submission.id,
            )

        for top_comment in submission.comments:
            if collected >= self._comment_limit:
                break

            # Skip deleted/removed comments
            body = getattr(top_comment, "body", "") or ""
            if not body.strip() or body.strip() in _DELETED_MARKERS:
                continue

            author = str(top_comment.author) if top_comment.author else "[deleted]"
            comments.append(
                RedditComment(
                    author=author,
                    body=body.strip(),
                    score=getattr(top_comment, "score", 0),
                    is_top_level=True,
                )
            )
            collected += 1

            # First-level replies
            replies = getattr(top_comment, "replies", None)
            if replies:
                for reply in replies:
                    if collected >= self._comment_limit:
                        break

                    reply_body = getattr(reply, "body", "") or ""
                    if not reply_body.strip() or reply_body.strip() in _DELETED_MARKERS:
                        continue

                    reply_author = str(reply.author) if reply.author else "[deleted]"
                    comments.append(
                        RedditComment(
                            author=reply_author,
                            body=reply_body.strip(),
                            score=getattr(reply, "score", 0),
                            is_top_level=False,
                        )
                    )
                    collected += 1

        logger.debug(
            "reddit.comments_collected",
            post_id=submission.id,
            count=len(comments),
            limit=self._comment_limit,
        )
        return comments
