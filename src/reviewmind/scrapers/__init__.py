"""reviewmind.scrapers — Data source scrapers (YouTube, Reddit, Web, Tavily)."""

from reviewmind.scrapers.reddit import (
    DEFAULT_COMMENT_LIMIT,
    DEFAULT_REVIEW_SUBREDDITS,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_SUBREDDIT,
    RedditComment,
    RedditPost,
    RedditScraper,
    SearchResult,
)
from reviewmind.scrapers.tavily import (
    DEFAULT_MAX_RESULTS as TAVILY_DEFAULT_MAX_RESULTS,
)
from reviewmind.scrapers.tavily import (
    DEFAULT_SEARCH_DEPTH as TAVILY_DEFAULT_SEARCH_DEPTH,
)
from reviewmind.scrapers.tavily import (
    DEFAULT_TIMEOUT as TAVILY_DEFAULT_TIMEOUT,
)
from reviewmind.scrapers.tavily import (
    MIN_CONTENT_LENGTH as TAVILY_MIN_CONTENT_LENGTH,
)
from reviewmind.scrapers.tavily import (
    TavilyResult,
    TavilyScraper,
)
from reviewmind.scrapers.web import (
    DEFAULT_TIMEOUT,
    MIN_TEXT_LENGTH,
    WebPage,
    WebScraper,
)
from reviewmind.scrapers.youtube import (
    DEFAULT_LANGUAGES,
    DEFAULT_SEARCH_MAX_RESULTS,
    MAX_AGE_DAYS,
    MIN_WORD_COUNT,
    YOUTUBE_API_BASE_URL,
    TranscriptResult,
    VideoInfo,
    YouTubeScraper,
)

__all__ = [
    "DEFAULT_COMMENT_LIMIT",
    "DEFAULT_LANGUAGES",
    "DEFAULT_REVIEW_SUBREDDITS",
    "DEFAULT_SEARCH_LIMIT",
    "DEFAULT_SEARCH_MAX_RESULTS",
    "DEFAULT_SUBREDDIT",
    "DEFAULT_TIMEOUT",
    "MAX_AGE_DAYS",
    "MIN_TEXT_LENGTH",
    "MIN_WORD_COUNT",
    "RedditComment",
    "RedditPost",
    "RedditScraper",
    "SearchResult",
    "TAVILY_DEFAULT_MAX_RESULTS",
    "TAVILY_DEFAULT_SEARCH_DEPTH",
    "TAVILY_DEFAULT_TIMEOUT",
    "TAVILY_MIN_CONTENT_LENGTH",
    "TavilyResult",
    "TavilyScraper",
    "TranscriptResult",
    "VideoInfo",
    "WebPage",
    "WebScraper",
    "YOUTUBE_API_BASE_URL",
    "YouTubeScraper",
]
