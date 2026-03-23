"""reviewmind/scrapers/fourpda.py — 4PDA forum scraper.

Fetches posts and topics from the 4PDA Russian tech forum (4pda.to),
which runs on Invision Power Board (IP.Board / IPS Community).

Supports:
  - Parsing a specific forum topic by URL (sync + async)
  - Searching the forum for topics matching a query (async)

Note on Cloudflare:
  4PDA uses Cloudflare CDN.  The scraper sends browser-like headers to
  mitigate basic bot detection.  If Cloudflare actively challenges the
  request (HTTP 403 / CAPTCHA page), the scraper will log a warning and
  return an empty result rather than crashing the caller.
"""

from __future__ import annotations

import asyncio
import re
import unicodedata
from dataclasses import dataclass, field
from urllib.parse import quote_plus, urljoin, urlparse

import httpx
import structlog
from bs4 import BeautifulSoup, Tag

from reviewmind.scrapers.web import WebScraper

logger = structlog.get_logger(__name__)

# ── Invisible chars cleanup (subset of cleaner.py logic) ─────

_INVISIBLE_CHARS_RE = re.compile(
    r"[\u200b\u200c\u200d\u200e\u200f\ufeff\u00ad\u2060\u2061\u2062\u2063\u2064]"
)


def _clean_post_text(text: str) -> str:
    """Lightweight text cleanup for a single forum post body.

    Applies a subset of the full ``cleaner.clean_text`` pipeline that is
    relevant to HTML-extracted forum text:
      - Unicode NFKC normalization
      - Invisible character removal
      - Collapse excessive whitespace / blank lines
      - Strip leading/trailing whitespace
    """
    text = unicodedata.normalize("NFKC", text)
    text = _INVISIBLE_CHARS_RE.sub("", text)
    text = re.sub(r"[^\S\n]+", " ", text)      # h-space → single space
    text = re.sub(r"\n{3,}", "\n\n", text)      # 3+ newlines → 2
    return text.strip()

# ── Constants ─────────────────────────────────────────────────

FOURPDA_BASE_URL = "https://4pda.to/forum/"
FOURPDA_SEARCH_URL = "https://4pda.to/forum/index.php?act=search"

DEFAULT_MAX_RESULTS: int = 5
DEFAULT_TIMEOUT: float = 15.0  # seconds
MIN_POST_LENGTH: int = 60  # skip noise / one-liner posts
DEFAULT_MAX_POSTS_PER_TOPIC: int = 30
DEFAULT_MAX_FOLLOWED_LINKS: int = 5  # max review links to follow from pinned post

# ── URL patterns ──────────────────────────────────────────────

_TOPIC_URL_RE = re.compile(
    r"4pda\.to/forum/(?:index\.php\?showtopic=\d+|t/\d+)",
    re.IGNORECASE,
)

_FORUM_URL_RE = re.compile(
    r"4pda\.to/forum",
    re.IGNORECASE,
)

# ── Shared browser-like headers ───────────────────────────────

_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://4pda.to/forum/",
    "Connection": "keep-alive",
    "DNT": "1",
}

# ── Review-link classification ────────────────────────────────

_REVIEW_KEYWORDS_RE = re.compile(
    r"обзор|review|отзыв|тест|benchmark|сравнен|распаковк|unboxing|hands.?on",
    re.IGNORECASE,
)

_SPECS_KEYWORDS_RE = re.compile(
    r"характеристик|specification|specs|параметр|техданн",
    re.IGNORECASE,
)

# Topic title preference for ranking search results.
# Higher weight = more relevant for device reviews.
_TOPIC_PREFER_RE = re.compile(
    r"обсуждение|обзор|review|отзыв",
    re.IGNORECASE,
)
_TOPIC_DEMOTE_RE = re.compile(
    r"покупка|продажа|куплю|продам|аксессуар|чехл|стекл|пленк",
    re.IGNORECASE,
)

# ── Result dataclasses ────────────────────────────────────────


@dataclass
class ExtractedLink:
    """A link extracted from the pinned post."""

    url: str
    text: str
    category: str = "other"  # review | specs | firmware | discussion | other


@dataclass
class PinnedPostData:
    """Structured data extracted from the pinned (first) post of a device topic."""

    post_id: str = ""
    specs_text: str = ""
    links: list[ExtractedLink] = field(default_factory=list)
    review_texts: list[str] = field(default_factory=list)

    @property
    def review_links(self) -> list[ExtractedLink]:
        return [ln for ln in self.links if ln.category == "review"]


@dataclass
class ForumPost4PDA:
    """A single post from a 4PDA forum topic."""

    author: str
    body: str
    date: str = ""
    post_id: str = ""


@dataclass
class ForumTopic4PDA:
    """Result of parsing a 4PDA forum topic page."""

    topic_id: str
    title: str
    url: str
    posts: list[ForumPost4PDA] = field(default_factory=list)
    source_url: str = ""
    pinned: PinnedPostData | None = None

    @property
    def full_text(self) -> str:
        """Concatenated text of the topic title + all post bodies
        plus pinned-post specs and fetched review content."""
        parts: list[str] = [self.title]

        if self.pinned:
            if self.pinned.specs_text:
                parts.append(f"[Характеристики]\n{self.pinned.specs_text}")
            for review in self.pinned.review_texts:
                parts.append(f"[Обзор]\n{review}")

        for post in self.posts:
            prefix = f"{post.author}: " if post.author else ""
            parts.append(f"{prefix}{post.body}")
        return "\n\n".join(p for p in parts if p)


@dataclass
class ForumSearchResult4PDA:
    """A single result item from 4PDA forum search."""

    title: str
    url: str
    snippet: str = ""
    topic_id: str = ""


# ── Scraper ───────────────────────────────────────────────────


class Forum4PDAScraper:
    """Scraper for the 4PDA Russian IT forum (4pda.to/forum/).

    HTML is fetched with ``httpx`` and parsed with ``BeautifulSoup`` (lxml
    backend).  Both synchronous (for the ingestion pipeline) and asynchronous
    (for the bot query handler) interfaces are provided.

    Parameters
    ----------
    timeout:
        HTTP request timeout in seconds (default 15 s).
    max_posts_per_topic:
        Maximum number of posts to parse from a single topic page (default 30).
    """

    def __init__(
        self,
        timeout: float = DEFAULT_TIMEOUT,
        max_posts_per_topic: int = DEFAULT_MAX_POSTS_PER_TOPIC,
        max_followed_links: int = DEFAULT_MAX_FOLLOWED_LINKS,
        follow_pinned_links: bool = True,
    ) -> None:
        self._timeout = timeout
        self._max_posts = max_posts_per_topic
        self._max_followed_links = max_followed_links
        self._follow_pinned_links = follow_pinned_links
        # Reusable clients — created lazily, closed explicitly if needed.
        self._sync_client: httpx.Client | None = None
        self._async_client: httpx.AsyncClient | None = None
        self._web_scraper = WebScraper(min_text_length=100)

    # ── Utility ───────────────────────────────────────────────

    @staticmethod
    def is_4pda_url(url: str) -> bool:
        """Return *True* if *url* points to the 4PDA forum."""
        return bool(_FORUM_URL_RE.search(url))

    @staticmethod
    def _validate_url(url: str) -> str | None:
        """Return normalised URL if valid HTTP(S), else ``None``."""
        if not url or not isinstance(url, str):
            return None
        url = url.strip()
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return None
        return url

    @staticmethod
    def _extract_topic_id(url: str) -> str:
        """Extract the numeric topic ID from a 4PDA URL, or return ``''``."""
        m = re.search(r"showtopic=(\d+)", url)
        if m:
            return m.group(1)
        m = re.search(r"/t/(\d+)", url)
        if m:
            return m.group(1)
        return ""

    # ── Pinned-post helpers ────────────────────────────────────

    @staticmethod
    def _classify_link(url: str, text: str) -> str:
        """Classify a link found in the pinned post."""
        combined = f"{text} {url}".lower()
        if _REVIEW_KEYWORDS_RE.search(combined):
            return "review"
        if _SPECS_KEYWORDS_RE.search(combined):
            return "specs"
        if re.search(r"прошив|firmware|recovery|root", combined, re.IGNORECASE):
            return "firmware"
        if re.search(r"showtopic=|/t/\d+", url):
            return "discussion"
        return "other"

    @staticmethod
    def _extract_specs_from_post(body_tag: Tag) -> str:
        """Extract device specifications from the pinned post body.

        Looks for:
          - <table> elements (spec tables)
          - <ul>/<ol> lists with spec-like items
          - Lines matching key-value patterns ("Процессор: ...", "Экран: ...", etc.)
        """
        specs_parts: list[str] = []

        # 1) Tables (often used for specs)
        for table in body_tag.find_all("table", limit=5):
            rows: list[str] = []
            for tr in table.find_all("tr", limit=50):
                cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                if cells and any(cells):
                    rows.append(" | ".join(cells))
            if rows:
                specs_parts.append("\n".join(rows))

        # 2) Lists inside the first post
        for lst in body_tag.find_all(["ul", "ol"], limit=5):
            items: list[str] = []
            for li in lst.find_all("li", limit=30):
                li_text = li.get_text(strip=True)
                if li_text and len(li_text) > 5:
                    items.append(f"• {li_text}")
            if items:
                specs_parts.append("\n".join(items))

        # 3) Key-value lines from text (Экран: IPS 6.7", etc.)
        body_text = body_tag.get_text(separator="\n", strip=True)
        kv_re = re.compile(
            r"^\s*(Процессор|Экран|Дисплей|Память|ОЗУ|RAM|Батарея|Аккумулятор|"
            r"Камера|Разрешение|Частота|Вес|Масса|Размер|Габарит|"
            r"Интерфейс|Bluetooth|Wi-?Fi|NFC|GPS|SIM|Зарядка|ОС|Android|iOS|"
            r"Chipset|CPU|GPU|Display|Battery|Camera|Storage|Weight|Dimensions)"
            r"\s*[:—–-]\s*(.+)$",
            re.IGNORECASE | re.MULTILINE,
        )
        kv_lines = kv_re.findall(body_text)
        if kv_lines:
            specs_parts.append(
                "\n".join(f"{k.strip()}: {v.strip()}" for k, v in kv_lines)
            )

        if not specs_parts:
            return ""

        merged = "\n\n".join(specs_parts)
        return _clean_post_text(merged)

    @staticmethod
    def _extract_links_from_post(body_tag: Tag, base_url: str) -> list[ExtractedLink]:
        """Extract and classify all meaningful links from the pinned post."""
        links: list[ExtractedLink] = []
        seen_urls: set[str] = set()

        for a in body_tag.find_all("a", href=True):
            href = a["href"].strip()
            if not href or href.startswith("#") or href.startswith("javascript:"):
                continue

            # Resolve relative URLs
            if not href.startswith("http"):
                href = urljoin(base_url, href)

            # Deduplicate
            if href in seen_urls:
                continue
            seen_urls.add(href)

            text = a.get_text(strip=True)
            category = Forum4PDAScraper._classify_link(href, text)
            links.append(ExtractedLink(url=href, text=text, category=category))

        return links

    def _find_pinned_post_tag(self, soup: BeautifulSoup) -> Tag | None:
        """Find the pinned / first post body element in a 4PDA topic page.

        On 4PDA device topics, the first post is the "шапка" (header) which
        serves as the pinned post. It contains specs, important links, etc.
        """
        # Try finding the first post container
        first_post = soup.find("div", id=re.compile(r"^post[-_]\d+$"))
        if not first_post:
            first_post = soup.find("article", attrs={"data-commentid": True})
        if not first_post:
            first_post = soup.find(
                "div", class_=re.compile(r"\bpost\b", re.I)
            )
        return first_post  # type: ignore[return-value]

    def _parse_pinned_post(self, soup: BeautifulSoup, base_url: str) -> PinnedPostData | None:
        """Extract structured data from the pinned (first) post."""
        post_tag = self._find_pinned_post_tag(soup)
        if post_tag is None:
            return None

        body_tag = post_tag.find(
            attrs={
                "class": re.compile(
                    r"post_body|msg_body|ipsComment_content|ipsType_richText|postcolor",
                    re.I,
                )
            }
        )
        if not body_tag:
            body_tag = post_tag

        post_id = ""
        div_id = post_tag.get("id", "") if hasattr(post_tag, "get") else ""
        m = re.search(r"\d+", str(div_id))
        if m:
            post_id = m.group()

        specs_text = self._extract_specs_from_post(body_tag)
        links = self._extract_links_from_post(body_tag, base_url)

        return PinnedPostData(
            post_id=post_id,
            specs_text=specs_text,
            links=links,
        )

    def _fetch_review_text_sync(self, url: str) -> str | None:
        """Synchronously fetch review content from a link."""
        if Forum4PDAScraper.is_4pda_url(url):
            html = self._get_sync(url)
            if not html:
                return None
            soup = BeautifulSoup(html, "lxml")
            # If it's a specific post link, extract just that post
            post_id_match = re.search(r"[&?]p=(\d+)", url)
            if post_id_match:
                pid = post_id_match.group(1)
                post_div = soup.find("div", id=f"post-{pid}") or soup.find(
                    "div", id=f"post_{pid}"
                )
                if post_div:
                    body_tag = post_div.find(
                        attrs={"class": re.compile(r"post_body|msg_body|postcolor", re.I)}
                    )
                    if body_tag:
                        return _clean_post_text(
                            body_tag.get_text(separator="\n", strip=True)
                        )
            # Fallback: parse whole topic page
            topic = self._parse_topic_html(html, url)
            return topic.full_text if topic.posts else None

        # External URL — use web scraper
        page = self._web_scraper.parse_url(url)
        return page.text if page else None

    async def _fetch_review_text_async(self, url: str) -> str | None:
        """Asynchronously fetch review content from a link."""
        if Forum4PDAScraper.is_4pda_url(url):
            html = await self._get_async(url)
            if not html:
                return None
            soup = BeautifulSoup(html, "lxml")
            post_id_match = re.search(r"[&?]p=(\d+)", url)
            if post_id_match:
                pid = post_id_match.group(1)
                post_div = soup.find("div", id=f"post-{pid}") or soup.find(
                    "div", id=f"post_{pid}"
                )
                if post_div:
                    body_tag = post_div.find(
                        attrs={"class": re.compile(r"post_body|msg_body|postcolor", re.I)}
                    )
                    if body_tag:
                        return _clean_post_text(
                            body_tag.get_text(separator="\n", strip=True)
                        )
            topic = self._parse_topic_html(html, url)
            return topic.full_text if topic.posts else None

        # External URL — use web scraper (sync under the hood, run in executor)
        loop = asyncio.get_running_loop()
        page = await loop.run_in_executor(None, self._web_scraper.parse_url, url)
        return page.text if page else None

    # ── HTML parsing helpers ──────────────────────────────────

    def _parse_topic_html(self, html: str, url: str) -> ForumTopic4PDA:
        """Parse a 4PDA topic page; returns a :class:`ForumTopic4PDA`."""
        soup = BeautifulSoup(html, "lxml")

        # ── Title ─────────────────────────────────────────────
        title = ""
        h1 = soup.find("h1", class_=re.compile(r"topic|thread|title", re.I))
        if not h1:
            h1 = soup.find("h1")
        if h1:
            title = h1.get_text(strip=True)
        if not title:
            t = soup.find("title")
            title = t.get_text(strip=True) if t else "4PDA topic"

        topic_id = self._extract_topic_id(url)

        # ── Post containers ────────────────────────────────────
        # IP.Board 3.x (used by 4pda.to): <div id="post-NNNN">
        post_divs = soup.find_all("div", id=re.compile(r"^post[-_]\d+$"))

        # IPS 4.x: <article data-commentid="...">
        if not post_divs:
            post_divs = soup.find_all("article", attrs={"data-commentid": True})

        # Generic fallback: any element with class matching "post"
        if not post_divs:
            post_divs = soup.find_all(
                "div", class_=re.compile(r"\bpost\b", re.I), limit=60
            )

        posts: list[ForumPost4PDA] = []

        for div in post_divs[: self._max_posts]:
            # Author
            author = ""
            for sel in (
                {"class": re.compile(r"post_author|author_name|ipsComment_author", re.I)},
                {"itemprop": "name"},
                {"class": re.compile(r"\bnickname\b|\busername\b", re.I)},
            ):
                tag = div.find(attrs=sel)
                if tag:
                    author = tag.get_text(strip=True)
                    break

            # Body — IP.Board 3.x: div.post_body / div.postcolor; IPS 4.x: .ipsComment_content
            body_tag = div.find(
                attrs={
                    "class": re.compile(
                        r"post_body|msg_body|ipsComment_content|ipsType_richText|postcolor",
                        re.I,
                    )
                }
            )
            body = _clean_post_text(
                (body_tag or div).get_text(separator="\n", strip=True)
            )

            # Date
            date = ""
            time_tag = div.find("time")
            if time_tag:
                date = time_tag.get("datetime", "") or time_tag.get_text(strip=True)

            # Post ID
            post_id = ""
            div_id = div.get("id", "") if hasattr(div, "get") else ""
            m = re.search(r"\d+", str(div_id))
            if m:
                post_id = m.group()

            if len(body) >= MIN_POST_LENGTH:
                posts.append(ForumPost4PDA(author=author, body=body, date=date, post_id=post_id))

        if not posts:
            logger.warning("fourpda_no_posts_parsed", url=url, title=title)
            # Fallback: try trafilatura extraction for the whole page
            try:
                from reviewmind.scrapers.web import WebScraper  # noqa: PLC0415

                page = WebScraper(min_text_length=100).parse_html(html, url=url)
                if page and page.text:
                    posts.append(
                        ForumPost4PDA(author="", body=_clean_post_text(page.text))
                    )
                    logger.info("fourpda_trafilatura_fallback_ok", url=url)
            except Exception:  # noqa: BLE001
                logger.debug("fourpda_trafilatura_fallback_failed", url=url)

        # ── Pinned-post extraction ──────────────────────────────
        pinned: PinnedPostData | None = None
        try:
            pinned = self._parse_pinned_post(soup, url)
        except Exception:  # noqa: BLE001
            logger.debug("fourpda_pinned_post_parse_failed", url=url)

        return ForumTopic4PDA(
            topic_id=topic_id,
            title=title,
            url=url,
            posts=posts,
            source_url=url,
            pinned=pinned,
        )

    def _parse_search_html(self, html: str) -> list[ForumSearchResult4PDA]:
        """Parse a 4PDA search results page; returns result items."""
        soup = BeautifulSoup(html, "lxml")
        results: list[ForumSearchResult4PDA] = []

        # IP.Board 3.x search rows
        rows = soup.find_all(
            attrs={"class": re.compile(r"search_result|ipsSearchResult", re.I)}
        )

        if rows:
            for row in rows:
                a_tag = row.find("a", href=re.compile(r"showtopic=\d+|/t/\d+", re.I))
                if not a_tag:
                    continue
                href = a_tag.get("href", "")
                if not href.startswith("http"):
                    href = urljoin(FOURPDA_BASE_URL, href)
                title = a_tag.get_text(strip=True)
                snippet_tag = row.find(
                    class_=re.compile(r"search_snippet|ipsSearchResult_snippet", re.I)
                )
                snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
                results.append(
                    ForumSearchResult4PDA(
                        title=title,
                        url=href,
                        snippet=snippet,
                        topic_id=self._extract_topic_id(href),
                    )
                )
        else:
            # Fallback: collect all anchor tags pointing to topic URLs
            seen: set[str] = set()
            for a in soup.find_all("a", href=re.compile(r"showtopic=\d+|/t/\d+", re.I)):
                href = a.get("href", "")
                if not href.startswith("http"):
                    href = urljoin(FOURPDA_BASE_URL, href)
                if href in seen:
                    continue
                seen.add(href)
                title = a.get_text(strip=True)
                if title:
                    results.append(
                        ForumSearchResult4PDA(
                            title=title,
                            url=href,
                            topic_id=self._extract_topic_id(href),
                        )
                    )

        return results

    # ── Sync HTTP (used by ingestion pipeline) ────────────────

    def _ensure_sync_client(self) -> httpx.Client:
        """Lazily create a reusable sync httpx client."""
        if self._sync_client is None or self._sync_client.is_closed:
            self._sync_client = httpx.Client(
                headers=_HEADERS,
                timeout=self._timeout,
                follow_redirects=True,
            )
        return self._sync_client

    def _get_sync(self, url: str) -> str | None:
        """Synchronous HTTP GET; return response text or ``None`` on failure."""
        validated = self._validate_url(url)
        if not validated:
            logger.warning("fourpda_invalid_url", url=url)
            return None
        try:
            client = self._ensure_sync_client()
            resp = client.get(validated)
            resp.raise_for_status()
            return resp.text
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "fourpda_http_error_sync",
                url=url,
                status=exc.response.status_code,
            )
            return None
        except httpx.RequestError as exc:
            logger.warning("fourpda_request_error_sync", url=url, error=str(exc))
            return None

    def parse_topic_sync(self, url: str) -> ForumTopic4PDA | None:
        """Synchronously fetch and parse a 4PDA forum topic.

        Intended for use inside the ingestion pipeline's synchronous
        ``_scrape()`` method.  Automatically extracts the pinned post,
        specs, and follows review links if ``follow_pinned_links`` is True.

        Returns ``None`` if the page cannot be retrieved.
        """
        html = self._get_sync(url)
        if not html:
            return None
        topic = self._parse_topic_html(html, url)
        if topic and topic.pinned and self._follow_pinned_links:
            self._follow_review_links_sync(topic.pinned)
        return topic

    def _follow_review_links_sync(self, pinned: PinnedPostData) -> None:
        """Synchronously follow review links from the pinned post."""
        review_links = pinned.review_links[: self._max_followed_links]
        for link in review_links:
            try:
                text = self._fetch_review_text_sync(link.url)
                if text and len(text) >= MIN_POST_LENGTH:
                    pinned.review_texts.append(text)
                    logger.info(
                        "fourpda_review_link_fetched",
                        url=link.url,
                        chars=len(text),
                    )
            except Exception:  # noqa: BLE001
                logger.debug("fourpda_review_link_failed", url=link.url)

    # ── Async HTTP (used by bot query handler) ────────────────

    async def _ensure_async_client(self) -> httpx.AsyncClient:
        """Lazily create a reusable async httpx client."""
        if self._async_client is None or self._async_client.is_closed:
            self._async_client = httpx.AsyncClient(
                headers=_HEADERS,
                timeout=self._timeout,
                follow_redirects=True,
            )
        return self._async_client

    async def _get_async(self, url: str) -> str | None:
        """Asynchronous HTTP GET; return response text or ``None`` on failure."""
        validated = self._validate_url(url)
        if not validated:
            logger.warning("fourpda_invalid_url", url=url)
            return None
        try:
            client = await self._ensure_async_client()
            resp = await client.get(validated)
            resp.raise_for_status()
            return resp.text
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "fourpda_http_error",
                url=url,
                status=exc.response.status_code,
            )
            return None
        except httpx.RequestError as exc:
            logger.warning("fourpda_request_error", url=url, error=str(exc))
            return None

    async def parse_topic(self, url: str) -> ForumTopic4PDA | None:
        """Asynchronously fetch and parse a 4PDA forum topic.

        Extracts the pinned post data (specs + links) and follows
        review links to fetch their content.

        Parameters
        ----------
        url:
            Full URL to a 4PDA topic, e.g.
            ``https://4pda.to/forum/index.php?showtopic=123456``.

        Returns
        -------
        ForumTopic4PDA | None
            Parsed topic, or ``None`` if the page could not be fetched.
        """
        html = await self._get_async(url)
        if not html:
            return None
        topic = self._parse_topic_html(html, url)
        if topic and topic.pinned and self._follow_pinned_links:
            await self._follow_review_links_async(topic.pinned)
        return topic

    async def _follow_review_links_async(self, pinned: PinnedPostData) -> None:
        """Asynchronously follow review links from the pinned post."""
        review_links = pinned.review_links[: self._max_followed_links]
        for link in review_links:
            try:
                text = await self._fetch_review_text_async(link.url)
                if text and len(text) >= MIN_POST_LENGTH:
                    pinned.review_texts.append(text)
                    logger.info(
                        "fourpda_review_link_fetched",
                        url=link.url,
                        chars=len(text),
                    )
            except Exception:  # noqa: BLE001
                logger.debug("fourpda_review_link_failed", url=link.url)

    @staticmethod
    def _rank_search_results(
        results: list[ForumSearchResult4PDA],
    ) -> list[ForumSearchResult4PDA]:
        """Rank search results, preferring 'обсуждение' topics over 'покупка'."""

        def _score(r: ForumSearchResult4PDA) -> int:
            combined = f"{r.title} {r.snippet}"
            if _TOPIC_PREFER_RE.search(combined):
                return 2
            if _TOPIC_DEMOTE_RE.search(combined):
                return 0
            return 1

        return sorted(results, key=_score, reverse=True)

    async def search_topics(
        self,
        query: str,
        max_results: int = DEFAULT_MAX_RESULTS,
    ) -> list[ForumSearchResult4PDA]:
        """Search the 4PDA forum for topics matching *query*.

        Results are ranked to prefer discussion/review topics over
        buy/sell/accessories topics.

        Parameters
        ----------
        query:
            Search string (Russian or English).
        max_results:
            Cap on results to return (default 5).

        Returns
        -------
        list[ForumSearchResult4PDA]
            Matching topics; may be empty if search fails or Cloudflare blocks.
        """
        search_url = (
            f"{FOURPDA_SEARCH_URL}&q={quote_plus(query)}&source_type=forums"
        )
        logger.info("fourpda_search", query=query, url=search_url)
        html = await self._get_async(search_url)
        if not html:
            return []
        results = self._parse_search_html(html)
        ranked = self._rank_search_results(results)
        return ranked[:max_results]
