"""Quick test: YouTubeScraper with cookie auth from config."""
import os
os.chdir("/Users/steve/Documents/modern_nlp_project")

from reviewmind.config import get_settings
from reviewmind.scrapers.youtube import YouTubeScraper

settings = get_settings()
print(f"cookie_path config: {settings.youtube_cookies_path}")

scraper = YouTubeScraper(cookie_path=settings.youtube_cookies_path or None)
result = scraper.get_transcript_by_url("https://www.youtube.com/watch?v=qf_lELm9NUQ")
if result:
    print(f"SUCCESS: {result.word_count} words, lang={result.language_code}, generated={result.is_generated}")
    print(f"First 200 chars: {result.text[:200]}")
else:
    print("FAILED: result is None")
