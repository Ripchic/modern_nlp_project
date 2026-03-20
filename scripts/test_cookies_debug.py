"""Debug: test _build_api approach directly."""
import os
os.chdir("/Users/steve/Documents/modern_nlp_project")

from reviewmind.scrapers.youtube import YouTubeScraper

scraper = YouTubeScraper(cookie_path="cookies.txt")
api = scraper._api

# Check the session inside the fetcher
fetcher = api._fetcher
http_client = fetcher._http_client
print(f"http_client type: {type(http_client)}")
print(f"cookies count: {len(http_client.cookies)}")
for c in list(http_client.cookies)[:5]:
    print(f"  cookie: {c.name} domain={c.domain}")

# Try fetch
try:
    result = api.fetch("qf_lELm9NUQ", languages=["ru", "en"])
    print(f"FETCH: SUCCESS {len(result.snippets)} snippets")
except Exception as e:
    print(f"FETCH: FAILED {type(e).__name__}")
