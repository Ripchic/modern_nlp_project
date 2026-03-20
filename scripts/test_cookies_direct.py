"""Direct test: same code that worked before."""
from youtube_transcript_api import YouTubeTranscriptApi
from http.cookiejar import MozillaCookieJar
from requests import Session

jar = MozillaCookieJar("cookies.txt")
jar.load(ignore_discard=True, ignore_expires=True)
print(f"Loaded {len(jar)} cookies")

session = Session()
session.cookies = jar

api = YouTubeTranscriptApi(http_client=session)

try:
    result = api.fetch("qf_lELm9NUQ", languages=["ru", "en"])
    print(f"SUCCESS: {len(result.snippets)} snippets")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")
