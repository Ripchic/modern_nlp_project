"""Quick test script to verify yt-dlp can fetch subtitles for a given video."""
import json
import re
import sys
import urllib.request

import yt_dlp


def fetch_subtitles_text(url: str, languages=("ru", "en"), cookies_browser: str | None = "chrome") -> str | None:
    """Fetch subtitle text using yt-dlp. Returns cleaned text or None."""
    ydl_opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": list(languages),
        "subtitlesformat": "json3",
        "quiet": True,
        "no_warnings": True,
    }
    if cookies_browser:
        ydl_opts["cookiesfrombrowser"] = (cookies_browser,)

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    title = info.get("title", "")
    channel = info.get("channel", "")
    print(f"Title: {title}")
    print(f"Channel: {channel}")

    subs = info.get("subtitles", {})
    auto_subs = info.get("automatic_captions", {})
    print(f"Manual subtitle langs: {list(subs.keys())[:5]}")
    print(f"Auto caption langs (sample): {list(auto_subs.keys())[:10]}")

    # Try each language, preferring manual subs over auto-generated
    for lang in languages:
        for label, source in [("manual", subs), ("auto", auto_subs)]:
            if lang not in source:
                continue
            # Find json3 format
            for fmt in source[lang]:
                if fmt["ext"] == "json3" and fmt.get("url"):
                    print(f"\nFetching {label} {lang} json3 subtitles...")
                    req = urllib.request.Request(fmt["url"])
                    with urllib.request.urlopen(req) as resp:
                        data = json.loads(resp.read())
                    events = data.get("events", [])
                    texts = []
                    for ev in events:
                        for seg in ev.get("segs", []):
                            t = seg.get("utf8", "").strip()
                            if t and t != "\n":
                                texts.append(t)
                    full = " ".join(texts)
                    full = re.sub(r"\s+", " ", full).strip()
                    word_count = len(full.split())
                    print(f"Got {word_count} words ({label}, lang={lang})")
                    print(f"Preview: {full[:300]}")
                    return full
    print("No subtitles found!")
    return None


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "https://www.youtube.com/watch?v=qf_lELm9NUQ"
    browser = sys.argv[2] if len(sys.argv) > 2 else "chrome"
    try:
        text = fetch_subtitles_text(url, cookies_browser=browser)
        if text:
            print(f"\nSUCCESS: {len(text.split())} words extracted")
        else:
            print("\nFAILED: no text extracted")
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
