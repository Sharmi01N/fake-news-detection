"""
live_feed.py
------------
Pulls current headlines from a handful of public news RSS feeds. This
powers the site's "Live Wire" — a continuously-refreshing stream of real
news that gets run through the classifier automatically, which is what
makes the app feel like a real-time monitor rather than a one-off text
checker.

Swap FEEDS for whatever outlets you want to watch (adding more
Bangladeshi sources is a natural extension mentioned in the proposal's
literature review / research gap).
"""
import feedparser

FEEDS = [
    {"name": "BBC News", "url": "http://feeds.bbci.co.uk/news/rss.xml"},
    {"name": "Al Jazeera", "url": "https://www.aljazeera.com/xml/rss/all.xml"},
    {"name": "NPR", "url": "https://feeds.npr.org/1001/rss.xml"},
    {"name": "The Daily Star (BD)", "url": "https://www.thedailystar.net/rss.xml"},
]


def fetch_live_headlines(limit_per_feed: int = 5) -> list[dict]:
    """Return a flat list of recent headline dicts from all configured feeds.

    Each item: {source, title, summary, link, published}
    Feeds that fail to load (offline, DNS blocked, etc.) are skipped
    silently so one bad feed doesn't take down the whole ticker.
    """
    items = []
    for feed in FEEDS:
        try:
            parsed = feedparser.parse(feed["url"])
        except Exception:
            continue
        if getattr(parsed, "bozo", 0) and not getattr(parsed, "entries", None):
            continue
        for entry in parsed.entries[:limit_per_feed]:
            items.append({
                "source": feed["name"],
                "title": entry.get("title", "").strip(),
                "summary": (entry.get("summary", "") or entry.get("description", "")).strip(),
                "link": entry.get("link", ""),
                "published": entry.get("published", ""),
            })
    return items
