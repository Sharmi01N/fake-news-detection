"""
fetch_article.py
-----------------
Given a news article URL, download the page and extract a best-effort
title + body text so it can be run through the same classifier pipeline
used for pasted text. This is what makes the app "real-time": instead of
only accepting text a user has already copied, it can go fetch and read
a live URL on demand.

No external article-parsing library is required (keeps requirements.txt
small) — extraction is done with requests + BeautifulSoup using a few
standard heuristics (og:title, <article> tag, longest cluster of <p>
tags).
"""
import re

import requests
from bs4 import BeautifulSoup

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 FakeNewsDetector/1.0"
    )
}


class ArticleFetchError(Exception):
    pass


def _extract_title(soup):
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        return og_title["content"].strip()
    if soup.title and soup.title.string:
        return re.sub(r"\s+", " ", soup.title.string).strip()
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(" ", strip=True)
    return None


def _extract_paragraphs(container):
    paragraphs = [p.get_text(" ", strip=True) for p in container.find_all("p")]
    paragraphs = [p for p in paragraphs if len(p.split()) > 4]
    return paragraphs


def fetch_article_text(url: str, timeout: int = 8) -> dict:
    """Fetch a URL and return {"title": str|None, "text": str, "url": str}.

    Raises ArticleFetchError on network failure or if the page can't be
    reached/parsed.
    """
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url

    try:
        resp = requests.get(url, headers=_HEADERS, timeout=timeout)
        resp.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise ArticleFetchError(str(exc)) from exc

    content_type = resp.headers.get("Content-Type", "")
    if "html" not in content_type and not resp.text.strip().startswith("<"):
        raise ArticleFetchError("That URL doesn't look like an HTML news page.")

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "noscript"]):
        tag.decompose()

    title = _extract_title(soup)

    # Prefer an <article> tag if present; it's usually the actual story body
    # and avoids nav/related-links boilerplate.
    article_tag = soup.find("article")
    paragraphs = _extract_paragraphs(article_tag) if article_tag else []

    # Fall back to the whole page if <article> was missing or too short.
    if len(" ".join(paragraphs).split()) < 40:
        paragraphs = _extract_paragraphs(soup)

    text = " ".join(paragraphs)

    return {"title": title, "text": text, "url": url}
