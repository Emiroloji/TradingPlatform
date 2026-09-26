"""News collection for candidate tickers (live only; no historical archive exists).

Sources (config `news.sources`), each failing independently without stopping the others:
  yfinance         - Yahoo Finance news for the symbol (same provider as the price data)
  google_news_rss  - Google News RSS search (a published feed, not HTML scraping)
  reddit           - social posts via Reddit's free official API (app-only OAuth); skipped
                     silently when REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET are not set
KAP has no official public API, so it is not scraped (KURALLAR §10). Social posts have no
historical archive here, so like news they are used live only (Gemini sentiment/veto).
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import yfinance as yf
from dotenv import load_dotenv

from config import Config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NewsItem:
    id: str
    source: str
    title: str
    summary: str
    url: str
    published: datetime  # timezone-aware UTC


def _item_id(title: str, url: str) -> str:
    return hashlib.sha256(f"{title.strip().lower()}|{url}".encode()).hexdigest()[:16]


def from_yfinance(symbol: str, cfg: Config) -> list[NewsItem]:
    items = []
    for raw in yf.Ticker(symbol).news or []:
        c = raw.get("content", raw)
        title = (c.get("title") or "").strip()
        url = ((c.get("canonicalUrl") or {}).get("url")) or c.get("link") or ""
        published = c.get("pubDate") or c.get("displayTime")
        if not title or not published:
            continue
        items.append(
            NewsItem(
                id=_item_id(title, url),
                source="yfinance",
                title=title,
                summary=(c.get("summary") or c.get("description") or "").strip(),
                url=url,
                published=datetime.fromisoformat(published.replace("Z", "+00:00")),
            )
        )
    return items


def parse_rss(xml_text: str, source: str) -> list[NewsItem]:
    items = []
    for node in ET.fromstring(xml_text).iter("item"):
        title = (node.findtext("title") or "").strip()
        url = (node.findtext("link") or "").strip()
        pub = node.findtext("pubDate")
        if not title or not pub:
            continue
        items.append(
            NewsItem(
                id=_item_id(title, url),
                source=source,
                title=title,
                summary=(node.findtext("description") or "").strip(),
                url=url,
                published=parsedate_to_datetime(pub).astimezone(timezone.utc),
            )
        )
    return items


def from_google_news_rss(ticker: str, name: str, cfg: Config, locale: str) -> list[NewsItem]:
    n = cfg.news
    query = urllib.parse.quote(n.query_template.format(name=name, ticker=ticker))
    url = n.google_news_rss_url.format(query=query, locale=locale)
    with urllib.request.urlopen(url, timeout=n.timeout_seconds) as resp:  # noqa: S310 (fixed https feed from config)
        return parse_rss(resp.read().decode("utf-8"), "google_news_rss")


class NotConfigured(Exception):
    """An optional source without credentials: skipped, not a failure."""


def _reddit_token(cfg: Config) -> str:
    load_dotenv(cfg.root / ".env")
    cid, secret = os.environ.get("REDDIT_CLIENT_ID", ""), os.environ.get("REDDIT_CLIENT_SECRET", "")
    if not cid or not secret:
        raise NotConfigured("reddit")
    req = urllib.request.Request(
        cfg.news.reddit_token_url,
        data=b"grant_type=client_credentials",
        headers={"Authorization": "Basic " + base64.b64encode(f"{cid}:{secret}".encode()).decode(),
                 "User-Agent": "TradingPlatform/0.1 (personal research)"},
    )
    with urllib.request.urlopen(req, timeout=cfg.news.timeout_seconds) as resp:  # noqa: S310 (fixed reddit URL)
        return json.loads(resp.read())["access_token"]


def parse_reddit(payload: dict) -> list[NewsItem]:
    items = []
    for child in payload.get("data", {}).get("children", []):
        d = child.get("data", {})
        title = (d.get("title") or "").strip()
        if not title or "created_utc" not in d:
            continue
        url = "https://www.reddit.com" + d.get("permalink", "")
        items.append(
            NewsItem(
                id=_item_id(title, url),
                source=f"reddit/r/{d.get('subreddit', '')}",
                title=title,
                summary=(d.get("selftext") or "").strip()[:500],
                url=url,
                published=datetime.fromtimestamp(d["created_utc"], tz=timezone.utc),
            )
        )
    return items


def from_reddit(ticker: str, subreddits: list[str], cfg: Config) -> list[NewsItem]:
    if not subreddits:
        raise NotConfigured("reddit: no subreddits for this market")
    token = _reddit_token(cfg)
    query = urllib.parse.urlencode({"q": f'"{ticker}" OR "${ticker}"', "restrict_sr": 1, "sort": "new", "t": "week", "limit": 25})
    url = cfg.news.reddit_search_url.format(subs="+".join(subreddits)) + "?" + query
    req = urllib.request.Request(url, headers={"Authorization": f"bearer {token}", "User-Agent": "TradingPlatform/0.1 (personal research)"})
    with urllib.request.urlopen(req, timeout=cfg.news.timeout_seconds) as resp:  # noqa: S310 (fixed reddit URL)
        return parse_reddit(json.loads(resp.read()))


def collect_news(
    ticker: str, name: str, symbol: str, cfg: Config, as_of: datetime, locale: str, subreddits: list[str] | None = None
) -> tuple[list[NewsItem], list[str]]:
    """Recent unique items for one ticker, newest first, plus the sources that failed."""
    fetchers = {
        "yfinance": lambda: from_yfinance(symbol, cfg),
        "google_news_rss": lambda: from_google_news_rss(ticker, name, cfg, locale),
        "reddit": lambda: from_reddit(ticker, subreddits or [], cfg),
    }
    items: dict[str, NewsItem] = {}
    failed = []
    for source in cfg.news.sources:
        try:
            for item in fetchers[source]():
                items.setdefault(item.id, item)
        except NotConfigured:
            continue  # optional source without credentials/settings
        except Exception as exc:  # one broken source must not stop the others
            logger.warning("News source %s failed for %s: %s", source, ticker, exc)
            failed.append(source)

    since = as_of - timedelta(days=cfg.news.lookback_days)
    recent = [i for i in items.values() if since <= i.published <= as_of]
    recent.sort(key=lambda i: i.published, reverse=True)
    return recent[: cfg.news.max_items], failed
