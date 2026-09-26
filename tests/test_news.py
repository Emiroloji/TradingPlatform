from datetime import datetime, timezone

from ai import news
from ai.news import NewsItem, collect_news, parse_rss

RSS = """<?xml version="1.0"?><rss><channel>
<item><title>THY yeni uçak siparişi verdi</title><link>https://example.com/a</link>
<pubDate>Thu, 24 Sep 2026 10:00:00 GMT</pubDate><description>özet a</description></item>
<item><title>Eski haber</title><link>https://example.com/b</link>
<pubDate>Mon, 01 Jan 2024 10:00:00 GMT</pubDate></item>
<item><title>Başlıksız tarih yok</title><link>https://example.com/c</link></item>
</channel></rss>"""

AS_OF = datetime(2026, 9, 26, tzinfo=timezone.utc)


def _item(title: str, day: int, source: str = "yfinance") -> NewsItem:
    return NewsItem(news._item_id(title, ""), source, title, "", "", datetime(2026, 9, day, tzinfo=timezone.utc))


def test_parse_rss():
    items = parse_rss(RSS, "google_news_rss")
    assert [i.title for i in items] == ["THY yeni uçak siparişi verdi", "Eski haber"]
    assert items[0].published == datetime(2026, 9, 24, 10, tzinfo=timezone.utc)


def test_collect_dedupes_filters_window_and_survives_failed_source(cfg, monkeypatch):
    monkeypatch.setattr(news, "from_yfinance", lambda s, c: [_item("A", 25), _item("B", 10), _item("C", 27)])

    def broken(*_):
        raise TimeoutError("feed down")

    monkeypatch.setattr(news, "from_google_news_rss", broken)
    items, failed = collect_news("THYAO", "Türk Hava Yolları", "THYAO.IS", cfg, AS_OF, "hl=tr")
    assert [i.title for i in items] == ["A"]  # B too old, C after as_of
    assert failed == ["google_news_rss"]


def test_same_story_from_two_sources_counted_once(cfg, monkeypatch):
    monkeypatch.setattr(news, "from_yfinance", lambda s, c: [_item("A", 25)])
    monkeypatch.setattr(news, "from_google_news_rss", lambda t, n, c, loc: [_item("A", 25, "google_news_rss")])
    items, failed = collect_news("THYAO", "THY", "THYAO.IS", cfg, AS_OF, "hl=tr")
    assert len(items) == 1 and failed == []
