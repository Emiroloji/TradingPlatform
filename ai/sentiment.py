"""News sentiment and risk veto via Gemini (KURALLAR §6).

Gemini returns structured JSON (`GeminiVerdict`), validated against the schema. Outcomes:
  ok                  - valid answer; `veto` decides
  no_news             - nothing to read; not an error, no veto
  invalid_response    - answer failed validation -> "veri yok", blocks the signal (conservative)
  check_failed        - API error -> "haber kontrolü yapılamadı", blocks the signal
Only `passes` results may become signals. Vetoes are appended to storage/logs/veto_log.csv.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from ai import news
from ai.gemini_client import LLM, GeminiError
from ai.news import NewsItem
from config import Config

logger = logging.getLogger(__name__)

_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def _numbers(text: str) -> set[str]:
    return {n.replace(",", ".") for n in _NUMBER.findall(text)}


Status = Literal["ok", "no_news", "invalid_response", "check_failed"]

STATUS_TEXT = {
    "ok": "haber kontrolü yapıldı",
    "no_news": "son dönemde haber yok",
    "invalid_response": "veri yok (geçersiz yanıt)",
    "check_failed": "haber kontrolü yapılamadı",
}


class GeminiVerdict(BaseModel):
    """Exact JSON contract asked from Gemini (FAZLAR Faz 4)."""

    duyarlilik: float = Field(ge=-1, le=1)
    veto: bool
    veto_nedeni: str
    ozet: str


class SentimentResult(BaseModel):
    ticker: str
    status: Status
    sentiment: float | None = None
    veto: bool = False
    veto_reason: str = ""
    summary: str = ""
    news_count: int = 0

    @property
    def passes(self) -> bool:
        return self.status in ("ok", "no_news") and not self.veto

    @property
    def status_text(self) -> str:
        return STATUS_TEXT[self.status]


def system_prompt(cfg: Config) -> str:
    categories = "\n".join(f"- {c}" for c in cfg.gemini.veto_categories)
    return (
        "Sen Borsa İstanbul hisseleri için bir risk kontrol asistanısın. Görevin yalnızca verilen haberleri "
        "okuyup duyarlılığı ve ciddi risk olup olmadığını değerlendirmek.\n"
        "Kurallar:\n"
        "- Fiyat, getiri veya hedef tahmini YAPMA; sayı üretme.\n"
        "- Yalnızca verilen haberleri kullan, dışarıdan bilgi ekleme.\n"
        "- duyarlilik: -1 (çok olumsuz) ile 1 (çok olumlu) arası.\n"
        "- veto: aşağıdaki durumlardan biri haberlerde açıkça varsa true, yoksa false:\n"
        f"{categories}\n"
        "- veto_nedeni: veto true ise hangi durum ve hangi haber; false ise boş bırak.\n"
        "- ozet: Türkçe, en fazla iki cümle, kesinlik ifadesi kullanmadan."
    )


def build_prompt(ticker: str, name: str, items: list[NewsItem]) -> str:
    lines = [f"Hisse: {ticker} ({name})", "Haberler (yeniden eskiye):"]
    for i, item in enumerate(items, 1):
        summary = f" — {item.summary[:300]}" if item.summary else ""
        lines.append(f"{i}. [{item.published:%Y-%m-%d}] {item.title}{summary}")
    return "\n".join(lines)


def analyze_news(ticker: str, items: list[NewsItem], llm: LLM, cfg: Config, name: str = "") -> SentimentResult:
    """MIMARI §7: sentiment + veto for one ticker, always returning a result (never raising)."""
    if not items:
        return SentimentResult(ticker=ticker, status="no_news", sentiment=0.0)
    prompt = build_prompt(ticker, name or ticker, items)
    try:
        raw = llm.generate_json(prompt, system_prompt(cfg), GeminiVerdict)
    except GeminiError as exc:
        logger.warning("%s: news check failed (%s)", ticker, exc)
        return SentimentResult(ticker=ticker, status="check_failed", news_count=len(items))
    try:
        verdict = GeminiVerdict.model_validate_json(raw)
    except ValidationError:
        logger.warning("%s: Gemini answer failed schema validation", ticker)
        return SentimentResult(ticker=ticker, status="invalid_response", news_count=len(items))
    if verdict.veto and not verdict.veto_nedeni.strip():
        logger.warning("%s: veto without a reason -> treated as invalid", ticker)
        return SentimentResult(ticker=ticker, status="invalid_response", news_count=len(items))
    summary = verdict.ozet.strip()
    invented = _numbers(summary) - _numbers(prompt)
    if invented:  # KURALLAR §6: numbers may only be quoted from the news, never produced
        logger.warning("%s: summary dropped, numbers not in the news: %s", ticker, sorted(invented))
        summary = ""
    return SentimentResult(
        ticker=ticker,
        status="ok",
        sentiment=verdict.duyarlilik,
        veto=verdict.veto,
        veto_reason=verdict.veto_nedeni.strip() if verdict.veto else "",
        summary=summary,
        news_count=len(items),
    )


def log_veto(result: SentimentResult, items: list[NewsItem], cfg: Config, as_of: datetime) -> None:
    """Append a vetoed candidate with its reason and the headlines it was based on (KURALLAR §6)."""
    path = cfg.storage_path / "logs" / "veto_log.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame(
        [{"as_of": as_of.isoformat(timespec="seconds"), "ticker": result.ticker, "reason": result.veto_reason,
          "summary": result.summary, "headlines": " | ".join(i.title for i in items)}]
    )
    row.to_csv(path, mode="a", header=not path.exists(), index=False)
    logger.info("%s vetoed: %s", result.ticker, result.veto_reason)


def check_candidates(
    candidates: pd.DataFrame, names: dict[str, str], llm: LLM, cfg: Config, as_of: datetime, locale: str
) -> pd.DataFrame:
    """MIMARI §3 step 6: news + Gemini check for candidate rows (columns `ticker`), adding
    gemini_sentiment, gemini_veto, veto_reason, news_status, news_summary, news_count, gemini_passes."""
    rows = []
    for symbol in candidates["ticker"]:
        ticker = symbol.split(".")[0]
        items, failed = news.collect_news(ticker, names.get(symbol, ticker), symbol, cfg, as_of, locale)
        if failed and not items:
            result = SentimentResult(ticker=symbol, status="check_failed")  # could not read any news
        else:
            result = analyze_news(symbol, items, llm, cfg, name=names.get(symbol, ticker))
        if result.veto:
            log_veto(result, items, cfg, as_of)
        rows.append(
            {
                "ticker": symbol,
                "gemini_sentiment": result.sentiment,
                "gemini_veto": result.veto,
                "veto_reason": result.veto_reason,
                "news_status": result.status_text,
                "news_summary": result.summary,
                "news_count": result.news_count,
                "gemini_passes": result.passes,
            }
        )
    if not rows:
        return candidates
    return candidates.merge(pd.DataFrame(rows), on="ticker", how="left")
