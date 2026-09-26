"""Short Turkish commentary for a signal (KURALLAR §6: Gemini does not produce numbers).

The facts are formatted by us; Gemini only puts them into prose. Its text is then checked:
every number in it must be one of the numbers we supplied, and no certainty wording
(config `banned_phrases`) may appear. Otherwise, or on any API error, a deterministic
template built from the same facts is used instead.
"""

from __future__ import annotations

import logging
import re

from pydantic import BaseModel

from ai.gemini_client import LLM, GeminiError
from config import Config

logger = logging.getLogger(__name__)

_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")

SYSTEM = (
    "Sen bir karar destek raporu yazarısın. Verilen bilgileri 3-4 cümlelik, sade Türkçe bir yoruma dönüştür.\n"
    "Kurallar:\n"
    "- YALNIZCA verilen sayıları, verildiği biçimde kullan; yeni sayı, hesap, tahmin veya hedef fiyat üretme.\n"
    "- Kesinlik ya da garanti ifade eden kelimeler kullanma; sonuçları olasılık olarak anlat.\n"
    "- Alım-satım talimatı verme; bu bir karar destek notudur."
)


class SignalRow(BaseModel):
    """What a report needs about one signal (MIMARI §4 signals table)."""

    ticker: str
    name: str
    total_score: float
    accumulation_score: float
    prob: float | None  # None while no approved ML model exists
    regime: str
    news_status: str
    news_summary: str
    entry_ref: float
    stop: float
    position_size: int
    bt_trades: int
    bt_win_rate: float
    bt_expectancy: float  # per trade, net of costs, vs benchmark
    bt_max_dd: float


REGIME_TR = {"bull": "yükseliş", "bear": "düşüş", "sideways": "yatay"}


def fmt_pct(x: float, signed: bool = False, decimals: int = 1) -> str:
    s = f"{abs(x) * 100:.{decimals}f}".replace(".", ",")
    return f"{'+' if x >= 0 else '-'}%{s}" if signed else f"%{s}"


def facts(s: SignalRow) -> list[str]:
    """Numbers formatted once, here; the commentary may only repeat these."""
    ml = f"ML olasılığı: {fmt_pct(s.prob)}" if s.prob is not None else "ML olasılığı: kullanılmıyor (onaylı model yok)"
    return [
        f"Hisse: {s.ticker} ({s.name})",
        f"Toplam skor: {s.total_score:.0f}/100",
        f"Toplama skoru: {s.accumulation_score:.0f}/100",
        ml,
        f"Piyasa rejimi: {REGIME_TR.get(s.regime, s.regime)}",
        f"Haber kontrolü: {s.news_status}" + (f" — {s.news_summary.rstrip('. ')}" if s.news_summary else ""),
        f"Geçmiş benzer kurulum: {s.bt_trades} işlem, kazanma oranı {fmt_pct(s.bt_win_rate)}, "
        f"endekse göre expectancy {fmt_pct(s.bt_expectancy, signed=True, decimals=2)} / işlem, max drawdown {fmt_pct(s.bt_max_dd, signed=True)}",
        f"Referans giriş: {s.entry_ref:.2f}, stop: {s.stop:.2f}, önerilen adet: {s.position_size}".replace(".", ","),
    ]


def _numbers(text: str) -> set[str]:
    return {n.replace(",", ".").rstrip("0").rstrip(".") or "0" for n in _NUMBER.findall(text)}


def check_commentary(text: str, allowed_facts: list[str], cfg: Config) -> list[str]:
    """Problems with a generated text (empty list = acceptable)."""
    problems = []
    invented = _numbers(text) - _numbers("\n".join(allowed_facts))
    if invented:
        problems.append(f"invented numbers: {sorted(invented)}")
    lowered = text.lower()
    problems += [f"banned phrase: {p}" for p in cfg.gemini.banned_phrases if p.lower() in lowered]
    if not text.strip():
        problems.append("empty")
    return problems


def template_commentary(s: SignalRow) -> str:
    """Fallback when Gemini is unavailable or its text fails the checks: no new facts, no repetition."""
    return (
        f"{s.ticker} için otomatik yorum üretilemedi; yukarıdaki değerler kural bazlı temkinli kurulumun "
        "geçmiş veriye dayalı olasılıklarıdır."
    )


def write_commentary(signal: SignalRow, llm: LLM, cfg: Config) -> str:
    """MIMARI §7: commentary using only the given numbers; template fallback on any doubt."""
    lines = facts(signal)
    try:
        text = llm.generate_text("Bilgiler:\n" + "\n".join(lines), SYSTEM).strip()
    except GeminiError as exc:
        logger.warning("%s: commentary fallback (API: %s)", signal.ticker, exc)
        return template_commentary(signal)
    problems = check_commentary(text, lines, cfg)
    if problems:
        logger.warning("%s: commentary rejected (%s), using template", signal.ticker, "; ".join(problems))
        return template_commentary(signal)
    return text
