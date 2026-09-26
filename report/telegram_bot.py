"""Daily Telegram summary (conservative signals only) and operational alerts.

The message text is built by pure functions (also saved as the daily report file); sending is
a thin wrapper that skips, with a warning, when TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID are unset.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import date

from dotenv import load_dotenv

from ai.report_writer import REGIME_TR, SignalRow, fmt_pct
from config import Config

logger = logging.getLogger(__name__)

DISCLAIMER = "Not: Bu bir olasılık tahminidir, yatırım tavsiyesi değildir."


@dataclass
class DailyContext:
    as_of: date
    market: str
    regime: str | None
    funnel: list[tuple[str, int]]
    setup: dict | None
    setup_note: str
    ml_note: str
    signals: list[tuple[SignalRow, str]] = field(default_factory=list)  # (row, commentary)
    vetoed: list[tuple[str, str]] = field(default_factory=list)  # (ticker, reason)
    news_blocked: list[tuple[str, str]] = field(default_factory=list)  # (ticker, status)
    live: dict = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)


def _signal_block(s: SignalRow, commentary: str, setup: dict | None) -> list[str]:
    period = setup.get("period", "") if setup else ""
    ml = f"ML olasılığı: {fmt_pct(s.prob)}" if s.prob is not None else "ML olasılığı: kullanılmıyor (onaylı model yok)"
    return [
        f"{s.ticker} — Temkinli Sinyal",
        f"Toplam skor: {s.total_score:.0f}/100 | Toplama skoru: {s.accumulation_score:.0f}/100",
        ml,
        f"Piyasa rejimi: {REGIME_TR.get(s.regime, s.regime)}",
        f"Gemini haber kontrolü: {s.news_status}" + (f" — {s.news_summary}" if s.news_summary else ""),
        f"Geçmiş benzer kurulum ({period}, out-of-sample, maliyetler dahil, endekse göre):",
        f"- {s.bt_trades} işlem, kazanma oranı {fmt_pct(s.bt_win_rate)}",
        f"- Expectancy: {fmt_pct(s.bt_expectancy, signed=True, decimals=2)} / işlem | Max drawdown: {fmt_pct(s.bt_max_dd, signed=True)}",
        f"Referans giriş: {s.entry_ref:.2f} | Önerilen stop: {s.stop:.2f} | Pozisyon: {s.position_size} adet".replace(".", ","),
        f"Yorum: {commentary}",
        "",
    ]


def build_daily_message(ctx: DailyContext) -> str:
    regime = REGIME_TR.get(ctx.regime, "bilinmiyor") if ctx.regime else "bilinmiyor"
    lines = [f"{ctx.market.upper()} günlük tarama — {ctx.as_of:%d.%m.%Y}", f"Piyasa rejimi (USD bazlı endeks): {regime}", ""]
    if ctx.signals:
        lines.append(f"{len(ctx.signals)} temkinli sinyal:")
        lines.append("")
        for row, commentary in ctx.signals:
            lines += _signal_block(row, commentary, ctx.setup)
    else:
        lines += ["Bugün temkinli sinyal yok.", ""]

    lines.append("Filtre hunisi (kalan hisse):")
    lines += [f"- {name}: {count}" for name, count in ctx.funnel]
    if ctx.vetoed:
        lines.append("Gemini vetosu:")
        lines += [f"- {t}: {r}" for t, r in ctx.vetoed]
    if ctx.news_blocked:
        lines.append("Haber kontrolü nedeniyle elenen:")
        lines += [f"- {t}: {st}" for t, st in ctx.news_blocked]
    lines += ["", f"Kurulum: {ctx.setup_note}", f"Model: {ctx.ml_note}"]
    if ctx.live.get("closed"):
        lv = ctx.live
        lines.append(
            f"Canlı takip: {lv['signals']} sinyal, {lv['closed']} kapandı; kazanma oranı {fmt_pct(lv['win_rate'])}, "
            f"endekse göre expectancy {fmt_pct(lv['expectancy_excess'], signed=True, decimals=2)} / işlem"
        )
    elif ctx.live:
        lines.append(f"Canlı takip: {ctx.live.get('signals', 0)} sinyal, henüz kapanan yok")
    if ctx.limitations:
        lines.append("Kısıtlar:")
        lines += [f"- {x}" for x in ctx.limitations]
    lines += ["", DISCLAIMER]
    return "\n".join(lines)


def split_message(text: str, limit: int) -> list[str]:
    """Split on line boundaries so no chunk exceeds Telegram's limit."""
    chunks, current = [], ""
    for line in text.split("\n"):
        while len(line) > limit:  # a single overlong line: hard cut
            chunks.append(line[:limit])
            line = line[limit:]
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) > limit:
            chunks.append(current)
            current = line
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def send(text: str, cfg: Config) -> bool:
    """Send (in chunks). Returns False, without raising, if unconfigured or Telegram fails."""
    load_dotenv(cfg.root / ".env")
    token, chat_id = os.environ.get("TELEGRAM_BOT_TOKEN", ""), os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        logger.warning("Telegram not configured (.env TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID); message not sent")
        return False

    async def _send_all() -> None:
        from telegram import Bot

        async with Bot(token) as bot:
            for chunk in split_message(text, cfg.telegram.max_message_chars):
                await bot.send_message(
                    chat_id=chat_id, text=chunk, read_timeout=cfg.telegram.timeout_seconds,
                    write_timeout=cfg.telegram.timeout_seconds,
                )

    try:
        asyncio.run(_send_all())
        return True
    except Exception as exc:
        logger.error("Telegram send failed: %s", type(exc).__name__)  # message may echo the token URL
        return False


def send_alert(text: str, cfg: Config) -> bool:
    return send(f"⚠️ UYARI: {text}", cfg)
