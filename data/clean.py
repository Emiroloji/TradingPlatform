"""Data quality checks: missing days, zero volume, invalid bars, abnormal jumps.

Raw data is never modified (KURALLAR §3); a cleaned copy is returned.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from config import Config

logger = logging.getLogger(__name__)

OHLC = ["open", "high", "low", "close"]


@dataclass(frozen=True)
class QualityReport:
    """One row per ticker with check results and the exclusion decision."""

    summary: pd.DataFrame
    calendar_start: pd.Timestamp
    calendar_end: pd.Timestamp

    @property
    def excluded(self) -> list[str]:
        return self.summary.index[self.summary["excluded"]].tolist()

    def to_markdown(self) -> str:
        s = self.summary
        lines = [
            "# Veri Kalite Raporu",
            "",
            f"- Takvim: {self.calendar_start.date()} → {self.calendar_end.date()}",
            f"- Hisse sayısı: {len(s)}",
            f"- Dışlanan: {int(s['excluded'].sum())}",
            f"- Sorunlu (uyarı): {int((s['issues'] != '').sum())}",
            "",
            "## Sorunlu hisseler",
            "",
        ]
        flagged = s[s["issues"] != ""]
        if flagged.empty:
            lines.append("Sorun bulunmadı.")
        else:
            lines.append("| ticker | satır | eksik % | sıfır hacim % | geçersiz | sıçrama | dışlandı | nedenler |")
            lines.append("|---|---|---|---|---|---|---|---|")
            for ticker, r in flagged.iterrows():
                lines.append(
                    f"| {ticker} | {r['rows']} | {r['missing_ratio']:.2%} | {r['zero_volume_ratio']:.2%} "
                    f"| {r['invalid_rows']} | {r['jump_days']} | {'evet' if r['excluded'] else 'hayır'} "
                    f"| {r['issues']} |"
                )
        lines += [
            "",
            "## Sıçrama tarihleri (bölünme/bedelsiz şüphesi)",
            "",
        ]
        jumps = s[s["jump_days"] > 0]
        lines += [f"- {t}: {r['jump_dates']}" for t, r in jumps.iterrows()] or ["Yok."]
        lines += [
            "",
            "## Kısıtlar",
            "",
            "- Survivorship bias: geçmiş endeks üyeliği verisi yok; evren bugünkü listeden oluşur.",
        ]
        return "\n".join(lines) + "\n"


def _check_ticker(
    bars: pd.DataFrame, calendar: pd.DatetimeIndex, cfg: Config, market: str, check_volume: bool = True
) -> tuple[dict, pd.DataFrame]:
    q = cfg.data.quality
    bars = bars.sort_values("date")

    expected = calendar[calendar >= bars["date"].min()]
    missing = expected.difference(pd.DatetimeIndex(bars["date"]))
    missing_ratio = len(missing) / max(len(expected), 1)

    nan_mask = bars[OHLC].isna().any(axis=1)
    invalid_mask = ~nan_mask & (
        (bars[OHLC] <= 0).any(axis=1)
        | (bars["high"] < bars["low"])
        | (bars["close"] > bars["high"])
        | (bars["close"] < bars["low"])
    )
    zero_volume_mask = (bars["volume"].fillna(0) <= 0) if check_volume else pd.Series(False, index=bars.index)
    zero_volume_ratio = float(zero_volume_mask.mean()) if len(bars) else 0.0

    clean = bars[~(nan_mask | invalid_mask | zero_volume_mask)]
    returns = clean["close"].pct_change().abs()
    jump_mask = returns > q.max_abs_daily_return
    jump_dates = clean.loc[jump_mask, "date"].dt.strftime("%Y-%m-%d").tolist()

    reasons = []
    if missing_ratio > q.max_missing_ratio:
        reasons.append("eksik gün")
    if zero_volume_ratio > q.max_zero_volume_ratio:
        reasons.append("sıfır hacim")
    if jump_dates:
        reasons.append("aşırı sıçrama")
    if invalid_mask.any():
        reasons.append("geçersiz bar")  # dropped, not a reason to exclude on its own

    blocking = {"eksik gün", "sıfır hacim"} | ({"aşırı sıçrama"} if q.jump_excludes[market] else set())
    exclude = q.exclude_on_issues and any(r in blocking for r in reasons)
    row = {
        "rows": len(bars),
        "first_date": bars["date"].min(),
        "last_date": bars["date"].max(),
        "missing_days": len(missing),
        "missing_ratio": missing_ratio,
        "nan_rows": int(nan_mask.sum()),
        "invalid_rows": int(invalid_mask.sum()),
        "zero_volume_days": int(zero_volume_mask.sum()),
        "zero_volume_ratio": zero_volume_ratio,
        "jump_days": len(jump_dates),
        "jump_dates": ", ".join(jump_dates),
        "issues": ", ".join(reasons),
        "excluded": exclude,
    }
    return row, clean


def clean_prices(
    df: pd.DataFrame, cfg: Config, market: str, price_only: frozenset[str] = frozenset()
) -> tuple[pd.DataFrame, QualityReport]:
    """Check every ticker against the shared trading calendar and return cleaned bars + report.

    The calendar is the union of dates across all tickers in `df`, so pass the benchmark
    together with the stocks. Cleaned data drops NaN/invalid/zero-volume bars and, if
    `data.quality.exclude_on_issues` is set, every ticker that breaches a threshold.
    Symbols in `price_only` (FX rates, which have no volume) skip the volume checks. Whether an
    abnormal jump excludes a ticker depends on the market (`data.quality.jump_excludes`).
    """
    calendar = pd.DatetimeIndex(sorted(df.loc[~df["ticker"].isin(price_only), "date"].unique()))
    rows: dict[str, dict] = {}
    kept: list[pd.DataFrame] = []

    for ticker, bars in df.groupby("ticker", sort=True):
        row, clean = _check_ticker(bars, calendar, cfg, market, check_volume=ticker not in price_only)
        rows[str(ticker)] = row
        if row["issues"]:
            logger.warning("%s: %s%s", ticker, row["issues"], " -> excluded" if row["excluded"] else "")
        if not row["excluded"]:
            kept.append(clean)

    summary = pd.DataFrame.from_dict(rows, orient="index").rename_axis("ticker")
    report = QualityReport(summary=summary, calendar_start=calendar.min(), calendar_end=calendar.max())
    cleaned = pd.concat(kept, ignore_index=True) if kept else df.iloc[0:0]
    logger.info("Cleaned %d tickers, excluded %d", summary["excluded"].eq(False).sum(), len(report.excluded))
    return cleaned.reset_index(drop=True), report
