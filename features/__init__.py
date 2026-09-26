"""Feature layer entry point: one row per ticker/day with every feature and score."""

from __future__ import annotations

import logging

import pandas as pd

from config import Config
from features.accumulation import accumulation_conditions, accumulation_score
from features.indicators import compute_indicators
from features.insider import coverage_end, insider_features
from features.liquidity import liquidity
from features.regime import market_regime
from features.relative import relative_strength
from features.scoring import total_score

logger = logging.getLogger(__name__)


def _by_date(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns="ticker", errors="ignore").set_index("date").sort_index()


def last_known(series: pd.Series, index: pd.Index) -> pd.Series:
    """Value of `series` on each date of `index`, using only its most recent past observation."""
    return series.reindex(series.index.union(index)).ffill().reindex(index)


def in_usd(bars: pd.DataFrame, fx: pd.Series | None) -> pd.DataFrame:
    """OHLC divided by the last known FX rate (local currency per USD)."""
    if fx is None:
        return bars
    rate = last_known(fx, bars.index)
    return bars.assign(**{c: bars[c] / rate for c in ["open", "high", "low", "close"]})


def ticker_features(
    bars: pd.DataFrame,
    bench: pd.DataFrame,
    regime: pd.Series,
    cfg: Config,
    market: str,
    fx: pd.Series | None = None,
    insider: tuple[pd.DataFrame, pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """All features for one ticker; `bars` and `bench` are date-indexed OHLCV, `fx` a close series,
    `insider` this ticker's Form 4 transactions and the data coverage end."""
    out = pd.concat(
        [
            compute_indicators(bars, cfg),
            accumulation_conditions(bars, cfg),
            accumulation_score(bars, cfg),
            relative_strength(bars, bench, cfg),
            liquidity(bars, cfg, market, None if fx is None else last_known(fx, bars.index)),
        ],
        axis=1,
    )
    if insider is not None:
        tx, covered_until = insider
        out = out.join(insider_features(out.index, tx, out["avg_turnover"], cfg, covered_until))
    out["fundamental_score"] = float("nan")  # no fundamental data source yet
    # regime is a benchmark property; carry its last known value onto the stock's dates
    out["regime"] = last_known(regime, out.index)
    return out


def compute_features(
    prices: pd.DataFrame,
    benchmark: pd.DataFrame,
    cfg: Config,
    market: str,
    fx: pd.DataFrame | None = None,
    insider: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Long-format prices (many tickers) + benchmark bars (+ FX bars) -> features table (MIMARI §4).

    With `fx`, the regime is computed on the benchmark in USD and liquidity is measured in USD.
    With `insider` (Form 4 transactions with a `ticker` column), insider features are added.
    """
    bench = _by_date(benchmark)
    fx_close = None if fx is None else _by_date(fx)["close"]
    regime = market_regime(in_usd(bench, fx_close), cfg)
    frames = []
    covered_until = coverage_end(insider) if insider is not None else None
    for ticker, bars in prices.groupby("ticker", sort=True):
        tx = None if insider is None else (insider[insider["ticker"] == ticker], covered_until)
        feats = ticker_features(_by_date(bars), bench, regime, cfg, market, fx_close, tx)
        feats.insert(0, "ticker", ticker)
        frames.append(feats)
    table = pd.concat(frames).rename_axis("date").reset_index()
    table["total_score"] = total_score(table, cfg)
    logger.info("Features: %d rows, %d tickers, %d columns", len(table), table["ticker"].nunique(), table.shape[1])
    return table
