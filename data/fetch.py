"""Daily OHLCV download from yfinance."""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from config import Config, MarketConfig

logger = logging.getLogger(__name__)

PRICE_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]

_YF_RENAME = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}


def load_universe(market: MarketConfig, cfg: Config) -> pd.DataFrame:
    """Universe CSV (ticker, name, sector) plus the yfinance `symbol` column."""
    universe = pd.read_csv(cfg.root / Path(market.universe))
    universe["symbol"] = universe["ticker"] + market.suffix
    return universe


def fetch_prices(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    """Long-format daily prices for yfinance symbols, `end` inclusive.

    Symbols without any data are logged and absent from the result.
    """
    if not tickers:
        return pd.DataFrame(columns=PRICE_COLUMNS)

    raw = yf.download(
        tickers,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),  # yfinance end is exclusive
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        progress=False,
        threads=True,
    )

    frames = []
    for symbol in tickers:
        if raw.empty or symbol not in raw.columns.get_level_values(0):
            logger.warning("No data returned for %s", symbol)
            continue
        part = raw[symbol].rename(columns=_YF_RENAME).dropna(how="all")
        if part.empty:
            logger.warning("No data returned for %s", symbol)
            continue
        part = part.rename_axis("date").reset_index()
        part["ticker"] = symbol
        frames.append(part)

    if not frames:
        return pd.DataFrame(columns=PRICE_COLUMNS)

    prices = pd.concat(frames, ignore_index=True)
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    prices[["open", "high", "low", "close", "adj_close", "volume"]] = prices[
        ["open", "high", "low", "close", "adj_close", "volume"]
    ].astype(float)
    logger.info("Fetched %d rows for %d/%d symbols", len(prices), len(frames), len(tickers))
    return prices[PRICE_COLUMNS].sort_values(["ticker", "date"], ignore_index=True)


def fetch_splits(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    """Stock split events [date, ticker, ratio] (ratio 10 = 10-for-1) from yfinance, `end` inclusive."""
    raw = yf.download(
        tickers, start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(), interval="1d",
        actions=True, auto_adjust=False, group_by="ticker", progress=False, threads=True,
    )
    frames = []
    for symbol in tickers:
        if raw.empty or symbol not in raw.columns.get_level_values(0):
            continue
        s = raw[symbol]["Stock Splits"]
        s = s[s > 0]
        frames.append(pd.DataFrame({"date": s.index.normalize(), "ticker": symbol, "ratio": s.to_numpy(dtype=float)}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["date", "ticker", "ratio"])
