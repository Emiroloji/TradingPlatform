"""Liquidity filter: average daily turnover (close x volume, in USD) over a lookback window."""

from __future__ import annotations

import pandas as pd

from config import Config


def liquidity(prices: pd.DataFrame, cfg: Config, market: str, fx: pd.Series | None = None) -> pd.DataFrame:
    """`avg_turnover` in USD and `liquidity_ok` (False during warm-up).

    `fx` is local currency per USD aligned to `prices.index`; None means prices are already USD.
    """
    n = cfg.liquidity.lookback_days
    threshold = cfg.liquidity.min_avg_turnover_usd[market]
    turnover = prices["close"] * prices["volume"]
    if fx is not None:
        turnover = turnover / fx
    avg_turnover = turnover.rolling(n, min_periods=n).mean()
    return pd.DataFrame(
        {"avg_turnover": avg_turnover, "liquidity_ok": (avg_turnover >= threshold).fillna(False)},
        index=prices.index,
    )
