"""Relative strength versus the benchmark index (excess return over several windows)."""

from __future__ import annotations

import pandas as pd

from config import Config


def relative_strength(prices: pd.DataFrame, benchmark: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """rs_<n> = stock return over n bars minus benchmark return over the same dates.

    `relative_strength` is the mean across windows (NaN until the longest window is ready).
    The benchmark is aligned to the stock's dates using only its last known close.
    """
    close = prices["close"]
    bench = benchmark["close"].reindex(close.index.union(benchmark.index)).ffill().reindex(close.index)
    out = pd.DataFrame(index=prices.index)
    for n in cfg.features.relative.periods:
        out[f"rs_{n}"] = (close / close.shift(n) - 1) - (bench / bench.shift(n) - 1)
    out["relative_strength"] = out.mean(axis=1, skipna=False)
    return out
