"""Market regime from the benchmark index: bull / bear / sideways.

bull:     close above EMA, EMA sloping up, and ADX shows a real trend
bear:     close below EMA, EMA sloping down, and ADX shows a real trend
sideways: anything else (mixed signals or no trend strength)
Warm-up days are NaN, which the signal filter treats as "not allowed".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta as ta

from config import Config
from features.indicators import _col, _series

BULL, BEAR, SIDEWAYS = "bull", "bear", "sideways"


def market_regime(benchmark: pd.DataFrame, cfg: Config) -> pd.Series:
    p = cfg.features.regime
    close = benchmark["close"]
    ema = _series(ta.ema(close, length=p.ema_period), benchmark.index)
    slope = ema / ema.shift(p.slope_days) - 1
    adx = _col(ta.adx(benchmark["high"], benchmark["low"], close, length=p.adx_period), "ADX_", benchmark.index)

    trending = adx >= p.adx_min
    bull = trending & (close > ema) & (slope > 0)
    bear = trending & (close < ema) & (slope < 0)
    regime = pd.Series(np.select([bull, bear], [BULL, BEAR], default=SIDEWAYS), index=benchmark.index, dtype=object)
    regime[ema.isna() | slope.isna() | adx.isna()] = np.nan
    return regime.rename("regime")
