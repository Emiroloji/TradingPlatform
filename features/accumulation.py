"""Accumulation score (0-100): signs of quiet institutional buying.

Four conditions, each evaluated on day t with bars up to t only:
  1. contraction  - the recent price band is narrow relative to the stock's own history
  2. volume_up    - volume inside the band is higher than before it
  3. divergence   - OBV or A/D rises while price stays flat or falls
  4. spring       - a recent dip below the band low that closed back inside (simple Wyckoff)
The score comes from how many hold (config `score_by_conditions`), so a single condition
can never produce a high score.
"""

from __future__ import annotations

import pandas as pd
import pandas_ta as ta

from config import Config
from features.indicators import _flag, _series

CONDITIONS = ["acc_contraction", "acc_volume_up", "acc_divergence", "acc_spring"]


def accumulation_conditions(prices: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    p = cfg.features.accumulation
    w, base = p.window_days, p.baseline_days
    high, low, close, volume = prices["high"], prices["low"], prices["close"], prices["volume"]
    out = pd.DataFrame(index=prices.index)

    band_high = high.rolling(w, min_periods=w).max()
    band_low = low.rolling(w, min_periods=w).min()
    out["acc_range_pct"] = (band_high - band_low) / close
    range_median = out["acc_range_pct"].rolling(base, min_periods=base).median()
    out["acc_contraction"] = _flag(out["acc_range_pct"] < range_median * p.range_contraction_ratio, range_median)

    band_volume = volume.rolling(w, min_periods=w).mean()
    prior_volume = volume.shift(w).rolling(base, min_periods=base).mean()
    out["acc_volume_ratio"] = band_volume / prior_volume
    out["acc_volume_up"] = _flag(out["acc_volume_ratio"] > p.volume_increase_ratio, prior_volume)

    obv = _series(ta.obv(close, volume), prices.index)
    ad = _series(ta.ad(high, low, close, volume), prices.index)
    obv_change, ad_change = obv - obv.shift(w), ad - ad.shift(w)
    price_return = close / close.shift(w) - 1
    out["acc_divergence"] = _flag(
        ((obv_change > 0) | (ad_change > 0)) & (price_return <= p.divergence_max_price_return),
        obv_change,
        ad_change,
        price_return,
    )

    prior_band_low = low.shift(1).rolling(w, min_periods=w).min()
    spring_day = _flag((low < prior_band_low) & (close > prior_band_low), prior_band_low)
    out["acc_spring"] = spring_day.rolling(p.spring_lookback_days, min_periods=p.spring_lookback_days).max()
    return out


def accumulation_score(prices: pd.DataFrame, cfg: Config) -> pd.Series:
    conditions = accumulation_conditions(prices, cfg)[CONDITIONS]
    count = conditions.sum(axis=1, skipna=False)
    table = dict(enumerate(cfg.features.accumulation.score_by_conditions))
    return count.map(table).rename("accumulation_score")
