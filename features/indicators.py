"""Category indicators (trend, momentum, volume, volatility) and their 0-100 scores.

Input is one ticker's bars indexed by date (raw OHLCV, ascending). Every value at day t
uses only bars up to and including t. A category score is the share of its conditions
that hold, x100; it is NaN while any input is still warming up.
"""

from __future__ import annotations

import pandas as pd
import pandas_ta as ta

from config import Config


def _nan(index: pd.Index) -> pd.Series:
    return pd.Series(float("nan"), index=index)


def _series(result: pd.Series | None, index: pd.Index) -> pd.Series:
    """pandas-ta returns None when history is shorter than the period; treat that as warm-up."""
    return _nan(index) if result is None else result


def _col(df: pd.DataFrame | None, prefix: str, index: pd.Index) -> pd.Series:
    """First column of a pandas-ta result whose name starts with `prefix` (NaN if no result)."""
    if df is None:
        return _nan(index)
    return df[[c for c in df.columns if c.startswith(prefix)][0]]


def _flag(condition: pd.Series, *inputs: pd.Series) -> pd.Series:
    """Condition as 1.0/0.0, NaN where any input is NaN (so warm-up never reads as 'false')."""
    out = condition.astype(float)
    for s in inputs:
        out[s.isna()] = float("nan")
    return out


def _score(flags: list[pd.Series]) -> pd.Series:
    return pd.concat(flags, axis=1).mean(axis=1, skipna=False) * 100


def trend(bars: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    p = cfg.features.trend
    close = bars["close"]
    out = pd.DataFrame(index=bars.index)
    emas = [_series(ta.ema(close, length=n), bars.index) for n in p.ema_periods]
    for n, ema in zip(p.ema_periods, emas):
        out[f"ema_{n}"] = ema
        out[f"close_vs_ema_{n}"] = close / ema - 1
    adx = ta.adx(bars["high"], bars["low"], close, length=p.adx_period)
    out["adx"], out["dmp"], out["dmn"] = (_col(adx, k, bars.index) for k in ("ADX_", "DMP_", "DMN_"))

    flags = [_flag(close > ema, ema) for ema in emas]
    flags += [_flag(fast > slow, fast, slow) for fast, slow in zip(emas, emas[1:])]
    flags.append(_flag((out["adx"] > p.adx_threshold) & (out["dmp"] > out["dmn"]), out["adx"]))
    out["trend_score"] = _score(flags)
    return out


def momentum(bars: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    p = cfg.features.momentum
    close = bars["close"]
    out = pd.DataFrame(index=bars.index)
    out["rsi"] = _series(ta.rsi(close, length=p.rsi_period), bars.index)
    macd = ta.macd(close, fast=p.macd_fast, slow=p.macd_slow, signal=p.macd_signal)
    out["macd"], out["macd_signal"], out["macd_hist"] = (_col(macd, k, bars.index) for k in ("MACD_", "MACDs_", "MACDh_"))
    stoch = ta.stoch(bars["high"], bars["low"], close, k=p.stoch_k, d=p.stoch_d, smooth_k=p.stoch_smooth_k)
    out["stoch_k"], out["stoch_d"] = (_col(stoch, k, bars.index) for k in ("STOCHk_", "STOCHd_"))
    out["roc"] = _series(ta.roc(close, length=p.roc_period), bars.index)

    prev_hist = out["macd_hist"].shift(1)
    flags = [
        _flag(out["rsi"].between(p.rsi_bull_min, p.rsi_overbought), out["rsi"]),
        _flag(out["macd"] > out["macd_signal"], out["macd"], out["macd_signal"]),
        _flag(out["macd_hist"] > prev_hist, out["macd_hist"], prev_hist),
        _flag((out["stoch_k"] > out["stoch_d"]) & (out["stoch_k"] < p.stoch_overbought), out["stoch_k"], out["stoch_d"]),
        _flag(out["roc"] > 0, out["roc"]),
    ]
    out["momentum_score"] = _score(flags)
    return out


def volume(bars: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    p = cfg.features.volume
    out = pd.DataFrame(index=bars.index)
    out["obv"] = _series(ta.obv(bars["close"], bars["volume"]), bars.index)
    out["obv_ema"] = _series(ta.ema(out["obv"], length=p.obv_ema_period), bars.index)
    out["cmf"] = _series(ta.cmf(bars["high"], bars["low"], bars["close"], bars["volume"], length=p.cmf_period), bars.index)
    avg_volume = bars["volume"].rolling(p.avg_volume_period, min_periods=p.avg_volume_period).mean()
    out["volume_ratio"] = bars["volume"] / avg_volume

    flags = [
        _flag(out["obv"] > out["obv_ema"], out["obv_ema"]),
        _flag(out["cmf"] > 0, out["cmf"]),
        _flag(out["volume_ratio"] >= p.volume_ratio_min, out["volume_ratio"]),
    ]
    out["volume_score"] = _score(flags)
    return out


def volatility(bars: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Calm or contracting volatility scores high (conservative mode prefers quiet setups)."""
    p = cfg.features.volatility
    out = pd.DataFrame(index=bars.index)
    out["atr"] = _series(ta.atr(bars["high"], bars["low"], bars["close"], length=cfg.risk.atr_period), bars.index)
    out["atr_pct"] = out["atr"] / bars["close"]
    bb = ta.bbands(bars["close"], length=p.bb_period, std=p.bb_std)
    out["bb_width"] = _col(bb, "BBB_", bars.index)

    base = p.baseline_days
    atr_median = out["atr_pct"].rolling(base, min_periods=base).median()
    bb_median = out["bb_width"].rolling(base, min_periods=base).median()
    flags = [
        _flag(out["atr_pct"] < atr_median, atr_median),
        _flag(out["bb_width"] < bb_median, bb_median),
    ]
    out["volatility_score"] = _score(flags)
    return out


def compute_indicators(bars: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    return pd.concat(
        [trend(bars, cfg), momentum(bars, cfg), volume(bars, cfg), volatility(bars, cfg)], axis=1
    )
