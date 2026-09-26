"""Training target for a signal at day t. Two definitions, chosen by `label.type`:

trade_outcome (default) - "would the backtest's own trade have beaten the index?"
  The trade the engine would open is simulated with the same rules: fill at the t+1 open with
  slippage and commission, stop entry - k x ATR_t, target at R x risk, gap fills at the open,
  a bar touching both counts as the stop, time exit at the close of day t+H. Label 1 if its net
  return minus the benchmark return (entry-day open -> exit-day close) > `min_trade_excess`.
  The model thus learns exactly the metric the system is judged on (excess expectancy).

triple_barrier - "within H days, did the stock beat the index by X% before touching its stop?"
  First touch of close-to-close excess return >= target_excess_return -> 1; stop / tie / time -> 0.

Either label is only known after day t+H, recorded as `label_end`; training must only use rows
whose label_end precedes the test window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta as ta
from numpy.lib.stride_tricks import sliding_window_view

from config import Config
from features import last_known


def _first_true(mask: np.ndarray) -> np.ndarray:
    """Index of the first True per row, or a sentinel beyond the row if none."""
    return np.where(mask.any(axis=1), mask.argmax(axis=1), mask.shape[1])


def horizon(cfg: Config) -> int:
    return cfg.backtest.max_holding_days if cfg.label.type == "trade_outcome" else cfg.label.horizon_days


def _pick(windows: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """windows[i, idx[i]] with idx clipped into the row."""
    return windows[np.arange(len(idx)), np.minimum(idx, windows.shape[1] - 1)]


def _windows(bars: pd.DataFrame, bench: pd.DataFrame, cfg: Config, h: int):
    """Future windows t+1..t+h for every t that has one, plus ATR_t and the benchmark entry/closes."""
    n = len(bars)

    atr = ta.atr(bars["high"], bars["low"], bars["close"], length=cfg.risk.atr_period)
    atr = np.full(n, np.nan) if atr is None else atr.to_numpy()
    b_close = last_known(bench["close"], bars.index).to_numpy()
    b_open = bench["open"].reindex(bars.index).to_numpy()
    b_open = np.where(np.isnan(b_open), np.roll(b_close, 1), b_open)  # index closed that morning

    m = n - h  # rows t = 0..m-1 have a full future window t+1..t+h
    window = {c: sliding_window_view(bars[c].to_numpy()[1:], h)[:m] for c in ["open", "high", "low", "close"]}
    b_closes = sliding_window_view(b_close[1:], h)[:m]
    bench_entry = b_open[1 : m + 1]
    invalid = np.isnan(atr[:m]) | np.isnan(window["open"][:, 0]) | np.isnan(bench_entry)
    return window, b_closes, bench_entry, atr[:m], invalid


def ticker_trade_returns(bars: pd.DataFrame, bench: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """For every day t: net return of the engine-rule trade signalled at t and the benchmark return
    over the same span (NaN where the trade has not finished yet). Used for labels and the journal."""
    h = cfg.backtest.max_holding_days
    out = pd.DataFrame({"trade_return": np.nan, "bench_return": np.nan, "exit_date": pd.NaT}, index=bars.index)
    if len(bars) <= h:
        return out
    window, b_closes, bench_entry, atr, invalid = _windows(bars, bench, cfg, h)
    trade_return, bench_return, exit_idx = _trade_returns(window, b_closes, bench_entry, atr, cfg)
    m = len(trade_return)
    trade_return[invalid], bench_return[invalid] = np.nan, np.nan
    out.iloc[:m, 0], out.iloc[:m, 1] = trade_return, bench_return
    out.iloc[:m, 2] = bars.index[1 + np.arange(m) + exit_idx]
    return out


def ticker_labels(bars: pd.DataFrame, bench: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """`bars`, `bench`: date-indexed OHLC. Returns label (1/0/NaN) and label_end per day."""
    h = horizon(cfg)
    out = pd.DataFrame({"label": np.nan, "label_end": pd.NaT}, index=bars.index)
    if len(bars) <= h:
        return out
    window, b_closes, bench_entry, atr, invalid = _windows(bars, bench, cfg, h)
    m = len(bench_entry)
    if cfg.label.type == "trade_outcome":
        trade_return, bench_return, _ = _trade_returns(window, b_closes, bench_entry, atr, cfg)
        label = (trade_return - bench_return > cfg.label.min_trade_excess).astype(float)
    else:
        label = _triple_barrier(window, b_closes, bench_entry, atr, cfg)
    label[invalid] = np.nan

    out.iloc[:m, 0] = label
    out.iloc[:m, 1] = bars.index[h : h + m]  # label known after day t+h
    return out


def _trade_returns(w: dict, b_closes: np.ndarray, bench_entry: np.ndarray, atr: np.ndarray, cfg: Config):
    """(net trade return, benchmark return, exit offset) per row, following the engine's rules."""
    bt, h = cfg.backtest, w["open"].shape[1]
    entry = w["open"][:, 0] * (1 + bt.slippage)
    stop = entry - cfg.risk.atr_stop_multiplier * atr
    target = entry + bt.target_r_multiple * (entry - stop)

    first_stop = _first_true(w["low"] <= stop[:, None])
    first_target = _first_true(w["high"] >= target[:, None])
    stop_exit = (first_stop <= first_target) & (first_stop < h)  # tie -> stop, as in the engine
    target_exit = ~stop_exit & (first_target < h)
    exit_idx = np.where(stop_exit, first_stop, np.where(target_exit, first_target, h - 1))

    open_at_exit = _pick(w["open"], exit_idx)
    after_entry_day = exit_idx > 0  # gaps only after the entry bar (entry itself is the open)
    level = np.where(
        stop_exit,
        np.where(after_entry_day & (open_at_exit <= stop), open_at_exit, stop),
        np.where(
            target_exit,
            np.where(after_entry_day & (open_at_exit >= target), open_at_exit, target),
            _pick(w["close"], exit_idx),
        ),
    )
    exit_price = level * (1 - bt.slippage)
    trade_return = exit_price * (1 - bt.commission) / (entry * (1 + bt.commission)) - 1
    bench_return = _pick(b_closes, exit_idx) / bench_entry - 1
    return trade_return, bench_return, exit_idx


def _triple_barrier(w: dict, b_closes: np.ndarray, bench_entry: np.ndarray, atr: np.ndarray, cfg: Config) -> np.ndarray:
    entry = w["open"][:, 0]
    stop = entry - cfg.risk.atr_stop_multiplier * atr
    excess = w["close"] / entry[:, None] - b_closes / bench_entry[:, None]
    first_up = _first_true(excess >= cfg.label.target_excess_return)
    if cfg.label.use_stop_barrier:
        first_stop = _first_true(w["low"] <= stop[:, None])
    else:
        first_stop = np.full(len(entry), w["open"].shape[1])
    return (first_up < first_stop).astype(float)  # tie -> stop wins


def make_labels_frame(prices: pd.DataFrame, benchmark: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Long prices + benchmark bars -> DataFrame [date, ticker, label, label_end]."""
    bench = benchmark.drop(columns="ticker", errors="ignore").set_index("date").sort_index()
    frames = []
    for ticker, bars in prices.groupby("ticker", sort=True):
        lab = ticker_labels(bars.drop(columns="ticker").set_index("date").sort_index(), bench, cfg)
        frames.append(lab.assign(ticker=ticker))
    out = pd.concat(frames).rename_axis("date").reset_index()
    return out[["date", "ticker", "label", "label_end"]]


def make_labels(prices: pd.DataFrame, benchmark: pd.DataFrame, cfg: Config) -> pd.Series:
    """MIMARI §7 contract: label per (date, ticker)."""
    return make_labels_frame(prices, benchmark, cfg).set_index(["date", "ticker"])["label"]
