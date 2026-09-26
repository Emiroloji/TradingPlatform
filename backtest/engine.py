"""Portfolio backtest: signal at close t, fill at open t+1, costs, ATR stop, target, time exit.

Daily order of events (so nothing uses information before it exists):
  1. open  - fill yesterday's signals at the open, using slots free after yesterday's exits
  2. intraday - stop / target checks on today's bar (a bar touching both counts as the stop)
  3. close - time exits after `max_holding_days`, then mark the portfolio to market
  4. queue today's signals for tomorrow's open
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import Config
from signals.risk import position_shares, select_within_limits, stop_price, target_price

logger = logging.getLogger(__name__)

SIGNAL_COLUMNS = ["date", "ticker", "sector", "total_score", "atr", "regime"]


@dataclass(frozen=True)
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.Series  # portfolio value at each close
    signals_seen: int
    signals_skipped: int  # no slot / sector limit / zero size / no next bar


@dataclass
class _Position:
    ticker: str
    sector: str
    regime: str
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_price: float
    shares: int
    stop: float
    target: float
    bars_held: int = 0


def _bar_table(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {t: g.set_index("date")[["open", "high", "low", "close"]] for t, g in prices.groupby("ticker")}


def run_backtest(
    signals: pd.DataFrame, prices: pd.DataFrame, cfg: Config, rank_by: str = "total_score"
) -> BacktestResult:
    """`signals`: one row per (date, ticker) that qualified at that day's close (SIGNAL_COLUMNS).
    `prices`: long-format clean bars for the same tickers.
    `rank_by`: column deciding which signals get the free slots (highest first).
    """
    bt = cfg.backtest
    bars = _bar_table(prices)
    calendar = pd.DatetimeIndex(sorted(prices["date"].unique()))
    by_day = {d: g.sort_values(rank_by, ascending=False) for d, g in signals.groupby("date")}

    cash = float(cfg.risk.capital)
    open_positions: list[_Position] = []
    trades: list[dict] = []
    equity: dict[pd.Timestamp, float] = {}
    pending = signals.iloc[0:0]
    skipped = 0
    prev_day: pd.Timestamp | None = None

    def close_position(pos: _Position, day: pd.Timestamp, level: float, reason: str) -> None:
        nonlocal cash
        exit_price = level * (1 - bt.slippage)
        cash += pos.shares * exit_price * (1 - bt.commission)
        cost_in = pos.entry_price * (1 + bt.commission)
        trades.append(
            {
                "ticker": pos.ticker,
                "sector": pos.sector,
                "regime": pos.regime,
                "signal_date": pos.signal_date,
                "entry_date": pos.entry_date,
                "exit_date": day,
                "entry_price": pos.entry_price,
                "exit_price": exit_price,
                "stop": pos.stop,
                "target": pos.target,
                "shares": pos.shares,
                "exit_reason": reason,
                "holding_days": pos.bars_held,
                "return_net": exit_price * (1 - bt.commission) / cost_in - 1,
                "pnl": pos.shares * (exit_price * (1 - bt.commission) - cost_in),
            }
        )

    for day in calendar:
        # 1. fills at the open
        if not pending.empty:
            held = {p.ticker for p in open_positions}
            chosen = select_within_limits(pending, [p.sector for p in open_positions], held, cfg)
            skipped += len(pending) - len(chosen)
            mark = equity[prev_day] if prev_day is not None else cash  # sized on yesterday's close, not today's
            for _, sig in chosen.iterrows():
                tb = bars.get(sig["ticker"])
                if tb is None or day not in tb.index:
                    skipped += 1
                    continue
                entry = tb.at[day, "open"] * (1 + bt.slippage)
                stop = stop_price(entry, sig["atr"], cfg)
                shares = position_shares(mark, entry, stop, cfg)
                shares = min(shares, int(cash // (entry * (1 + bt.commission))))
                if shares <= 0:
                    skipped += 1
                    continue
                cash -= shares * entry * (1 + bt.commission)
                open_positions.append(
                    _Position(
                        sig["ticker"], sig["sector"], sig["regime"], sig["date"], day, entry, shares, stop,
                        target_price(entry, stop, cfg),
                    )
                )
            pending = signals.iloc[0:0]

        # 2-3. intraday stop/target, then time exit at the close
        still_open = []
        for pos in open_positions:
            tb = bars[pos.ticker]
            if day not in tb.index:
                still_open.append(pos)  # no bar today (halt): carry over
                continue
            o, h, low, c = tb.loc[day, ["open", "high", "low", "close"]]
            pos.bars_held += 1
            if day > pos.entry_date and o <= pos.stop:
                close_position(pos, day, o, "stop_gap")
            elif low <= pos.stop:
                close_position(pos, day, pos.stop, "stop")
            elif day > pos.entry_date and o >= pos.target:
                close_position(pos, day, o, "target_gap")
            elif h >= pos.target:
                close_position(pos, day, pos.target, "target")
            elif pos.bars_held >= bt.max_holding_days:
                close_position(pos, day, c, "time")
            elif day == tb.index[-1] and day != calendar[-1]:  # ticker's data stops early
                close_position(pos, day, c, "data_end")
            else:
                still_open.append(pos)
        open_positions = still_open

        equity[day] = cash + sum(p.shares * bars[p.ticker]["close"].asof(day) for p in open_positions)

        # 4. today's signals wait for tomorrow's open
        if day in by_day:
            pending = by_day[day]
        prev_day = day

    for pos in open_positions:  # backtest end: mark out at the last close
        last = calendar[-1]
        close_position(pos, last, bars[pos.ticker]["close"].asof(last), "backtest_end")

    trade_df = pd.DataFrame(trades)
    logger.info("Backtest: %d signals, %d trades, %d skipped", len(signals), len(trade_df), skipped)
    return BacktestResult(trade_df, pd.Series(equity, name="equity"), len(signals), skipped)


def add_benchmark_returns(trades: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    """Benchmark return over each trade (entry-day open -> exit-day close) and the excess return."""
    if trades.empty:
        return trades.assign(bench_return=pd.Series(dtype=float), excess_return=pd.Series(dtype=float))
    b = benchmark.set_index("date").sort_index()
    bench_open = b["open"].reindex(trades["entry_date"]).to_numpy()
    fallback = b["close"].shift(1).reindex(trades["entry_date"]).to_numpy()  # index had no open that day
    bench_open = np.where(np.isnan(bench_open), fallback, bench_open)
    bench_close = np.array([b["close"].asof(d) for d in trades["exit_date"]])
    out = trades.copy()
    out["bench_return"] = bench_close / bench_open - 1
    out["excess_return"] = out["return_net"] - out["bench_return"]
    return out
