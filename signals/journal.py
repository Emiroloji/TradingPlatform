"""Signal journal (KURALLAR §8): every issued signal is stored and its outcome filled in later.

The outcome is the trade the backtest engine would have made from that signal (t+1 open,
costs, ATR stop, target, gaps, max holding), so live results compare directly with the backtest.
storage/signals/journal_<market>.parquet, one row per (date, ticker, setup). `mode` is "canlı" for
live signals and "gözlem" for observation-mode setups, which are tracked but never reported as signals.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config
from model.labels import ticker_trade_returns

logger = logging.getLogger(__name__)

OUTCOME_COLUMNS = ["outcome_return", "outcome_bench_return", "outcome_relative", "outcome_exit_date"]


def journal_path(cfg: Config, market: str) -> Path:
    return cfg.storage_path / "signals" / f"journal_{market}.parquet"


def read_journal(cfg: Config, market: str) -> pd.DataFrame:
    path = journal_path(cfg, market)
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def record_signals(signals: pd.DataFrame, cfg: Config, market: str, mode: str = "canlı", setup: str = "baseline") -> int:
    """Append signals; a (date, ticker, setup) already in the journal is never overwritten."""
    if signals.empty:
        return 0
    journal = read_journal(cfg, market)
    new = signals.assign(**{c: np.nan for c in OUTCOME_COLUMNS})
    new["outcome_exit_date"] = pd.NaT
    new["mode"] = mode
    if "setup" not in new:
        new["setup"] = setup
    if not journal.empty:
        journal = journal.assign(setup=journal.get("setup", "baseline"), mode=journal.get("mode", "canlı"))
        known = set(zip(journal["date"], journal["ticker"], journal["setup"]))
        new = new[[k not in known for k in zip(new["date"], new["ticker"], new["setup"])]]
    out = pd.concat([journal, new], ignore_index=True) if not journal.empty else new
    path = journal_path(cfg, market)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    logger.info("Journal [%s]: %d new signals", market, len(new))
    return len(new)


def update_outcomes(cfg: Config, market: str, prices: pd.DataFrame, benchmark: pd.DataFrame) -> int:
    """Fill outcomes for signals whose holding window has finished. Returns rows filled."""
    journal = read_journal(cfg, market)
    if journal.empty:
        return 0
    open_rows = journal["outcome_return"].isna()
    if not open_rows.any():
        return 0
    bench = benchmark.drop(columns="ticker", errors="ignore").set_index("date").sort_index()
    filled = 0
    for ticker in journal.loc[open_rows, "ticker"].unique():
        bars = prices[prices["ticker"] == ticker].drop(columns="ticker").set_index("date").sort_index()
        returns = ticker_trade_returns(bars, bench, cfg)
        rows = open_rows & (journal["ticker"] == ticker)
        for idx in journal.index[rows]:
            day = journal.at[idx, "date"]
            if day not in returns.index or np.isnan(returns.at[day, "trade_return"]):
                continue  # still inside the holding window
            r = returns.loc[day]
            journal.loc[idx, OUTCOME_COLUMNS] = [
                r["trade_return"], r["bench_return"], r["trade_return"] - r["bench_return"], r["exit_date"]
            ]
            filled += 1
    journal.to_parquet(journal_path(cfg, market), index=False)
    logger.info("Journal [%s]: %d outcomes filled", market, filled)
    return filled


def live_performance(journal: pd.DataFrame, mode: str | None = "canlı", setup: str | None = None) -> dict:
    """Closed-signal statistics in the same terms as the backtest report, for one mode/setup."""
    if not journal.empty:
        if mode is not None:
            journal = journal[journal.get("mode", pd.Series("canlı", index=journal.index)) == mode]
        if setup is not None:
            journal = journal[journal["setup"] == setup]
    closed = journal.dropna(subset=["outcome_return"]) if not journal.empty else journal
    if closed.empty:
        return {"signals": len(journal), "closed": 0}
    rel = closed["outcome_relative"]
    return {
        "signals": len(journal),
        "closed": len(closed),
        "win_rate": float((closed["outcome_return"] > 0).mean()),
        "expectancy": float(closed["outcome_return"].mean()),
        "expectancy_excess": float(rel.mean()),
        "beat_benchmark_rate": float((rel > 0).mean()),
    }
