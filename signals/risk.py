"""Risk rules (KURALLAR §7): ATR stop, fixed-fraction sizing, position and sector limits.

Shared by the backtest engine and the live signal flow so both size trades identically.
"""

from __future__ import annotations

import math

import pandas as pd

from config import Config


def stop_price(entry: float, atr: float, cfg: Config) -> float:
    return entry - cfg.risk.atr_stop_multiplier * atr


def target_price(entry: float, stop: float, cfg: Config) -> float:
    return entry + cfg.backtest.target_r_multiple * (entry - stop)


def position_shares(equity: float, entry: float, stop: float, cfg: Config) -> int:
    """(capital x risk %) / (entry - stop), whole shares; 0 if the stop is not below entry."""
    risk_per_share = entry - stop
    if risk_per_share <= 0 or math.isnan(risk_per_share):
        return 0
    return int(equity * cfg.risk.risk_per_trade // risk_per_share)


def select_within_limits(
    candidates: pd.DataFrame, held_sectors: list[str], held_tickers: set[str], cfg: Config
) -> pd.DataFrame:
    """Highest-ranked candidates (already sorted) that fit max_positions and max_per_sector."""
    slots = cfg.risk.max_positions - len(held_sectors)
    sector_count = pd.Series(held_sectors, dtype=object).value_counts().to_dict()
    chosen = []
    for idx, row in candidates.iterrows():
        if len(chosen) >= slots:
            break
        if row["ticker"] in held_tickers or sector_count.get(row["sector"], 0) >= cfg.risk.max_per_sector:
            continue
        sector_count[row["sector"]] = sector_count.get(row["sector"], 0) + 1
        chosen.append(idx)
    return candidates.loc[chosen]


def size_positions(candidates: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Live sizing (MIMARI §7): add stop and position size for candidates entering at `entry_ref`."""
    out = candidates.sort_values("total_score", ascending=False)
    out = select_within_limits(out, [], set(), cfg).copy()
    out["stop"] = [stop_price(e, a, cfg) for e, a in zip(out["entry_ref"], out["atr"])]
    out["position_size"] = [position_shares(cfg.risk.capital, e, s, cfg) for e, s in zip(out["entry_ref"], out["stop"])]
    return out
