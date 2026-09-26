"""Insider-trading features (US, SEC Form 4) per ticker/day.

On day t only filings with filing_date < t count (a filing is known from the next session),
within the last `insider.window_days` calendar days. Days after the data coverage ends are NaN:
"no dataset yet" must never read as "no insider activity".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import Config

COLUMNS = [
    "insider_buy_value", "insider_sell_value", "insider_net_ratio", "insider_buyers",
    "insider_cluster_buy", "insider_buy_to_turnover",
]


def insider_features(
    dates: pd.DatetimeIndex, tx: pd.DataFrame, avg_turnover: pd.Series, cfg: Config, coverage_end: pd.Timestamp
) -> pd.DataFrame:
    """`tx`: one ticker's transactions (filing_date, owner_cik, code, value).

    A filing counts on day t when filing_date < t <= filing_date + window.
    """
    window = pd.Timedelta(days=cfg.insider.window_days)
    t = dates.to_numpy()
    tx = tx.sort_values("filing_date")
    filed = tx["filing_date"].to_numpy()
    lo = np.searchsorted(filed, t - np.timedelta64(window), side="left")
    hi = np.searchsorted(filed, t, side="left")  # strictly before t

    def rolling_sum(values: np.ndarray) -> np.ndarray:
        cum = np.concatenate([[0.0], np.cumsum(values)])
        return cum[hi] - cum[lo]

    is_buy = (tx["code"] == "P").to_numpy()
    b = rolling_sum(np.where(is_buy, tx["value"].to_numpy(), 0.0))
    s = rolling_sum(np.where(tx["code"].to_numpy() == "S", tx["value"].to_numpy(), 0.0))

    # distinct buyers: each owner is active on the union of its (filing, filing + window] spans
    buyers = np.zeros(len(t))
    for _, own in tx[is_buy].groupby("owner_cik"):
        diff = np.zeros(len(t) + 1)
        f = own["filing_date"].to_numpy()
        np.add.at(diff, np.searchsorted(t, f, side="right"), 1)
        np.add.at(diff, np.searchsorted(t, f + np.timedelta64(window), side="right"), -1)
        buyers += np.cumsum(diff)[:-1] > 0

    total = b + s
    out = pd.DataFrame(
        {
            "insider_buy_value": b,
            "insider_sell_value": s,
            "insider_net_ratio": np.divide(b - s, total, out=np.zeros_like(total), where=total > 0),
            "insider_buyers": buyers,
            "insider_cluster_buy": (buyers >= cfg.insider.cluster_min_buyers).astype(float),
        },
        index=dates,
    )
    out["insider_buy_to_turnover"] = out["insider_buy_value"] / avg_turnover.reindex(dates)
    out.loc[out.index > coverage_end + pd.Timedelta(days=1)] = np.nan  # beyond published data: unknown
    return out[COLUMNS]


def coverage_end(transactions: pd.DataFrame) -> pd.Timestamp:
    """Last day of the last published quarterly dataset."""
    last = transactions["filing_date"].max()
    return (last + pd.offsets.QuarterEnd(0)).normalize()
