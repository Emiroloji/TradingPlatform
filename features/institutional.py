"""Institutional-ownership features (US, SEC 13F) per ticker/day.

On day t the latest quarter whose aggregate is available (available_from <= t) is used:
number of 13F filers holding the stock and the quarter-on-quarter change in filers and shares.
A quarter's values expire when the next quarter should have become available; if that data is
missing the features are NaN (unknown), never carried forward indefinitely.
"""

from __future__ import annotations

import pandas as pd

from config import Config

COLUMNS = ["inst_filers", "inst_filers_chg", "inst_shares_chg"]


def quarterly_changes(rows: pd.DataFrame, max_filers_change: float | None = None) -> pd.DataFrame:
    """One ticker's (period, filers, shares, available_from) with changes vs the previous quarter.

    A filer count jumping by more than `max_filers_change` marks a CUSIP break (IPO, merger,
    re-domicile: the old CUSIP's holders are missing), so that quarter's changes are NaN.
    """
    rows = rows.sort_values("period").reset_index(drop=True)
    consecutive = rows["period"].shift(1) + pd.offsets.QuarterEnd(1) == rows["period"]
    rows["inst_filers"] = rows["filers"].astype(float)
    rows["inst_filers_chg"] = (rows["filers"] / rows["filers"].shift(1) - 1).where(consecutive)
    split = rows["split_factor"] if "split_factor" in rows else 1.0  # shares on the same (post-split) basis
    rows["inst_shares_chg"] = (rows["shares"] / (rows["shares"].shift(1) * split) - 1).where(consecutive)
    if max_filers_change is not None:
        broken = rows["inst_filers_chg"].abs() > max_filers_change
        rows.loc[broken, ["inst_filers_chg", "inst_shares_chg"]] = float("nan")
    return rows


def institutional_features(dates: pd.DatetimeIndex, rows: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = pd.DataFrame(index=dates, columns=COLUMNS, dtype=float)
    if rows.empty:
        return out
    q = quarterly_changes(rows, cfg.institutional.max_filers_change)
    deadline = pd.Timedelta(days=cfg.institutional.filing_deadline_days + 1)
    q["valid_until"] = q["period"] + pd.offsets.QuarterEnd(1) + deadline  # next quarter due by then
    # parquet/pandas may carry different datetime resolutions (ms vs us); merge_asof needs one
    left = pd.DataFrame({"date": dates.astype("datetime64[ns]")})
    q["available_from"] = q["available_from"].astype("datetime64[ns]")
    joined = pd.merge_asof(left, q.sort_values("available_from"), left_on="date", right_on="available_from", direction="backward")
    fresh = joined["date"] < joined["valid_until"]
    for c in COLUMNS:
        out[c] = joined[c].where(fresh).to_numpy()
    return out
