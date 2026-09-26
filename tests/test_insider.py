import numpy as np
import pandas as pd

from data.insider import parse_quarter, quarters
from features.insider import coverage_end, insider_features

DATES = pd.bdate_range("2024-01-01", "2024-06-28")


def _tx(rows):
    return pd.DataFrame(rows, columns=["filing_date", "owner_cik", "code", "value"]).assign(
        filing_date=lambda d: pd.to_datetime(d["filing_date"])
    )


def test_quarters():
    from datetime import date

    assert quarters(date(2019, 8, 1), date(2020, 2, 1)) == ["2019q3", "2019q4", "2020q1"]


def test_filing_known_only_from_next_day_and_window(cfg):
    tx = _tx([("2024-03-01", 1, "P", 100.0), ("2024-03-04", 2, "P", 50.0), ("2024-03-05", 3, "S", 30.0)])
    out = insider_features(DATES, tx, pd.Series(1000.0, index=DATES), cfg, pd.Timestamp("2024-06-30"))
    assert out.loc["2024-03-01", "insider_buy_value"] == 0  # filed that day: not yet known at the close
    assert out.loc["2024-03-04", "insider_buy_value"] == 100  # the Friday filing is known on Monday
    row = out.loc["2024-03-06"]
    assert row["insider_buy_value"] == 150 and row["insider_sell_value"] == 30 and row["insider_buyers"] == 2
    assert row["insider_cluster_buy"] == 1 and np.isclose(row["insider_net_ratio"], 120 / 180)
    assert row["insider_buy_to_turnover"] == 0.15
    late = pd.Timestamp("2024-03-01") + pd.Timedelta(days=cfg.insider.window_days + 5)
    assert out.loc[out.index >= late, "insider_buy_value"].iloc[0] == 0  # rolled out of the window


def test_beyond_coverage_is_unknown(cfg):
    tx = _tx([("2024-03-01", 1, "P", 100.0)])
    out = insider_features(DATES, tx, pd.Series(1000.0, index=DATES), cfg, pd.Timestamp("2024-03-31"))
    assert out.loc["2024-04-15"].isna().all()
    assert coverage_end(tx) == pd.Timestamp("2024-03-31")


def test_no_lookahead(cfg):
    tx = _tx([("2024-02-01", 1, "P", 10.0), ("2024-04-01", 2, "P", 20.0), ("2024-05-02", 3, "S", 5.0)])
    full = insider_features(DATES, tx, pd.Series(1.0, index=DATES), cfg, pd.Timestamp("2024-06-30"))
    for t in DATES[::15]:
        part = insider_features(DATES, tx[tx["filing_date"] < t], pd.Series(1.0, index=DATES), cfg, pd.Timestamp("2024-06-30"))
        pd.testing.assert_series_equal(full.loc[t], part.loc[t])


def test_parse_quarter_keeps_open_market_trades_of_universe():
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("SUBMISSION.tsv", "ACCESSION_NUMBER\tFILING_DATE\tDOCUMENT_TYPE\tISSUERCIK\n"
                   "A1\t05-MAR-2024\t4\t0000000111\nA2\t06-MAR-2024\t4/A\t0000000111\nA3\t07-MAR-2024\t4\t0000000999\n")
        z.writestr("NONDERIV_TRANS.tsv", "ACCESSION_NUMBER\tTRANS_DATE\tTRANS_CODE\tTRANS_SHARES\tTRANS_PRICEPERSHARE\n"
                   "A1\t01-MAR-2024\tP\t100\t10.5\nA1\t01-MAR-2024\tA\t50\t0\nA2\t01-MAR-2024\tP\t100\t10.5\nA3\t01-MAR-2024\tS\t1\t1\n")
        z.writestr("REPORTINGOWNER.tsv", "ACCESSION_NUMBER\tRPTOWNERCIK\tRPTOWNER_RELATIONSHIP\nA1\t0000000555\tDirector\n")
    out = parse_quarter(buf.getvalue(), {111})
    assert len(out) == 1  # award (A), amendment (4/A) and other issuers dropped
    r = out.iloc[0]
    assert r["code"] == "P" and r["value"] == 1050.0 and r["owner_cik"] == 555
