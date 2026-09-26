import io
import zipfile

import pandas as pd

from data.institutional import aggregate, parse_dataset
from features.institutional import institutional_features

DATES = pd.bdate_range("2024-01-01", "2024-12-31")


def _holdings(rows):
    return pd.DataFrame(rows, columns=["period", "filing_date", "filer_cik", "cusip", "shares"]).assign(
        period=lambda d: pd.to_datetime(d["period"]), filing_date=lambda d: pd.to_datetime(d["filing_date"])
    )


def test_aggregate_counts_only_on_time_filings(cfg):
    h = _holdings([
        ("2024-03-31", "2024-05-01", 1, "X", 100), ("2024-03-31", "2024-05-15", 2, "X", 50),
        ("2024-03-31", "2024-06-20", 3, "X", 999),  # late: after the 45-day deadline
    ])
    a = aggregate(h, cfg).iloc[0]
    assert a["filers"] == 2 and a["shares"] == 150
    assert a["available_from"] == pd.Timestamp("2024-05-16")


def _agg(cfg):
    h = _holdings([
        ("2023-12-31", "2024-02-01", 1, "X", 100), ("2023-12-31", "2024-02-01", 2, "X", 100),
        ("2024-03-31", "2024-05-01", 1, "X", 150), ("2024-03-31", "2024-05-01", 2, "X", 100),
        ("2024-03-31", "2024-05-01", 3, "X", 50),
    ])
    return aggregate(h, cfg)


def test_features_switch_on_availability_and_expire(cfg):
    out = institutional_features(DATES, _agg(cfg), cfg)
    assert pd.isna(out.loc["2024-02-14", "inst_filers"])  # Q4 aggregate available only from Feb 15
    assert out.loc["2024-02-15", "inst_filers"] == 2
    assert out.loc["2024-05-15", "inst_filers"] == 2  # Q1 not yet available
    q1 = out.loc["2024-05-16"]
    assert q1["inst_filers"] == 3 and q1["inst_filers_chg"] == 0.5 and q1["inst_shares_chg"] == 0.5
    assert out.loc["2024-08-16"].isna().all()  # Q2 was due by Aug 15 and is missing -> unknown


def test_no_lookahead(cfg):
    agg = _agg(cfg)
    full = institutional_features(DATES, agg, cfg)
    for t in DATES[::20]:
        part = institutional_features(DATES, agg[agg["available_from"] <= t], cfg)
        pd.testing.assert_series_equal(full.loc[t], part.loc[t])


import pytest  # noqa: E402


@pytest.mark.parametrize("folder", ["", "01JUN2025-31AUG2025_form13f/"])
def test_parse_dataset_filters(folder):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(folder + "SUBMISSION.tsv", "ACCESSION_NUMBER\tFILING_DATE\tSUBMISSIONTYPE\tCIK\tPERIODOFREPORT\n"
                   "A1\t01-MAY-2024\t13F-HR\t0000000001\t31-MAR-2024\nA2\t02-MAY-2024\t13F-HR/A\t0000000002\t31-MAR-2024\n")
        z.writestr(folder + "INFOTABLE.tsv", "ACCESSION_NUMBER\tCUSIP\tSSHPRNAMT\tSSHPRNAMTTYPE\tPUTCALL\n"
                   "A1\tabc123456\t100\tSH\t\nA1\tABC123456\t5\tSH\tPut\nA1\tZZZ999999\t7\tSH\t\n"
                   "A1\tABC123456\t9\tPRN\t\nA2\tABC123456\t50\tSH\t\n")
    out = parse_dataset(buf.getvalue(), {"ABC123456"})
    assert len(out) == 1 and out.iloc[0]["shares"] == 100 and out.iloc[0]["cusip"] == "ABC123456"
