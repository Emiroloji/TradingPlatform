import pandas as pd
import pytest

from config import load_config
from data.clean import clean_prices


@pytest.fixture
def cfg():
    return load_config()


def _bars(ticker: str, closes: list[float], volumes: list[float] | None = None, start: str = "2024-01-01") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=len(closes))
    volumes = volumes or [1000.0] * len(closes)
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": ticker,
            "open": closes,
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "close": closes,
            "adj_close": closes,
            "volume": volumes,
        }
    )


def test_clean_ticker_passes(cfg):
    df = _bars("AAA", [10.0 + i * 0.1 for i in range(50)])
    cleaned, report = clean_prices(df, cfg, "bist")
    assert report.excluded == []
    assert len(cleaned) == 50


def test_jump_is_flagged_and_excluded(cfg):
    closes = [10.0] * 20 + [5.0] * 20  # -50% overnight: split/bonus-issue artefact
    df = pd.concat([_bars("AAA", [10.0] * 40), _bars("BBB", closes)])
    cleaned, report = clean_prices(df, cfg, "bist")
    assert report.excluded == ["BBB"]
    assert set(cleaned["ticker"]) == {"AAA"}
    assert report.summary.loc["BBB", "jump_days"] == 1


def test_missing_days_measured_against_shared_calendar(cfg):
    full = _bars("AAA", [10.0] * 40)
    gappy = _bars("BBB", [10.0] * 40).iloc[::2]  # half the days missing
    _, report = clean_prices(pd.concat([full, gappy]), cfg, "bist")
    assert report.summary.loc["BBB", "missing_ratio"] == pytest.approx(0.5, abs=0.03)
    assert "BBB" in report.excluded


def test_zero_volume_rows_dropped_and_ratio_enforced(cfg):
    volumes = [0.0] * 10 + [1000.0] * 30
    cleaned, report = clean_prices(_bars("AAA", [10.0] * 40, volumes), cfg, "bist")
    assert report.summary.loc["AAA", "zero_volume_days"] == 10
    assert report.excluded == ["AAA"]


def test_invalid_bar_dropped_without_exclusion(cfg):
    df = _bars("AAA", [10.0] * 40)
    df.loc[5, "high"] = 1.0  # high < low
    cleaned, report = clean_prices(df, cfg, "bist")
    assert report.summary.loc["AAA", "invalid_rows"] == 1
    assert report.excluded == []
    assert len(cleaned) == 39


def test_raw_input_not_modified(cfg):
    df = _bars("AAA", [10.0] * 10 + [5.0] * 10)
    before = df.copy()
    clean_prices(df, cfg, "bist")
    pd.testing.assert_frame_equal(df, before)


def test_price_only_symbol_skips_volume_checks(cfg):
    fx = _bars("USDTRY=X", [30.0 + i * 0.01 for i in range(40)], volumes=[0.0] * 40)
    stock = _bars("AAA", [10.0] * 40)
    cleaned, report = clean_prices(pd.concat([stock, fx]), cfg, "bist", price_only=frozenset({"USDTRY=X"}))
    assert report.excluded == []
    assert (cleaned["ticker"] == "USDTRY=X").sum() == 40


def test_us_jump_is_flagged_but_kept(cfg):
    closes = [10.0] * 20 + [16.0] * 20  # +60% earnings gap: real in a market without price limits
    cleaned, report = clean_prices(_bars("AAA", closes), cfg, "us")
    assert report.summary.loc["AAA", "jump_days"] == 1
    assert report.excluded == [] and len(cleaned) == 40
