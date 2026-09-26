import pandas as pd
import pytest

from config import load_config
from data import store


@pytest.fixture
def cfg(tmp_path):
    base = load_config()
    return base.model_copy(update={"data": base.data.model_copy(update={"storage_dir": str(tmp_path)})})


def _bars(symbol: str, start: str, periods: int, close: float) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=periods)
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": symbol,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "adj_close": close,
            "volume": 100.0,
        }
    )


def test_append_raw_is_incremental_and_never_rewrites(cfg):
    assert store.append_raw(_bars("AAA.IS", "2024-01-01", 5, 10.0), cfg, "bist") == 5
    # overlapping batch with different prices: existing days must keep original values
    overlap = _bars("AAA.IS", "2024-01-04", 5, 99.0)
    assert store.append_raw(overlap, cfg, "bist") == 3

    stored = store.read_prices(store.raw_dir(cfg, "bist"))
    assert len(stored) == 8
    assert stored["date"].is_unique
    assert (stored.iloc[:5]["close"] == 10.0).all()
    assert (stored.iloc[5:]["close"] == 99.0).all()


def test_last_dates(cfg):
    store.append_raw(_bars("AAA.IS", "2024-01-01", 5, 10.0), cfg, "bist")
    last = store.last_dates(store.raw_dir(cfg, "bist"), ["AAA.IS", "BBB.IS"])
    assert last == {"AAA.IS": pd.Timestamp("2024-01-05")}


def test_write_clean_removes_excluded_tickers(cfg):
    both = pd.concat([_bars("AAA.IS", "2024-01-01", 3, 1.0), _bars("BBB.IS", "2024-01-01", 3, 1.0)])
    store.write_clean(both, cfg, "bist")
    store.write_clean(both[both["ticker"] == "AAA.IS"], cfg, "bist")
    assert set(store.read_prices(store.clean_dir(cfg, "bist"))["ticker"]) == {"AAA.IS"}
