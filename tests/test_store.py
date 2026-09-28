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


def _features(tickers: list[str], periods: int = 4) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=periods)
    return pd.DataFrame(
        [{"date": d, "ticker": t, "total_score": float(i), "regime": "bull", "liquidity_ok": True}
         for t in tickers for i, d in enumerate(dates)]
    )


def test_features_written_in_chunks_read_back_as_one_table(cfg):
    table = _features(["AAA.IS", "BBB.IS", "CCC.IS"])
    chunks = [table[table["ticker"] == t].reset_index(drop=True) for t in ["AAA.IS", "BBB.IS", "CCC.IS"]]
    path, latest = store.write_features_chunks(iter(chunks), cfg, "bist")

    pd.testing.assert_frame_equal(store.read_features(cfg, "bist"), table)
    pd.testing.assert_frame_equal(pd.concat(store.iter_features(cfg, "bist", batch_rows=5), ignore_index=True), table)
    expected_day = table[table["date"] == table["date"].max()].reset_index(drop=True)
    pd.testing.assert_frame_equal(latest, expected_day)
    pd.testing.assert_frame_equal(store.read_features_day(cfg, "bist"), expected_day)
    pd.testing.assert_frame_equal(
        store.read_features_ticker(cfg, "bist", "BBB.IS"), table[table["ticker"] == "BBB.IS"].reset_index(drop=True)
    )
    assert not path.with_suffix(".parquet.tmp").exists()


def test_failed_chunked_write_keeps_previous_table(cfg):
    store.write_features_chunks(iter([_features(["AAA.IS"])]), cfg, "bist")

    def broken():
        yield _features(["BBB.IS"])
        raise RuntimeError("feature step failed")

    with pytest.raises(RuntimeError):
        store.write_features_chunks(broken(), cfg, "bist")
    assert set(store.read_features(cfg, "bist")["ticker"]) == {"AAA.IS"}


def test_iter_prices_covers_every_ticker_once(cfg):
    both = pd.concat([_bars(s, "2024-01-01", 3, 1.0) for s in ["AAA.IS", "BBB.IS", "CCC.IS"]])
    store.write_clean(both, cfg, "bist")
    chunks = list(store.iter_prices(store.clean_dir(cfg, "bist"), files_per_chunk=2))
    assert len(chunks) == 2
    streamed = pd.concat(chunks, ignore_index=True).sort_values(["ticker", "date"], ignore_index=True)
    pd.testing.assert_frame_equal(streamed, store.read_prices(store.clean_dir(cfg, "bist")))
