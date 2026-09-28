import pandas as pd
import pytest

from data.db import database_url, replace_market


def _connect(cfg):
    psycopg = pytest.importorskip("psycopg")
    url = database_url(cfg)
    if not url:
        pytest.skip("DATABASE_URL not set")
    try:
        return psycopg.connect(url, connect_timeout=3)
    except Exception:
        pytest.skip("database not reachable")


def test_replace_market_swaps_rows_and_widens_table(cfg):
    conn = _connect(cfg)
    with conn:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS test_features")
        df = pd.DataFrame({"date": pd.to_datetime(["2024-01-02", "2024-01-03"]), "ticker": "A", "x": [1.0, 2.0]})
        assert replace_market(conn, "test_features", "bist", df, cfg) == 2
        wider = df.assign(y=[True, False])
        replace_market(conn, "test_features", "bist", wider.iloc[:1], cfg)  # new column, fewer rows
        with conn.cursor() as cur:
            cur.execute("SELECT count(*), max(x) FROM test_features WHERE market = 'bist'")
            assert cur.fetchone() == (1, 1.0)
            cur.execute("DROP TABLE test_features")


def test_replace_market_streams_chunks(cfg, monkeypatch):
    import data.db

    conn = _connect(cfg)
    monkeypatch.setattr(data.db, "COPY_CHUNK_ROWS", 2)
    with conn:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS test_chunks")
        dates = pd.bdate_range("2024-01-01", periods=5)
        df = pd.DataFrame({"date": dates, "ticker": "A", "x": range(5)}).astype({"x": float})
        assert replace_market(conn, "test_chunks", "us", df, cfg) == 5  # one frame, cut into slices of 2
        parts = [df.iloc[:3].assign(ticker="B"), df.iloc[3:].assign(ticker="B")]
        assert replace_market(conn, "test_chunks", "us", iter(parts), cfg) == 5  # an iterable of frames
        with conn.cursor() as cur:
            cur.execute("SELECT ticker, count(*), sum(x) FROM test_chunks WHERE market = 'us' GROUP BY ticker")
            assert cur.fetchall() == [("B", 5, 10.0)]
            cur.execute("DROP TABLE test_chunks")
