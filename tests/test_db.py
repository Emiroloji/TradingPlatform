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
