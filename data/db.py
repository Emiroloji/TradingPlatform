"""PostgreSQL + TimescaleDB mirror of the working parquet stores (FAZLAR Faz 6).

Parquet stays the fast local working layer; the database is the durable, queryable store.
Each sync replaces one market's rows per table inside a single transaction (COPY), so readers
never see a half-written market. Price rows form a TimescaleDB hypertable on `date`.
Wide tables (features, signals) gain new columns automatically when the pipeline adds some.
"""

from __future__ import annotations

import itertools
import logging
import os
import select
from collections.abc import Iterable, Iterator

import pandas as pd
from dotenv import load_dotenv

from config import Config

logger = logging.getLogger(__name__)

_SQL_TYPES = {"b": "BOOLEAN", "i": "DOUBLE PRECISION", "u": "DOUBLE PRECISION", "f": "DOUBLE PRECISION", "M": "TIMESTAMP"}
KEYS = ["market", "ticker", "date"]
COPY_CHUNK_ROWS = 10_000


def database_url(cfg: Config) -> str | None:
    load_dotenv(cfg.root / ".env")
    return os.environ.get("DATABASE_URL") or None


def _sql_type(series: pd.Series) -> str:
    if series.name == "date":
        return "DATE"
    return _SQL_TYPES.get(series.dtype.kind, "TEXT")


def _ensure_table(cur, table: str, df: pd.DataFrame, hypertable: bool, cfg: Config) -> None:
    cols = ", ".join(f'"{c}" {_sql_type(df[c])}' for c in df.columns)
    cur.execute(f'CREATE TABLE IF NOT EXISTS "{table}" ({cols}, PRIMARY KEY (market, ticker, date))')
    if hypertable:
        cur.execute(
            "SELECT create_hypertable(%s, 'date', chunk_time_interval => %s::interval, if_not_exists => TRUE, migrate_data => TRUE)",
            (table, f"{cfg.database.hypertable_chunk_days} days"),
        )
    cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = %s", (table,))
    existing = {r[0] for r in cur.fetchall()}
    for c in df.columns:
        if c not in existing:  # the pipeline added a feature: widen the table
            cur.execute(f'ALTER TABLE "{table}" ADD COLUMN "{c}" {_sql_type(df[c])}')


def _flushing_writer(cursor):
    """COPY writer that sends each chunk before taking the next. On Linux psycopg leaves COPY data in
    libpq's output buffer, which then grows by the whole table when the server is slower than we are."""
    from psycopg.copy import LibpqWriter

    class FlushingWriter(LibpqWriter):
        def write(self, data) -> None:
            super().write(data)
            pgconn = self.connection.pgconn
            while pgconn.flush() == 1:  # 1: data still pending because the socket is full
                select.select([], [pgconn.socket], [])

    return FlushingWriter(cursor)


def _prepare(df: pd.DataFrame, market: str) -> pd.DataFrame:
    df = df.assign(market=market)
    df = df[KEYS + [c for c in df.columns if c not in KEYS]]
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def _chunks(data: pd.DataFrame | Iterable[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    """A frame is cut into COPY_CHUNK_ROWS slices (an empty frame still yields itself once)."""
    if isinstance(data, pd.DataFrame):
        for start in range(0, max(len(data), 1), COPY_CHUNK_ROWS):
            yield data.iloc[start : start + COPY_CHUNK_ROWS]
    else:
        yield from data


def replace_market(
    conn, table: str, market: str, data: pd.DataFrame | Iterable[pd.DataFrame], cfg: Config, hypertable: bool = False
) -> int:
    """Swap one market's rows in `table` for `data` (same transaction as the caller's commit).

    `data` is a frame or an iterable of frames with the same columns; rows are sent in chunks so
    memory stays bounded (the whole US features table as one CSV string took gigabytes)."""
    chunks = _chunks(data)
    first = next(chunks, None)
    if first is None:
        return 0
    first = _prepare(first, market)
    cols = ", ".join(f'"{c}"' for c in first.columns)
    rows = 0
    with conn.cursor() as cur:
        _ensure_table(cur, table, first, hypertable, cfg)
        cur.execute(f'DELETE FROM "{table}" WHERE market = %s', (market,))
        statement = f'COPY "{table}" ({cols}) FROM STDIN WITH (FORMAT csv, NULL \'\\N\')'
        with cur.copy(statement, writer=_flushing_writer(cur)) as copy:
            for df in itertools.chain([first], (_prepare(c, market) for c in chunks)):
                copy.write(df.to_csv(index=False, header=False, na_rep="\\N"))
                rows += len(df)
    return rows


def sync_market(cfg: Config, market: str) -> dict[str, int] | None:
    """Mirror clean prices, features and the signal journal of `market`. None if disabled."""
    import psycopg

    from data import store
    from signals.journal import read_journal

    url = database_url(cfg)
    if not cfg.database.enabled or not url:
        logger.info("Database sync skipped (disabled or DATABASE_URL unset)")
        return None
    tables = {
        # prices and features are streamed: neither table is ever whole in memory
        "prices": (store.iter_prices(store.clean_dir(cfg, market)), True),
        "features": (store.iter_features(cfg, market, COPY_CHUNK_ROWS), False),
        "signals": (read_journal(cfg, market), False),
    }
    counts = {}
    with psycopg.connect(url) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")
        for table, (data, hyper) in tables.items():
            if isinstance(data, pd.DataFrame) and data.empty:
                continue
            counts[table] = replace_market(conn, table, market, data, cfg, hypertable=hyper)
        conn.commit()
    logger.info("Database [%s]: %s", market, counts)
    return counts
