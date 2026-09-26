"""PostgreSQL + TimescaleDB mirror of the working parquet stores (FAZLAR Faz 6).

Parquet stays the fast local working layer; the database is the durable, queryable store.
Each sync replaces one market's rows per table inside a single transaction (COPY), so readers
never see a half-written market. Price rows form a TimescaleDB hypertable on `date`.
Wide tables (features, signals) gain new columns automatically when the pipeline adds some.
"""

from __future__ import annotations

import io
import logging
import os

import pandas as pd
from dotenv import load_dotenv

from config import Config

logger = logging.getLogger(__name__)

_SQL_TYPES = {"b": "BOOLEAN", "i": "DOUBLE PRECISION", "u": "DOUBLE PRECISION", "f": "DOUBLE PRECISION", "M": "TIMESTAMP"}
KEYS = ["market", "ticker", "date"]


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


def replace_market(conn, table: str, market: str, df: pd.DataFrame, cfg: Config, hypertable: bool = False) -> int:
    """Swap one market's rows in `table` for `df` (same transaction as the caller's commit)."""
    df = df.assign(market=market)
    df = df[KEYS + [c for c in df.columns if c not in KEYS]]
    df["date"] = pd.to_datetime(df["date"]).dt.date
    with conn.cursor() as cur:
        _ensure_table(cur, table, df, hypertable, cfg)
        cur.execute(f'DELETE FROM "{table}" WHERE market = %s', (market,))
        buf = io.StringIO()
        df.to_csv(buf, index=False, header=False, na_rep="\\N")
        buf.seek(0)
        cols = ", ".join(f'"{c}"' for c in df.columns)
        with cur.copy(f'COPY "{table}" ({cols}) FROM STDIN WITH (FORMAT csv, NULL \'\\N\')') as copy:
            copy.write(buf.read())
    return len(df)


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
        "prices": (store.read_prices(store.clean_dir(cfg, market)), True),
        "features": (store.read_features(cfg, market), False),
        "signals": (read_journal(cfg, market), False),
    }
    counts = {}
    with psycopg.connect(url) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")
        for table, (df, hyper) in tables.items():
            if df.empty:
                continue
            counts[table] = replace_market(conn, table, market, df, cfg, hypertable=hyper)
        conn.commit()
    logger.info("Database [%s]: %s", market, counts)
    return counts
