"""Parquet storage, one file per ticker. Raw and cleaned data live in separate trees.

storage/prices/raw/<market>/<symbol>.parquet    append-only, existing days never rewritten
storage/prices/clean/<market>/<symbol>.parquet  derived, fully rewritten on every clean
storage/features/<market>.parquet               features table (one row per ticker/day)
storage/reports/                                data quality reports
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Iterator
from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config import Config
from data.clean import QualityReport
from data.fetch import PRICE_COLUMNS

logger = logging.getLogger(__name__)


def raw_dir(cfg: Config, market: str) -> Path:
    return cfg.storage_path / "prices" / "raw" / market


def clean_dir(cfg: Config, market: str) -> Path:
    return cfg.storage_path / "prices" / "clean" / market


def features_path(cfg: Config, market: str) -> Path:
    return cfg.storage_path / "features" / f"{market}.parquet"


def write_features(table: pd.DataFrame, cfg: Config, market: str) -> Path:
    path = features_path(cfg, market)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
    return path


def write_features_chunks(chunks: Iterable[pd.DataFrame], cfg: Config, market: str) -> tuple[Path, pd.DataFrame]:
    """Write the features table chunk by chunk (one row group each) so it is never whole in memory.

    The file is written next to the target and renamed at the end: readers (dashboard) never see
    a half-written table. Returns the path and the rows of the table's latest day."""
    path = features_path(cfg, market)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    writer, schema, latest = None, None, []
    try:
        for chunk in chunks:
            if writer is None:
                schema = pa.Schema.from_pandas(chunk, preserve_index=False)
                writer = pq.ParquetWriter(tmp, schema)
            writer.write_table(pa.Table.from_pandas(chunk, schema=schema, preserve_index=False))
            latest.append(chunk[chunk["date"] == chunk["date"].max()])
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError(f"No feature rows for {market}")
    tmp.replace(path)
    last = pd.concat(latest, ignore_index=True)
    return path, last[last["date"] == last["date"].max()].reset_index(drop=True)


def read_features(cfg: Config, market: str, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(features_path(cfg, market), columns=columns)


def feature_columns(cfg: Config, market: str) -> list[str]:
    return pq.read_schema(features_path(cfg, market)).names


def iter_features(cfg: Config, market: str, batch_rows: int) -> Iterator[pd.DataFrame]:
    """The features table in row batches of at most `batch_rows` (bounded memory)."""
    for batch in pq.ParquetFile(features_path(cfg, market)).iter_batches(batch_size=batch_rows):
        yield batch.to_pandas()


def read_features_ticker(cfg: Config, market: str, ticker: str) -> pd.DataFrame:
    return pd.read_parquet(features_path(cfg, market), filters=[("ticker", "==", ticker)]).reset_index(drop=True)


def read_features_day(cfg: Config, market: str) -> pd.DataFrame:
    """Rows of the latest day only; the daily flow needs nothing else from the table."""
    path = features_path(cfg, market)
    last = pd.read_parquet(path, columns=["date"])["date"].max()
    return pd.read_parquet(path, filters=[("date", "==", last)]).reset_index(drop=True)


def setup_path(cfg: Config, market: str) -> Path:
    return cfg.storage_path / "reports" / f"setup_{market}.json"


def write_setup(summary: dict, cfg: Config, market: str) -> None:
    """Out-of-sample track record of the live setup, read by the signal filter (KURALLAR §2)."""
    path = setup_path(cfg, market)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")


def read_setup(cfg: Config, market: str) -> dict | None:
    path = setup_path(cfg, market)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def reports_dir(cfg: Config) -> Path:
    return cfg.storage_path / "reports"


def _path(directory: Path, symbol: str) -> Path:
    return directory / f"{symbol}.parquet"


def read_prices(directory: Path, symbols: list[str] | None = None) -> pd.DataFrame:
    """All stored bars in `directory` (optionally only `symbols`) in long format."""
    files = [_path(directory, s) for s in symbols] if symbols else sorted(directory.glob("*.parquet"))
    frames = [pd.read_parquet(f) for f in files if f.exists()]
    if not frames:
        return pd.DataFrame(columns=PRICE_COLUMNS)
    return pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"], ignore_index=True)


def iter_prices(directory: Path, files_per_chunk: int = 25) -> Iterator[pd.DataFrame]:
    """The bars of `directory` a few tickers at a time (bounded memory; each ticker's rows by date)."""
    files = sorted(directory.glob("*.parquet"))
    for start in range(0, len(files), files_per_chunk):
        yield pd.concat([pd.read_parquet(f) for f in files[start : start + files_per_chunk]], ignore_index=True)


def last_dates(directory: Path, symbols: list[str]) -> dict[str, pd.Timestamp]:
    """Last stored date per symbol; symbols with no file are absent."""
    out = {}
    for symbol in symbols:
        path = _path(directory, symbol)
        if path.exists():
            out[symbol] = pd.read_parquet(path, columns=["date"])["date"].max()
    return out


def append_raw(new: pd.DataFrame, cfg: Config, market: str) -> int:
    """Append only days not yet stored. Returns the number of rows added."""
    directory = raw_dir(cfg, market)
    directory.mkdir(parents=True, exist_ok=True)
    added = 0
    for symbol, bars in new.groupby("ticker"):
        path = _path(directory, str(symbol))
        if path.exists():
            existing = pd.read_parquet(path)
            bars = bars[~bars["date"].isin(existing["date"])]
            if bars.empty:
                continue
            bars = pd.concat([existing, bars], ignore_index=True)
            added_rows = len(bars) - len(existing)
        else:
            added_rows = len(bars)
        bars.sort_values("date").reset_index(drop=True).to_parquet(path, index=False)
        added += added_rows
    logger.info("Raw store [%s]: %d new rows", market, added)
    return added


def write_clean(cleaned: pd.DataFrame, cfg: Config, market: str) -> None:
    """Replace the clean tree for this market with `cleaned` (excluded tickers disappear)."""
    directory = clean_dir(cfg, market)
    directory.mkdir(parents=True, exist_ok=True)
    keep = set(cleaned["ticker"].unique())
    for stale in directory.glob("*.parquet"):
        if stale.stem not in keep:
            stale.unlink()
    for symbol, bars in cleaned.groupby("ticker"):
        bars.reset_index(drop=True).to_parquet(_path(directory, str(symbol)), index=False)
    logger.info("Clean store [%s]: %d tickers written", market, len(keep))


def write_quality_report(report: QualityReport, cfg: Config, market: str, as_of: date) -> Path:
    """Save the report as markdown (for humans) and CSV (for later analysis)."""
    directory = reports_dir(cfg)
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"data_quality_{market}_{as_of.isoformat()}"
    report.summary.to_csv(directory / f"{stem}.csv")
    md_path = directory / f"{stem}.md"
    md_path.write_text(report.to_markdown(), encoding="utf-8")
    return md_path
