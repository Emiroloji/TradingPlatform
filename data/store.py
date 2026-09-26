"""Parquet storage, one file per ticker. Raw and cleaned data live in separate trees.

storage/prices/raw/<market>/<symbol>.parquet    append-only, existing days never rewritten
storage/prices/clean/<market>/<symbol>.parquet  derived, fully rewritten on every clean
storage/features/<market>.parquet               features table (one row per ticker/day)
storage/reports/                                data quality reports
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import pandas as pd

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


def read_features(cfg: Config, market: str) -> pd.DataFrame:
    return pd.read_parquet(features_path(cfg, market))


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
