"""SEC EDGAR insider transactions (Form 4) from the free quarterly "Insider Transactions Data Sets".

Keeps open-market purchases (code P) and sales (code S) of common stock for our universe,
matched by issuer CIK. `filing_date` is when the market could first know about a trade.
storage/insider/<market>_transactions.parquet; downloaded zips cached in storage/cache/sec/.
"""

from __future__ import annotations

import io
import logging
import os
import time
import urllib.error
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from config import Config

logger = logging.getLogger(__name__)

COLUMNS = ["filing_date", "cik", "owner_cik", "relationship", "trans_date", "code", "shares", "price", "value"]


def quarters(start: date, end: date) -> list[str]:
    """'2019q3', ... for every calendar quarter touching [start, end]."""
    out, y, q = [], start.year, (start.month - 1) // 3 + 1
    while (y, q) <= (end.year, (end.month - 1) // 3 + 1):
        out.append(f"{y}q{q}")
        y, q = (y + 1, 1) if q == 4 else (y, q + 1)
    return out


def _user_agent(cfg: Config) -> str:
    load_dotenv(cfg.root / ".env")
    ua = os.environ.get("SEC_USER_AGENT", "").strip().strip('"')
    if "@" not in ua:
        raise RuntimeError("SEC_USER_AGENT (.env) must be 'Name email@domain' per SEC fair-access policy")
    return ua


def _download(quarter: str, cfg: Config) -> bytes | None:
    """Cached zip bytes, or None if SEC has not published that quarter yet."""
    cache = cfg.storage_path / "cache" / "sec" / f"{quarter}_form345.zip"
    if cache.exists():
        return cache.read_bytes()
    req = urllib.request.Request(cfg.insider.dataset_url.format(quarter=quarter), headers={"User-Agent": _user_agent(cfg)})
    try:
        with urllib.request.urlopen(req, timeout=cfg.insider.timeout_seconds) as resp:  # noqa: S310 (fixed sec.gov URL)
            data = resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    finally:
        time.sleep(cfg.insider.request_interval_seconds)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(data)
    return data


def parse_quarter(zip_bytes: bytes, ciks: set[int]) -> pd.DataFrame:
    """Open-market P/S transactions of `ciks` from one quarterly dataset."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        read = lambda name, cols: pd.read_csv(  # noqa: E731
            z.open(name), sep="\t", usecols=cols, dtype=str, quoting=3, on_bad_lines="skip"
        )
        sub = read("SUBMISSION.tsv", ["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERCIK"])
        trans = read(
            "NONDERIV_TRANS.tsv",
            ["ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE", "TRANS_SHARES", "TRANS_PRICEPERSHARE"],
        )
        owners = read("REPORTINGOWNER.tsv", ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP"])

    sub = sub[sub["DOCUMENT_TYPE"] == "4"].copy()  # originals only; amendments would double count
    sub["cik"] = pd.to_numeric(sub["ISSUERCIK"], errors="coerce")
    sub = sub[sub["cik"].isin(ciks)]
    trans = trans[trans["TRANS_CODE"].isin(["P", "S"])]
    owners = owners.drop_duplicates("ACCESSION_NUMBER")  # joint filings: count the filing once
    df = sub.merge(trans, on="ACCESSION_NUMBER").merge(owners, on="ACCESSION_NUMBER", how="left")
    out = pd.DataFrame(
        {
            "filing_date": pd.to_datetime(df["FILING_DATE"], format="%d-%b-%Y", errors="coerce"),
            "cik": df["cik"].astype("Int64"),
            "owner_cik": pd.to_numeric(df["RPTOWNERCIK"], errors="coerce").astype("Int64"),
            "relationship": df["RPTOWNER_RELATIONSHIP"].fillna(""),
            "trans_date": pd.to_datetime(df["TRANS_DATE"], format="%d-%b-%Y", errors="coerce"),
            "code": df["TRANS_CODE"],
            "shares": pd.to_numeric(df["TRANS_SHARES"], errors="coerce"),
            "price": pd.to_numeric(df["TRANS_PRICEPERSHARE"], errors="coerce"),
        }
    )
    out["value"] = out["shares"] * out["price"]
    return out.dropna(subset=["filing_date", "cik", "value"])[COLUMNS]


def insider_path(cfg: Config, market: str) -> Path:
    return cfg.storage_path / "insider" / f"{market}_transactions.parquet"


def _cached(quarter: str, cfg: Config) -> bool:
    return (cfg.storage_path / "cache" / "sec" / f"{quarter}_form345.zip").exists()


def update_insider(cfg: Config, market: str, ciks: set[int], start: date, end: date) -> pd.DataFrame:
    """Download (cached) every quarterly dataset in range and store the combined transactions.

    Cheap when nothing is new: only uncached quarters are requested (unpublished ones return 404),
    and the stored table is reused unless a new quarter arrived.
    """
    wanted = quarters(start, end)
    missing = [q for q in wanted if not _cached(q, cfg)]
    newly = [q for q in missing if _download(q, cfg) is not None]
    existing = read_insider(cfg, market)
    if existing is not None and not newly:
        return existing
    frames = []
    for q in wanted:
        data = _download(q, cfg)
        if data is None:
            logger.info("SEC dataset %s not published yet", q)
            continue
        frames.append(parse_quarter(data, ciks))
    table = pd.concat(frames, ignore_index=True).drop_duplicates().sort_values("filing_date", ignore_index=True)
    path = insider_path(cfg, market)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
    logger.info("Insider [%s]: %d transactions, filings %s → %s", market, len(table),
                table["filing_date"].min().date(), table["filing_date"].max().date())
    return table


def read_insider(cfg: Config, market: str) -> pd.DataFrame | None:
    path = insider_path(cfg, market)
    return pd.read_parquet(path) if path.exists() else None
