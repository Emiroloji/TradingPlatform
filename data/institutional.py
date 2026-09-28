"""SEC Form 13F institutional holdings (free) — the US stand-in for paid broker/custody (AKD/takas) data.

1. CUSIP ↔ ticker map from SEC fails-to-deliver files (CUSIP lists themselves are licensed).
2. Every 13F dataset (filings grouped by filing date) is streamed once, filtered to our CUSIPs
   (common shares, no options) and cached as a small parquet; the large zip is not kept.
3. Per (report period, CUSIP): number of filers and total shares, counting only original 13F-HR
   filings made by the legal deadline. That aggregate is known from `available_from`
   = period end + deadline + 1 day, so late filings can never leak into earlier dates.
4. 13F share counts are not split-adjusted: `split_factor` is the product of stock splits since
   the previous report period (yfinance), used to compare quarters on the same share basis.
"""

from __future__ import annotations

import io
import logging
import re
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from config import Config
from data.insider import _user_agent

logger = logging.getLogger(__name__)

SEC = "https://www.sec.gov"
INFOTABLE_BLOCK_BYTES = 32 * 1024 * 1024
AGGREGATE_CUSIP_GROUPS = 16


def _get(url: str, cfg: Config) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _user_agent(cfg)})
    try:
        with urllib.request.urlopen(req, timeout=cfg.insider.timeout_seconds) as resp:  # noqa: S310 (sec.gov)
            return resp.read()
    finally:
        time.sleep(cfg.insider.request_interval_seconds)


def _download(url: str, target: Path, cfg: Config) -> None:
    """Stream a (100 MB) file to disk instead of holding it in memory."""
    req = urllib.request.Request(url, headers={"User-Agent": _user_agent(cfg)})
    try:
        with urllib.request.urlopen(req, timeout=cfg.insider.timeout_seconds) as resp, open(target, "wb") as out:  # noqa: S310
            shutil.copyfileobj(resp, out, 1024 * 1024)
    finally:
        time.sleep(cfg.insider.request_interval_seconds)


def _cache(cfg: Config) -> Path:
    path = cfg.storage_path / "cache" / "sec" / "13f"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _norm(symbol: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", symbol.upper())


def cusip_map(cfg: Config, tickers: list[str]) -> pd.DataFrame:
    """ticker → cusip from the latest SEC fails-to-deliver files (most frequent CUSIP per symbol)."""
    c = cfg.institutional
    page = _get(c.ftd_listing_url, cfg).decode("utf-8", "replace")
    links = re.findall(r'href="(/files/data/fails-deliver-data/cnsfails\d{6}[ab]\.zip)"', page)[: c.ftd_files_for_cusip_map]
    frames = []
    for link in links:
        with zipfile.ZipFile(io.BytesIO(_get(SEC + link, cfg))) as z:
            frames.append(
                pd.read_csv(z.open(z.namelist()[0]), sep="|", dtype=str, usecols=["CUSIP", "SYMBOL"],
                            encoding="latin-1", on_bad_lines="skip")
            )
    ftd = pd.concat(frames).dropna()
    ftd["key"] = ftd["SYMBOL"].map(_norm)
    best = ftd.groupby(["key", "CUSIP"]).size().reset_index(name="n").sort_values("n").drop_duplicates("key", keep="last")
    wanted = pd.DataFrame({"ticker": tickers, "key": [_norm(t) for t in tickers]})
    out = wanted.merge(best[["key", "CUSIP"]], on="key", how="left").rename(columns={"CUSIP": "cusip"})
    missing = out.loc[out["cusip"].isna(), "ticker"].tolist()
    if missing:
        logger.warning("No CUSIP for %d tickers: %s", len(missing), missing)
    return out.dropna()[["ticker", "cusip"]]


def dataset_links(cfg: Config) -> list[str]:
    page = _get(cfg.institutional.listing_url, cfg).decode("utf-8", "replace")
    return sorted(set(re.findall(r'href="(/files/[^"]*form-13f-data-sets/[^"]+\.zip)"', page)))


def _our_holdings(stream, cusips: set[str]) -> pd.DataFrame:
    """INFOTABLE rows of `cusips` (common shares, no options), parsed block by block.

    The table is ~400 MB uncompressed and we keep ~3% of it; read whole it took ~1.7 GB. Blocks end
    on a line break and the parser never lets a value span lines, so this equals a single read."""
    header = stream.readline()
    frames = []
    while block := stream.read(INFOTABLE_BLOCK_BYTES):
        block += stream.readline()  # complete the block's last row
        info = pd.read_csv(io.BytesIO(header + block), sep="\t", dtype=str, engine="pyarrow",
                           usecols=["ACCESSION_NUMBER", "CUSIP", "SSHPRNAMT", "SSHPRNAMTTYPE", "PUTCALL"])
        ours = info["CUSIP"].str.upper().isin(cusips) & (info["SSHPRNAMTTYPE"] == "SH") & info["PUTCALL"].isna()
        frames.append(info.loc[ours, ["ACCESSION_NUMBER", "CUSIP", "SSHPRNAMT"]])  # the filter columns are done
    return pd.concat(frames, ignore_index=True)


def parse_dataset(source: bytes | Path, cusips: set[str]) -> pd.DataFrame:
    """Holdings of `cusips` in one 13F dataset (zip bytes or a zip file): filer, period, filing date, shares."""
    with zipfile.ZipFile(io.BytesIO(source) if isinstance(source, bytes) else source) as z:
        # some archives keep the tables in a sub-folder: find them by file name
        member = {Path(n).name: n for n in z.namelist()}
        sub = pd.read_csv(z.open(member["SUBMISSION.tsv"]), sep="\t", dtype=str,
                          usecols=["ACCESSION_NUMBER", "FILING_DATE", "SUBMISSIONTYPE", "CIK", "PERIODOFREPORT"])
        with z.open(member["INFOTABLE.tsv"]) as stream:
            info = _our_holdings(stream, cusips)
    sub = sub[sub["SUBMISSIONTYPE"] == "13F-HR"]  # originals only
    df = info.merge(sub, on="ACCESSION_NUMBER")
    return pd.DataFrame(
        {
            "period": pd.to_datetime(df["PERIODOFREPORT"], format="%d-%b-%Y", errors="coerce"),
            "filing_date": pd.to_datetime(df["FILING_DATE"], format="%d-%b-%Y", errors="coerce"),
            "filer_cik": pd.to_numeric(df["CIK"], errors="coerce"),
            "cusip": df["CUSIP"].str.upper(),
            "shares": pd.to_numeric(df["SSHPRNAMT"], errors="coerce"),
        }
    ).dropna()


def aggregate(holdings: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Per (period, cusip): filers and shares from on-time original filings, plus availability date."""
    deadline = pd.Timedelta(days=cfg.institutional.filing_deadline_days)
    on_time = holdings[holdings["filing_date"] <= holdings["period"] + deadline]
    agg = on_time.groupby(["period", "cusip"]).agg(filers=("filer_cik", "nunique"), shares=("shares", "sum")).reset_index()
    agg["available_from"] = agg["period"] + deadline + pd.Timedelta(days=1)
    return agg


def _aggregate_cached(files: list[Path], cusips: list[str], first: pd.Timestamp, cfg: Config) -> pd.DataFrame:
    """aggregate() of every cached dataset from `first` on, for `cusips`, a group of CUSIPs at a time.

    All quarters together are ~25M rows (~4.7 GB peak in one frame). Duplicates and groups never
    span CUSIPs, so aggregating CUSIP groups separately and sorting gives the same table."""
    import pyarrow.dataset as ds

    data = ds.dataset([str(f) for f in files], format="parquet")
    size = -(-len(cusips) // AGGREGATE_CUSIP_GROUPS)
    parts = []
    for start in range(0, len(cusips), size):
        group = ds.field("cusip").isin(cusips[start : start + size])
        # one file at a time: the default read-ahead decodes several ~1M-row files at once
        holdings = data.to_table(filter=group, fragment_readahead=1, batch_readahead=1, use_threads=False).to_pandas()
        holdings = holdings[holdings["period"] >= first].drop_duplicates()
        parts.append(aggregate(holdings, cfg))
    return pd.concat(parts, ignore_index=True).sort_values(["period", "cusip"], ignore_index=True)


def add_split_factors(table: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """split_factor per (ticker, period): product of split ratios dated in (previous period, period]."""
    table = table.sort_values(["ticker", "period"]).copy()
    table["prev_period"] = table.groupby("ticker")["period"].shift(1)
    factors = []
    for row in table.itertuples():
        s = splits[(splits["ticker"] == row.ticker) & (splits["date"] > row.prev_period) & (splits["date"] <= row.period)]
        factors.append(float(s["ratio"].prod()) if len(s) and pd.notna(row.prev_period) else 1.0)
    table["split_factor"] = factors
    return table.drop(columns="prev_period")


def institutional_path(cfg: Config, market: str) -> Path:
    return cfg.storage_path / "institutional" / f"{market}_holdings.parquet"


def update_institutional(cfg: Config, market: str, tickers: list[str]) -> pd.DataFrame:
    """Parse every not-yet-cached dataset, then rebuild the per-ticker quarterly aggregate."""
    cache = _cache(cfg)
    mapping_path = cache / "cusip_map.parquet"
    if mapping_path.exists():
        mapping = pd.read_parquet(mapping_path)
    else:
        mapping = cusip_map(cfg, tickers)
        mapping.to_parquet(mapping_path, index=False)
    cusips = set(mapping["cusip"])
    first = pd.Timestamp(cfg.institutional.first_period)
    path = institutional_path(cfg, market)

    new_data = False
    for link in dataset_links(cfg):
        target = cache / (Path(link).stem + ".parquet")
        if target.exists():
            continue
        m = re.search(r"(\d{4})q(\d)", link)
        if m and pd.Timestamp(int(m.group(1)), 3 * int(m.group(2)), 1) + pd.offsets.MonthEnd(0) < first:
            continue  # quarter-named file entirely before the periods we need
        logger.info("13F dataset %s", Path(link).name)
        zip_path = cache / (Path(link).stem + ".zip.tmp")
        try:
            _download(SEC + link, zip_path, cfg)
            parse_dataset(zip_path, cusips).to_parquet(target, index=False)
        finally:
            zip_path.unlink(missing_ok=True)  # only the small filtered parquet is kept
        new_data = True

    # nothing new and a recent table: reuse it (split factors are refreshed at least weekly)
    if not new_data and path.exists() and pd.Timestamp.now() - pd.Timestamp(path.stat().st_mtime, unit="s") < pd.Timedelta(days=7):
        return pd.read_parquet(path)

    table = _aggregate_cached(sorted(cache.glob("*_form13f.parquet")), sorted(cusips), first, cfg).merge(mapping, on="cusip")
    from data.fetch import fetch_splits

    splits = fetch_splits(sorted(table["ticker"].unique()), first.date(), pd.Timestamp.today().date())
    table = add_split_factors(table, splits)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
    logger.info("13F [%s]: %d ticker-quarters, periods %s → %s", market, len(table),
                table["period"].min().date(), table["period"].max().date())
    return table


def read_institutional(cfg: Config, market: str) -> pd.DataFrame | None:
    path = institutional_path(cfg, market)
    return pd.read_parquet(path) if path.exists() else None
