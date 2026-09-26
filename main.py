"""CLI entry point. Commands are added phase by phase (MIMARI §8)."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv

from backtest.engine import add_benchmark_returns, run_backtest
from backtest.metrics import breakdown, breakdown_table, compute_metrics, metrics_table
from config import Config, load_config
from data import store
from data.clean import clean_prices
from data.fetch import fetch_prices, load_universe
from features import compute_features
from model.labels import make_labels_frame
from model.predict import load_oos_predictions
from model.train import registry_for, save_artifact, set_approval, train_walk_forward
from signals.filter import apply_conservative_filter, ml_signals, rule_based_signals, setup_ok

logger = logging.getLogger("borsa")


class RedactSecrets(logging.Filter):
    """Masks every secret from .env in log output (KURALLAR §10), whatever library logs it."""

    SECRET_KEYS = ("GEMINI_API_KEY", "TELEGRAM_BOT_TOKEN")

    def __init__(self) -> None:
        super().__init__()
        self.secrets = [v for k in self.SECRET_KEYS if len(v := os.environ.get(k, "")) >= 8]

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if any(secret in message for secret in self.secrets):
            for secret in self.secrets:
                message = message.replace(secret, "***")
            record.msg, record.args = message, None
        return True


def setup_logging(cfg: Config) -> None:
    load_dotenv(cfg.root / ".env")
    log_dir = cfg.storage_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler(), logging.FileHandler(log_dir / "borsa.log", encoding="utf-8")]
    for handler in handlers:
        handler.addFilter(RedactSecrets())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", handlers=handlers)
    # HTTP client libraries log request URLs at INFO; the Telegram Bot API URL contains the token
    for noisy in ("httpx", "httpcore", "telegram", "urllib3", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def last_complete_session(cfg: Config, market: str) -> date:
    """Today if the scheduled post-close run time has passed, otherwise yesterday.

    Prevents storing an unfinished intraday bar in the append-only raw store.
    """
    now = datetime.now(ZoneInfo(cfg.schedule.timezone))
    run_at = datetime.strptime(getattr(cfg.schedule, f"{market}_run_time"), "%H:%M").time()
    return now.date() if now.time() >= run_at else now.date() - timedelta(days=1)


def fetch_market(cfg: Config, market: str) -> None:
    mcfg = cfg.markets[market]
    universe = load_universe(mcfg, cfg)
    fx = [mcfg.fx] if mcfg.fx else []
    symbols = universe["symbol"].tolist() + [mcfg.benchmark] + fx
    end = last_complete_session(cfg, market)
    full_start = end - pd.DateOffset(years=cfg.data.history_years)

    # 1) fetch: new symbols get full history, stored ones only the missing tail
    known = store.last_dates(store.raw_dir(cfg, market), symbols)
    new_symbols = [s for s in symbols if s not in known]
    batches: list[tuple[list[str], date]] = []
    if new_symbols:
        batches.append((new_symbols, full_start.date()))
    if known:
        batches.append((list(known), (min(known.values()) + timedelta(days=1)).date()))
    for batch, start in batches:
        if start > end:
            logger.info("[%s] %d symbols already up to date", market, len(batch))
            continue
        logger.info("[%s] fetching %d symbols %s → %s", market, len(batch), start, end)
        store.append_raw(fetch_prices(batch, start, end), cfg, market)

    # 2) clean the full raw history (raw stays untouched)
    raw = store.read_prices(store.raw_dir(cfg, market), symbols)
    raw = raw[raw["date"] >= full_start]
    cleaned, report = clean_prices(raw, cfg, market, price_only=frozenset(fx))
    if market in cfg.insider.markets:
        from data.insider import update_insider

        ciks = set(load_universe(mcfg, cfg)["cik"].dropna().astype(int))
        update_insider(cfg, market, ciks, full_start.date(), end)
    if market in cfg.institutional.markets:
        from data.institutional import update_institutional

        update_institutional(cfg, market, load_universe(mcfg, cfg)["symbol"].tolist())
    for required in [mcfg.benchmark, *fx]:
        if required in report.excluded or required not in set(raw["ticker"]):
            raise RuntimeError(f"{required} failed quality checks; see report")

    # 3) store cleaned data and the quality report
    store.write_clean(cleaned, cfg, market)
    report_path = store.write_quality_report(report, cfg, market, end)
    logger.info(
        "[%s] done: %d/%d symbols clean, %d excluded, report: %s",
        market,
        cleaned["ticker"].nunique(),
        len(symbols),
        len(report.excluded),
        report_path,
    )


def cmd_fetch(cfg: Config) -> None:
    for market in cfg.enabled_markets():
        fetch_market(cfg, market)


def features_market(cfg: Config, market: str) -> None:
    mcfg = cfg.markets[market]
    prices = store.read_prices(store.clean_dir(cfg, market))
    if prices.empty:
        raise RuntimeError(f"No clean prices for {market}; run 'fetch' first")
    is_bench = prices["ticker"] == mcfg.benchmark
    is_fx = prices["ticker"] == mcfg.fx
    fx = prices[is_fx] if mcfg.fx else None
    insider = None
    if market in cfg.insider.markets:
        from data.insider import read_insider

        insider = read_insider(cfg, market)
        if insider is not None:
            # issuer-level filings apply to every share class of the company (GOOGL/GOOG, FOXA/FOX, ...)
            symbols = load_universe(mcfg, cfg)[["cik", "symbol"]].rename(columns={"symbol": "ticker"})
            insider = insider.astype({"cik": int}).merge(symbols, on="cik")
    institutional = None
    if market in cfg.institutional.markets:
        from data.institutional import read_institutional

        institutional = read_institutional(cfg, market)
    table = compute_features(prices[~is_bench & ~is_fx], prices[is_bench], cfg, market, fx, insider, institutional)
    path = store.write_features(table, cfg, market)

    latest = table[table["date"] == table["date"].max()]
    logger.info(
        "[%s] features: %d rows, %d tickers, %s → %s; latest day %s: regime=%s, %d liquid, %d total_score>=%s",
        market,
        len(table),
        table["ticker"].nunique(),
        table["date"].min().date(),
        table["date"].max().date(),
        latest["date"].max().date(),
        latest["regime"].iloc[0],
        int(latest["liquidity_ok"].sum()),
        int((latest["total_score"] >= cfg.signal.min_total_score).sum()),
        cfg.signal.min_total_score,
    )
    logger.info("[%s] saved %s", market, path)


def cmd_features(cfg: Config) -> None:
    for market in cfg.enabled_markets():
        features_market(cfg, market)


def _simulate(
    signals: pd.DataFrame, prices: pd.DataFrame, bench: pd.DataFrame, cfg: Config, start, end, rank_by: str = "total_score"
) -> dict:
    """Backtest one period with a fresh portfolio; returns trades, metrics and breakdowns."""
    sig = signals[(signals["date"] >= start) & (signals["date"] <= end)]
    px = prices[(prices["date"] >= start) & (prices["date"] <= end)]
    result = run_backtest(sig, px, cfg, rank_by)
    trades = add_benchmark_returns(result.trades, bench)
    out = {"trades": trades, "metrics": compute_metrics(trades, bench, result.equity, cfg), "signals": len(sig)}
    if not trades.empty:
        out["by_year"] = breakdown(trades, bench, trades["signal_date"].dt.year, cfg)
        out["by_regime"] = breakdown(trades, bench, trades["regime"], cfg)
    return out


def _log_trial(cfg: Config, name: str, oos: dict) -> int:
    """Record every backtest configuration tried (KURALLAR §4); returns distinct configs so far."""
    settings = {k: getattr(cfg, k).model_dump() for k in ["features", "scoring", "signal", "backtest", "risk", "liquidity"]}
    digest = hashlib.sha256(json.dumps([name, settings], sort_keys=True, default=str).encode()).hexdigest()[:12]
    path = store.reports_dir(cfg) / "backtest_trials.csv"
    row = pd.DataFrame(
        [{"run_at": datetime.now().isoformat(timespec="seconds"), "name": name, "config_hash": digest,
          "oos_trades": oos.get("trades"), "oos_expectancy": oos.get("expectancy"),
          "oos_expectancy_excess": oos.get("expectancy_excess")}]
    )
    log = pd.concat([pd.read_csv(path), row]) if path.exists() else row
    log.to_csv(path, index=False)
    return log["config_hash"].nunique()


def backtest_market(cfg: Config, market: str) -> None:
    mcfg = cfg.markets[market]
    features = store.read_features(cfg, market)
    prices = store.read_prices(store.clean_dir(cfg, market))
    bench = prices[prices["ticker"] == mcfg.benchmark]
    sectors = load_universe(mcfg, cfg).set_index("symbol")["sector"]
    features["sector"] = features["ticker"].map(sectors)

    oos_start = pd.Timestamp(cfg.backtest.oos_start)
    first, last = features["date"].min(), features["date"].max()
    periods = {"OOS": (oos_start, last), "IS": (first, oos_start - pd.Timedelta(days=1))}

    no_regime = cfg.model_copy(
        update={"signal": cfg.signal.model_copy(update={"allowed_regimes": ["bull", "bear", "sideways"]})}
    )
    variants = {"baseline": rule_based_signals(features, cfg), "tanı: rejim filtresi yok": rule_based_signals(features, no_regime)}
    runs = {(v, p): _simulate(sig, prices, bench, cfg, *periods[p]) for v, sig in variants.items() for p in periods}
    tried = [_log_trial(cfg, v, runs[(v, "OOS")]["metrics"]) for v in variants][-1]

    as_of = last.date()
    reports = store.reports_dir(cfg)
    runs[("baseline", "OOS")]["trades"].to_csv(reports / f"backtest_baseline_trades_{market}_{as_of}.csv", index=False)
    md = _baseline_report(cfg, market, runs, periods, tried, features)
    path = reports / f"backtest_baseline_{market}_{as_of}.md"
    path.write_text(md, encoding="utf-8")
    m = runs[("baseline", "OOS")]["metrics"]
    store.write_setup(
        {k: m.get(k) for k in ["trades", "win_rate", "avg_win", "avg_loss", "expectancy", "expectancy_excess", "max_drawdown"]}
        | {"period": f"{periods['OOS'][0].date()} → {periods['OOS'][1].date()}", "report": str(path), "setup": "baseline"},
        cfg,
        market,
    )
    logger.info(
        "[%s] baseline OOS: %s trades, expectancy %s, excess %s, max DD %s -> %s",
        market, m.get("trades"), m.get("expectancy"), m.get("expectancy_excess"), m.get("max_drawdown"), path,
    )


def _baseline_report(cfg: Config, market: str, runs: dict, periods: dict, tried: int, features: pd.DataFrame) -> str:
    s, bt, r = cfg.signal, cfg.backtest, cfg.risk
    fmt = lambda p: f"{periods[p][0].date()} → {periods[p][1].date()}"  # noqa: E731
    lines = [
        f"# Baseline Backtest Raporu — {market.upper()} (kural bazlı, ML yok)",
        "",
        f"- Rapor tarihi: {datetime.now().date()} | Veri: {features['date'].min().date()} → {features['date'].max().date()}, "
        f"{features['ticker'].nunique()} hisse",
        f"- Sinyal kuralları: likidite ≥ {cfg.liquidity.min_avg_turnover_usd[market]:,.0f} USD, rejim ∈ {s.allowed_regimes}, "
        f"toplam skor ≥ {s.min_total_score:.0f}, toplama skoru ≥ {s.min_accumulation_score:.0f}",
        f"- İşlem: sinyal t kapanış → t+1 açılış; komisyon %{bt.commission * 100:.2f} + slipaj %{bt.slippage * 100:.2f} (tek yön); "
        f"stop giriş − {r.atr_stop_multiplier}×ATR, hedef {bt.target_r_multiple}R, en fazla {bt.max_holding_days} gün",
        f"- Portföy: sermaye {r.capital:,.0f}, işlem başı risk %{r.risk_per_trade * 100:.0f}, en fazla {r.max_positions} pozisyon, "
        f"sektör başına {r.max_per_sector}; kaldıraç yok",
        f"- Out-of-sample: {fmt('OOS')} | In-sample: {fmt('IS')}",
        f"- Denenen farklı ayar sayısı (tüm geçmiş): {tried} — çok deneme, şans eseri iyi sonuç riskini artırır (KURALLAR §4).",
        "",
        "## 1. Ana sonuç — Out-of-sample",
        "",
        metrics_table({"Baseline OOS": runs[("baseline", "OOS")]["metrics"],
                       "Tanı (rejimsiz) OOS": runs[("tanı: rejim filtresi yok", "OOS")]["metrics"]}),
        "",
        "## 2. In-sample (bilgi amaçlı)",
        "",
        metrics_table({"Baseline IS": runs[("baseline", "IS")]["metrics"],
                       "Tanı (rejimsiz) IS": runs[("tanı: rejim filtresi yok", "IS")]["metrics"]}),
        "",
        "## 3. Yıllara göre",
        "",
    ]
    for v in ["baseline", "tanı: rejim filtresi yok"]:
        for p in ["OOS", "IS"]:
            if "by_year" in runs[(v, p)]:
                lines.append(breakdown_table(runs[(v, p)]["by_year"], f"{v} — {p}"))
    lines += ["## 4. Piyasa rejimlerine göre (sinyal günündeki USD bazlı XU100 rejimi)", ""]
    for p in ["OOS", "IS"]:
        if "by_regime" in runs[("tanı: rejim filtresi yok", p)]:
            lines.append(breakdown_table(runs[("tanı: rejim filtresi yok", p)]["by_regime"], f"tanı (rejimsiz) — {p}"))
    lines += ["## 5. Çıkış nedenleri", ""]
    for key, run in runs.items():
        lines.append(f"- {key[0]} {key[1]}: {run['metrics'].get('exit_reasons', {})}")
    lines += [
        "",
        "## 6. Kısıtlar",
        "",
        "- Survivorship bias: geçmiş endeks üyeliği verisi yok; evren bugünkü likit hisselerden oluşur → sonuçlar olduğundan iyi görünebilir.",
        "- Evren resmi BIST 100 değil, hacme dayalı vekil listedir; bedelsiz/bölünme hatalı 12 hisse dışlanmıştır.",
        "- Temel veri ve haber/duyarlılık skoru henüz yok (toplam skor: trend, momentum, hacim, volatilite).",
        "- Getiriler geçmiş veriye dayalı olasılıklardır; garanti değildir.",
        "",
        "_Bu rapor yatırım tavsiyesi değildir._",
    ]
    return "\n".join(lines) + "\n"


def train_market(cfg: Config, market: str) -> None:
    mcfg = cfg.markets[market]
    features = store.read_features(cfg, market)
    prices = store.read_prices(store.clean_dir(cfg, market))
    bench = prices[prices["ticker"] == mcfg.benchmark]
    stocks = prices[prices["ticker"].isin(features["ticker"].unique())]
    labels = make_labels_frame(stocks, bench, cfg)
    art = train_walk_forward(features, labels, cfg)
    path = save_artifact(art, registry_for(market))
    logger.info("[%s] model %s saved to %s", market, art.version, path)
    logger.info("[%s] folds:\n%s", market, art.folds.to_string(index=False))
    logger.info("[%s] top features:\n%s", market, art.importance.head(10).round(3).to_string())


def drift_market(cfg: Config, market: str) -> list[str]:
    """Feature PSI (recent vs the live model's training window) and OOS AUC trend; returns alerts."""
    from model.drift import drift_report, feature_drift
    from model.predict import load_model
    from report.telegram_bot import send_alert

    model = load_model(market)
    root = registry_for(market) / model.version
    features = store.read_features(cfg, market)
    features = features[features["liquidity_ok"].astype(bool)]
    X = features.assign(regime_code=features["regime"].map({"bear": -1, "sideways": 0, "bull": 1}))
    last = X["date"].max()
    recent_start = sorted(X["date"].unique())[-cfg.model.drift.recent_days]
    train = X[(X["date"] >= last - pd.DateOffset(years=cfg.model.train_years)) & (X["date"] < recent_start)]
    recent = X[X["date"] >= recent_start]
    psi_values = feature_drift(train, recent, model.meta["features"], cfg)

    folds = pd.read_csv(root / "folds.csv").dropna(subset=["auc"])
    auc_recent, auc_oos = float(folds["auc"].iloc[-1]), float(folds["auc"].iloc[:-1].mean())
    importance = pd.read_csv(root / "feature_importance.csv", index_col=0)["gain_share"]
    body, alerts = drift_report(psi_values, importance, auc_recent, auc_oos, cfg)
    header = (
        f"# Model Drift Raporu — {market.upper()} (model {model.version})\n\n"
        f"- Eğitim penceresi: {train['date'].min().date()} → {train['date'].max().date()} | "
        f"son dönem: {recent['date'].min().date()} → {last.date()} ({cfg.model.drift.recent_days} işlem günü)\n"
        f"- Son AUC = son walk-forward test dönemi ({folds['test_start'].iloc[-1]}); OOS ortalaması = önceki dönemler\n\n"
    )
    path = store.reports_dir(cfg) / f"drift_{market}_{last.date()}.md"
    path.write_text(header + body, encoding="utf-8")
    if alerts:
        send_alert(f"{market.upper()} model kayması: " + "; ".join(alerts), cfg)
    logger.info("[%s] drift: %d alerts -> %s", market, len(alerts), path)
    return alerts


def retrain_market(cfg: Config, market: str) -> None:
    """Monthly: retrain, re-run the live-approval backtest (KURALLAR §5), drift report, Telegram note."""
    from model.predict import load_model
    from report.telegram_bot import send

    train_market(cfg, market)
    ml_backtest_market(cfg, market)
    alerts = drift_market(cfg, market)
    model = load_model(market)
    status = "canlıya ONAYLI" if model.meta.get("approved_for_live") else "baseline'ı geçemedi, kural bazlı sistem sürüyor"
    send(f"{market.upper()} aylık yeniden eğitim: model {model.version} — {status}. Drift uyarısı: {len(alerts)}.", cfg)


def cmd_drift(cfg: Config) -> None:
    for market in cfg.enabled_markets():
        drift_market(cfg, market)


def cmd_retrain(cfg: Config) -> None:
    for market in cfg.enabled_markets():
        retrain_market(cfg, market)


def cmd_train(cfg: Config) -> None:
    for market in cfg.enabled_markets():
        train_market(cfg, market)


def _fold_comparison(runs: dict, folds: pd.DataFrame) -> pd.DataFrame:
    """Excess expectancy per walk-forward test window for every run."""
    rows = []
    for _, f in folds.iterrows():
        start, end = pd.Timestamp(f["test_start"]), pd.Timestamp(f["test_end"])
        row = {"fold": f["fold"], "test": f"{f['test_start']} → {f['test_end']}"}
        for name, run in runs.items():
            t = run["trades"]
            in_fold = t[(t["signal_date"] >= start) & (t["signal_date"] < end)] if not t.empty else t
            row[f"{name}_trades"] = len(in_fold)
            row[f"{name}_excess"] = in_fold["excess_return"].mean() if len(in_fold) else float("nan")
        rows.append(row)
    table = pd.DataFrame(rows)
    # compared only where at least one side traded; ML without trades counts as not better
    table["comparable"] = (table["ml_trades"] > 0) | (table["baseline_trades"] > 0)
    table["ml_better"] = (table["ml_excess"] > table["baseline_excess"]) | (
        (table["ml_trades"] > 0) & (table["baseline_trades"] == 0) & (table["ml_excess"] > 0)
    )
    return table


def ml_backtest_market(cfg: Config, market: str) -> None:
    mcfg = cfg.markets[market]
    preds = load_oos_predictions(market)
    version = preds["model_version"].iloc[0]
    folds = pd.read_csv(registry_for(market) / version / "folds.csv")
    features = store.read_features(cfg, market).merge(preds[["date", "ticker", "prob"]], on=["date", "ticker"], how="inner")
    prices = store.read_prices(store.clean_dir(cfg, market))
    bench = prices[prices["ticker"] == mcfg.benchmark]
    features["sector"] = features["ticker"].map(load_universe(mcfg, cfg).set_index("symbol")["sector"])

    start, end = pd.Timestamp(cfg.model.first_test_start), features["date"].max()
    s = cfg.signal
    ml_only = features[
        features["liquidity_ok"].astype(bool) & features["regime"].isin(s.allowed_regimes) & (features["prob"] >= s.min_probability)
    ]
    runs = {
        "baseline": _simulate(rule_based_signals(features, cfg), prices, bench, cfg, start, end),
        "ml": _simulate(ml_signals(features, cfg), prices, bench, cfg, start, end, rank_by="prob"),
        "ml_only": _simulate(ml_only, prices, bench, cfg, start, end, rank_by="prob"),
    }
    tried = [_log_trial(cfg, f"ml:{name}:{version}", runs[name]["metrics"]) for name in ["ml", "ml_only"]][-1]

    table = _fold_comparison(runs, folds)
    base_m, ml_m = runs["baseline"]["metrics"], runs["ml"]["metrics"]
    win_ratio = float(table.loc[table["comparable"], "ml_better"].mean())
    checks = {
        "ml_trades_enough": ml_m.get("trades", 0) >= s.min_backtest_trades,
        "overall_better": ml_m.get("expectancy_excess", float("-inf")) > base_m.get("expectancy_excess", float("inf")),
        "fold_win_ratio_ok": win_ratio >= cfg.model.min_fold_win_ratio,
    }
    approved = all(checks.values())
    set_approval(version, approved, root=registry_for(market), evidence={"checks": checks, "fold_win_ratio": win_ratio,
                                     "ml_expectancy_excess": ml_m.get("expectancy_excess"),
                                     "baseline_expectancy_excess": base_m.get("expectancy_excess")})

    as_of = end.date()
    reports = store.reports_dir(cfg)
    runs["ml"]["trades"].to_csv(reports / f"backtest_ml_trades_{market}_{as_of}.csv", index=False)
    path = reports / f"backtest_ml_{market}_{as_of}.md"
    report = _ml_report(cfg, market, version, runs, table, checks, approved, win_ratio, tried, start, end)
    path.write_text(report, encoding="utf-8")
    logger.info("[%s] ML %s vs baseline: approved=%s checks=%s -> %s", market, version, approved, checks, path)


def _ml_report(cfg, market, version, runs, table, checks, approved, win_ratio, tried, start, end) -> str:
    s = cfg.signal
    fmt = lambda v: "—" if pd.isna(v) else f"{v:+.2%}"  # noqa: E731
    imp = pd.read_csv(registry_for(market) / version / "feature_importance.csv", index_col=0)
    verdict = (
        "ML canlıya alınabilir."
        if approved
        else "ML baseline'ı tutarlı şekilde geçmiyor → canlıya alınmaz; kural bazlı sistemle devam edilir."
    )
    lines = [
        f"# ML Backtest Raporu — {market.upper()} (model {version})",
        "",
        f"- Out-of-sample dönem: {start.date()} → {end.date()} (walk-forward test dönemleri; her tahmin o dönemi görmemiş modelden)",
        f"- ML sinyali = kural bazlı sinyal ∧ ML olasılığı ≥ {s.min_probability} (KURALLAR §2); adaylar olasılığa göre sıralanır",
        "- Tanı (yalnız ML) = likidite ∧ izinli rejim ∧ olasılık ≥ eşik (skor eşikleri yok)",
        "- Maliyetler, stop, hedef, süre ve portföy kuralları baseline ile aynı",
        f"- Denenen farklı ayar sayısı (tüm geçmiş): {tried} (KURALLAR §4)",
        "",
        "## 1. Karşılaştırma (out-of-sample)",
        "",
        metrics_table({"Baseline": runs["baseline"]["metrics"], "ML sinyali": runs["ml"]["metrics"],
                       "Tanı: yalnız ML": runs["ml_only"]["metrics"]}),
        "",
        "## 2. Walk-forward test dönemlerine göre (endekse göre expectancy / işlem)",
        "",
        "| Dönem | Test | Baseline (işlem) | ML (işlem) | Yalnız ML (işlem) | ML daha iyi |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in table.iterrows():
        lines.append(
            f"| {r['fold']} | {r['test']} | {fmt(r['baseline_excess'])} ({r['baseline_trades']}) "
            f"| {fmt(r['ml_excess'])} ({r['ml_trades']}) | {fmt(r['ml_only_excess'])} ({r['ml_only_trades']}) "
            f"| {('evet' if r['ml_better'] else 'hayır') if r['comparable'] else 'kıyas yok'} |"
        )
    lines += [
        "",
        "## 3. Karar (KURALLAR §5)",
        "",
        f"- ML en az {s.min_backtest_trades} işlem: {'✔' if checks['ml_trades_enough'] else '✘'}",
        f"- OOS toplamında endekse göre expectancy baseline'dan iyi: {'✔' if checks['overall_better'] else '✘'}",
        f"- Test dönemlerinin ≥ %{cfg.model.min_fold_win_ratio * 100:.0f}'ında daha iyi: "
        f"{'✔' if checks['fold_win_ratio_ok'] else '✘'} (%{win_ratio * 100:.0f})",
        "",
        f"**Sonuç: {verdict}**",
        "",
        "## 4. Özellik önemi (ortalama kazanç payı, ilk 15)",
        "",
        "| Özellik | Pay |",
        "|---|---|",
        *[f"| {k} | {v:.3f} |" for k, v in imp["gain_share"].head(15).items()],
        "",
        "## 5. Kısıtlar",
        "",
        "- Survivorship bias ve vekil evren (bkz. baseline raporu). Temel veri ve haber duyarlılığı yok.",
        "- Olasılıklar geçmiş veriye dayalıdır; garanti değildir.",
        "",
        "_Bu rapor yatırım tavsiyesi değildir._",
    ]
    return "\n".join(lines) + "\n"


LIMITATIONS = [
    "Survivorship bias: geçmiş endeks üyeliği verisi yok; evren bugünkü likit hisselerden oluşan vekil listedir.",
    "KAP bildirimleri doğrudan okunmuyor (resmi açık API yok, ücretli lisans yok); yalnızca haberlere yansıdığı kadar.",
    "Temel (bilanço) verisi yok; skor teknik göstergelerden oluşur.",
]


def _ml_state(market: str) -> tuple[bool, str, object]:
    """(approved, note, model) for the market's latest registry model (KURALLAR §5)."""
    from model.predict import load_model

    try:
        model = load_model(market)
    except FileNotFoundError:
        return False, "model yok; kural bazlı sistem", None
    if model.meta.get("approved_for_live"):
        return True, f"ML modeli {model.version} (canlıya onaylı)", model
    return False, f"ML modeli {model.version} baseline'ı geçemedi → kullanılmıyor, kural bazlı sistem", None


def run_market(cfg: Config, market: str) -> Path:
    """MIMARI §3 daily flow for one market; returns the daily report path."""
    from ai.gemini_client import GeminiClient
    from ai.report_writer import SignalRow, write_commentary
    from ai.sentiment import check_candidates
    from model.predict import predict
    from report.telegram_bot import DailyContext, build_daily_message, send, send_alert
    from signals.journal import live_performance, read_journal, record_signals, update_outcomes
    from signals.risk import size_positions

    mcfg = cfg.markets[market]
    step = "veri çekme"
    try:
        # [1] data  [2] features
        fetch_market(cfg, market)
        step = "özellikler"
        features_market(cfg, market)
        features = store.read_features(cfg, market)
        prices = store.read_prices(store.clean_dir(cfg, market))
        bench = prices[prices["ticker"] == mcfg.benchmark]
        last_day = features["date"].max()
        expected = pd.Timestamp(last_complete_session(cfg, market))
        stale = last_day < expected and expected.dayofweek < 5
        if stale:
            send_alert(f"{market.upper()}: {expected.date()} verisi gelmedi (son veri {last_day.date()}); tatil olabilir.", cfg)

        # outcomes of earlier signals are updated every run
        step = "sinyal günlüğü"
        update_outcomes(cfg, market, prices, bench)

        # [3]-[5] filters on the latest day (never re-issued for a stale day)
        step = "filtre"
        today = features[features["date"] == last_day].copy()
        universe = load_universe(mcfg, cfg).set_index("symbol")
        today["sector"] = today["ticker"].map(universe["sector"])
        ml_approved, ml_note, model = _ml_state(market)
        today["prob"] = predict(model, today)["prob"].to_numpy() if ml_approved else float("nan")
        setup = store.read_setup(cfg, market)
        candidates, funnel = apply_conservative_filter(today, cfg, setup, ml_approved)
        if stale:
            candidates = candidates.iloc[0:0]
            funnel.append(("yeni işlem günü yok", 0))

        # [6] Gemini only for candidates
        step = "Gemini"
        llm = GeminiClient(cfg)
        vetoed, blocked, signal_rows = [], [], []
        if not candidates.empty:
            now = datetime.now(ZoneInfo(cfg.schedule.timezone))
            checked = check_candidates(candidates, universe["name"].to_dict(), llm, cfg, now, mcfg.news_locale, mcfg.subreddits)
            vetoed = [(r.ticker, r.veto_reason) for r in checked.itertuples() if r.gemini_veto]
            blocked = [(r.ticker, r.news_status) for r in checked.itertuples() if not r.gemini_passes and not r.gemini_veto]
            passed = checked[checked["gemini_passes"]]
            funnel.append(("Gemini veto / haber kontrolü", len(passed)))

            # [7] risk  [8] commentary
            step = "risk ve yorum"
            close = prices[prices["date"] == last_day].set_index("ticker")["close"]
            sized = size_positions(passed.assign(entry_ref=passed["ticker"].map(close)), cfg)
            funnel.append(("pozisyon / sektör limiti", len(sized)))
            for r in sized.itertuples():
                row = SignalRow(
                    ticker=r.ticker.split(".")[0], name=universe.at[r.ticker, "name"], total_score=r.total_score,
                    accumulation_score=r.accumulation_score, prob=None if not ml_approved else r.prob, regime=r.regime,
                    news_status=r.news_status, news_summary=r.news_summary or "", entry_ref=r.entry_ref, stop=r.stop,
                    position_size=int(r.position_size), bt_trades=int(setup["trades"]), bt_win_rate=setup["win_rate"],
                    bt_expectancy=setup["expectancy_excess"], bt_max_dd=setup["max_drawdown"],
                )
                signal_rows.append((r, row, write_commentary(row, llm, cfg)))

        # [9] journal + report + Telegram
        step = "rapor"
        record_signals(
            pd.DataFrame(
                [
                    {"date": r.date, "ticker": r.ticker, "total_score": r.total_score, "accumulation_score": r.accumulation_score,
                     "prob": row.prob, "regime": r.regime, "gemini_sentiment": r.gemini_sentiment, "gemini_veto": r.gemini_veto,
                     "veto_reason": r.veto_reason, "entry_ref": row.entry_ref, "stop": row.stop, "position_size": row.position_size,
                     "bt_trades": row.bt_trades, "bt_win_rate": row.bt_win_rate, "bt_expectancy": row.bt_expectancy,
                     "bt_max_dd": row.bt_max_dd, "commentary": text, "ml_note": ml_note}
                    for r, row, text in signal_rows
                ]
            ),
            cfg,
            market,
        )
        ctx = DailyContext(
            as_of=last_day.date(), market=market, regime=today["regime"].iloc[0] if len(today) else None,
            funnel=funnel, setup=setup, setup_note=setup_ok(setup, cfg)[1], ml_note=ml_note,
            signals=[(row, text) for _, row, text in signal_rows], vetoed=vetoed, news_blocked=blocked,
            live=live_performance(read_journal(cfg, market)), limitations=LIMITATIONS,
        )
        text = build_daily_message(ctx)
        path = store.reports_dir(cfg) / f"daily_{market}_{last_day.date()}.md"
        path.write_text(text + "\n", encoding="utf-8")
        sent = send(text, cfg)
        logger.info("[%s] daily run done: %d signals, report %s, telegram %s", market, len(signal_rows), path, sent)
        return path
    except Exception as exc:
        send_alert(f"{market.upper()} günlük akış '{step}' adımında durdu: {type(exc).__name__}: {str(exc)[:200]}", cfg)
        raise


def cmd_run(cfg: Config) -> None:
    for market in cfg.enabled_markets():
        run_market(cfg, market)


def build_scheduler(cfg: Config):
    """Daily run per market on weekdays after the close, monthly retrain per market (APScheduler)."""
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    scheduler = BlockingScheduler(timezone=cfg.schedule.timezone)
    for market in cfg.enabled_markets():
        hour, minute = map(int, getattr(cfg.schedule, f"{market}_run_time").split(":"))
        trigger = CronTrigger(day_of_week="mon-fri", hour=hour, minute=minute, timezone=cfg.schedule.timezone)

        def job(m: str = market) -> None:
            try:
                run_market(load_config(), m)  # re-read config so edits apply without a restart
            except Exception:
                logger.exception("Scheduled run for %s failed", m)  # already alerted; keep the scheduler alive

        scheduler.add_job(job, trigger, id=f"daily_{market}", misfire_grace_time=3600, coalesce=True)
        logger.info("Scheduled %s: weekdays %02d:%02d %s", market, hour, minute, cfg.schedule.timezone)

        r_hour, r_minute = map(int, cfg.schedule.retrain_time.split(":"))
        monthly = CronTrigger(day=cfg.schedule.retrain_day_of_month, hour=r_hour, minute=r_minute, timezone=cfg.schedule.timezone)

        def retrain_job(m: str = market) -> None:
            try:
                retrain_market(load_config(), m)
            except Exception as exc:
                logger.exception("Monthly retrain for %s failed", m)
                from report.telegram_bot import send_alert

                send_alert(f"{m.upper()} aylık yeniden eğitim başarısız: {type(exc).__name__}", load_config())

        scheduler.add_job(retrain_job, monthly, id=f"retrain_{market}", misfire_grace_time=6 * 3600, coalesce=True)
        logger.info("Scheduled %s retrain: day %d of each month %s", market, cfg.schedule.retrain_day_of_month, cfg.schedule.retrain_time)
    return scheduler


def cmd_schedule(cfg: Config) -> None:
    build_scheduler(cfg).start()


def cmd_backtest(cfg: Config, ml: bool = False) -> None:
    for market in cfg.enabled_markets():
        if ml:
            ml_backtest_market(cfg, market)
        else:
            backtest_market(cfg, market)


COMMANDS = {
    "fetch": cmd_fetch,
    "features": cmd_features,
    "train": cmd_train,
    "backtest": cmd_backtest,
    "run": cmd_run,
    "schedule": cmd_schedule,
    "drift": cmd_drift,
    "retrain": cmd_retrain,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="borsa", description="Temkinli hisse tarama ve sinyal sistemi")
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("--ml", action="store_true", help="backtest: compare the latest ML model with the baseline")
    args = parser.parse_args(argv)

    cfg = load_config()
    setup_logging(cfg)
    try:
        if args.command == "backtest":
            cmd_backtest(cfg, ml=args.ml)
        else:
            COMMANDS[args.command](cfg)
    except Exception:
        logger.exception("Command '%s' failed", args.command)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
