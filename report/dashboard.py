"""Streamlit dashboard (read-only view of storage/): `streamlit run report/dashboard.py`.

Tabs: signals (daily report, journal, live performance), stock detail chart, backtest reports,
model performance. Nothing here computes signals; it only displays what the daily run produced.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root, when run by streamlit

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from config import load_config  # noqa: E402
from data import store  # noqa: E402
from model.train import registry_for  # noqa: E402
from signals.journal import live_performance, read_journal  # noqa: E402

DISCLAIMER = "Bu panel bir karar destek aracıdır; gösterilen değerler geçmiş veriye dayalı olasılıklardır, yatırım tavsiyesi değildir."


def _latest(pattern: str, cfg) -> Path | None:
    files = sorted(store.reports_dir(cfg).glob(pattern))
    return files[-1] if files else None


def signals_tab(cfg, market: str) -> None:
    report = _latest(f"daily_{market}_*.md", cfg)
    if report:
        st.subheader(f"Son günlük rapor — {report.stem.split('_')[-1]}")
        st.text(report.read_text(encoding="utf-8"))
    else:
        st.info("Henüz günlük rapor yok: `python main.py run`")

    journal = read_journal(cfg, market)
    st.subheader("Sinyal günlüğü")
    if journal.empty:
        st.write("Henüz verilmiş sinyal yok.")
        return
    perf = live_performance(journal)
    cols = st.columns(4)
    cols[0].metric("Sinyal", perf["signals"])
    cols[1].metric("Kapanan", perf["closed"])
    if perf["closed"]:
        cols[2].metric("Kazanma oranı", f"%{perf['win_rate'] * 100:.1f}")
        cols[3].metric("Endekse göre expectancy", f"{perf['expectancy_excess'] * 100:+.2f}%")
    show = ["date", "ticker", "total_score", "accumulation_score", "regime", "entry_ref", "stop", "position_size",
            "outcome_return", "outcome_relative", "outcome_exit_date"]
    st.dataframe(journal[[c for c in show if c in journal.columns]].sort_values("date", ascending=False), hide_index=True)


def stock_tab(cfg, market: str) -> None:
    features = store.read_features(cfg, market)
    tickers = sorted(features["ticker"].unique())
    ticker = st.selectbox("Hisse", tickers)
    years = st.slider("Dönem (yıl)", 1, cfg.data.history_years, 1)
    prices = store.read_prices(store.clean_dir(cfg, market), [ticker])
    start = prices["date"].max() - pd.DateOffset(years=years)
    px = prices[prices["date"] >= start]
    ft = features[(features["ticker"] == ticker) & (features["date"] >= start)]

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.55, 0.2, 0.25], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=px["date"], open=px["open"], high=px["high"], low=px["low"], close=px["close"], name=ticker), 1, 1)
    for n in cfg.features.trend.ema_periods:
        fig.add_trace(go.Scatter(x=ft["date"], y=ft[f"ema_{n}"], name=f"EMA {n}", line={"width": 1}), 1, 1)
    journal = read_journal(cfg, market)
    if not journal.empty:
        mine = journal[(journal["ticker"] == ticker) & (journal["date"] >= start)]
        fig.add_trace(go.Scatter(x=mine["date"], y=mine["stop"], mode="markers", name="sinyal stop", marker={"symbol": "triangle-up", "size": 10}), 1, 1)
    fig.add_trace(go.Bar(x=px["date"], y=px["volume"], name="hacim"), 2, 1)
    for col in ["total_score", "accumulation_score"]:
        fig.add_trace(go.Scatter(x=ft["date"], y=ft[col], name=col), 3, 1)
    fig.add_hline(y=cfg.signal.min_total_score, line_dash="dot", row=3, col=1)
    fig.update_layout(height=750, xaxis_rangeslider_visible=False, margin={"t": 20})
    st.plotly_chart(fig, use_container_width=True)
    last = ft.iloc[-1]
    st.caption(
        f"Son gün {last['date']:%d.%m.%Y}: rejim {last['regime']}, likit {bool(last['liquidity_ok'])}, "
        f"trend {last['trend_score']:.0f}, momentum {last['momentum_score']:.0f}, hacim {last['volume_score']:.0f}, "
        f"volatilite {last['volatility_score']:.0f}, toplama {last['accumulation_score']:.0f}"
    )


def backtest_tab(cfg, market: str) -> None:
    for title, pattern in [("Baseline (kural bazlı)", f"backtest_baseline_{market}_*.md"), ("ML karşılaştırması", f"backtest_ml_{market}_*.md")]:
        path = _latest(pattern, cfg)
        with st.expander(title, expanded=title.startswith("Baseline")):
            st.markdown(path.read_text(encoding="utf-8") if path else "Rapor yok.")
    trials = store.reports_dir(cfg) / "backtest_trials.csv"
    if trials.exists():
        log = pd.read_csv(trials)
        st.caption(f"Denenen farklı ayar sayısı: {log['config_hash'].nunique()} (KURALLAR §4)")


def model_tab(market: str) -> None:
    registry = registry_for(market)
    latest = registry / "LATEST"
    if not latest.exists():
        st.info("Kayıtlı model yok: `python main.py train`")
        return
    versions = sorted([p.name for p in registry.iterdir() if p.is_dir()], reverse=True)
    version = st.selectbox("Model sürümü", versions, index=versions.index(latest.read_text().strip()))
    path = registry / version
    meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    approved = meta.get("approved_for_live")
    (st.success if approved else st.warning)(
        "Canlıya onaylı" if approved else "Canlıya ONAYSIZ — baseline'ı out-of-sample'da tutarlı geçemedi (KURALLAR §5)"
    )
    if "approval_evidence" in meta:
        st.json(meta["approval_evidence"])
    st.subheader("Walk-forward dönemleri")
    st.dataframe(pd.read_csv(path / "folds.csv"), hide_index=True)
    st.subheader("Özellik önemi")
    imp = pd.read_csv(path / "feature_importance.csv", index_col=0)["gain_share"].head(15)
    st.bar_chart(imp)


def main() -> None:
    st.set_page_config(page_title="Temkinli Hisse Tarama", layout="wide")
    cfg = load_config()
    markets = list(cfg.enabled_markets())
    market = st.sidebar.selectbox("Piyasa", markets)
    st.sidebar.caption(DISCLAIMER)
    st.title("Temkinli Hisse Tarama ve Sinyal Sistemi")
    tabs = st.tabs(["Sinyaller", "Hisse detay", "Backtest", "Model"])
    with tabs[0]:
        signals_tab(cfg, market)
    with tabs[1]:
        stock_tab(cfg, market)
    with tabs[2]:
        backtest_tab(cfg, market)
    with tabs[3]:
        model_tab(market)


main()
