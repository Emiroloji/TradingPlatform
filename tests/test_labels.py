import numpy as np
import pandas as pd
import pandas_ta as ta

import pytest

from backtest.engine import add_benchmark_returns, run_backtest
from config import load_config
from model.labels import horizon, ticker_labels
from tests.conftest import make_bars

N = 60


@pytest.fixture
def cfg():
    """Tests below the trade_outcome section exercise the original triple-barrier label."""
    base = load_config()
    return base.model_copy(update={"label": base.label.model_copy(update={"type": "triple_barrier"})})


@pytest.fixture
def trade_cfg():
    base = load_config()
    return base.model_copy(update={"label": base.label.model_copy(update={"type": "trade_outcome"})})


def _flat(price: float = 100.0) -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-01", periods=N, name="date")
    return pd.DataFrame({"open": price, "high": price * 1.01, "low": price * 0.99, "close": price, "volume": 1e6}, index=idx)


def test_upper_barrier_hit(cfg):
    bars, bench = _flat(), _flat(1000.0)
    t = 30
    bars.iloc[t + 3, bars.columns.get_loc("close")] = 100 * (1 + cfg.label.target_excess_return + 0.01)
    assert ticker_labels(bars, bench, cfg)["label"].iloc[t] == 1


def test_stop_before_target_is_zero_and_tie_is_zero(cfg):
    bars, bench = _flat(), _flat(1000.0)
    t = 30
    bars.iloc[t + 2, bars.columns.get_loc("low")] = 50.0  # stop first
    bars.iloc[t + 4, bars.columns.get_loc("close")] = 110.0
    assert ticker_labels(bars, bench, cfg)["label"].iloc[t] == 0

    bars2 = _flat()
    bars2.iloc[t + 2, bars2.columns.get_loc("low")] = 50.0  # same day: stop and target
    bars2.iloc[t + 2, bars2.columns.get_loc("close")] = 110.0
    assert ticker_labels(bars2, bench, cfg)["label"].iloc[t] == 0


def test_benchmark_rally_cancels_stock_rally(cfg):
    bars, bench = _flat(), _flat(1000.0)
    t = 30
    bars.iloc[t + 3, bars.columns.get_loc("close")] = 110.0
    bench.iloc[t + 3, bench.columns.get_loc("close")] = 1100.0  # index up just as much
    assert ticker_labels(bars, bench, cfg)["label"].iloc[t] == 0


def test_tail_is_unknown_and_label_end(cfg):
    h = cfg.label.horizon_days
    out = ticker_labels(_flat(), _flat(1000.0), cfg)
    assert out["label"].iloc[-h:].isna().all()
    assert out["label_end"].iloc[20] == out.index[20 + h]


def test_label_uses_only_its_window(cfg):
    h = cfg.label.horizon_days
    bars, bench = make_bars(n=200, seed=7), make_bars(n=200, seed=8)
    base = ticker_labels(bars, bench, cfg)
    t = 100
    changed = bars.copy()
    changed.iloc[t + h + 1 :] *= 3  # anything after t+h must not matter for day t
    assert ticker_labels(changed, bench, cfg)["label"].iloc[t] == base["label"].iloc[t]


def _reference(bars, bench, cfg):
    """Plain loop implementation of the same rule."""
    h, k, target = cfg.label.horizon_days, cfg.risk.atr_stop_multiplier, cfg.label.target_excess_return
    atr = ta.atr(bars["high"], bars["low"], bars["close"], length=cfg.risk.atr_period)
    out = []
    for t in range(len(bars) - h):
        if np.isnan(atr.iloc[t]):
            out.append(np.nan)
            continue
        entry, b_entry = bars["open"].iloc[t + 1], bench["open"].iloc[t + 1]
        stop, lab = entry - k * atr.iloc[t], 0.0
        for d in range(t + 1, t + h + 1):
            if bars["low"].iloc[d] <= stop:
                break
            if bars["close"].iloc[d] / entry - bench["close"].iloc[d] / b_entry >= target:
                lab = 1.0
                break
        out.append(lab)
    return np.array(out)


def test_matches_reference_loop(cfg):
    bars, bench = make_bars(n=250, seed=21), make_bars(n=250, seed=22)
    fast = ticker_labels(bars, bench, cfg)["label"].to_numpy()[: 250 - cfg.label.horizon_days]
    np.testing.assert_array_equal(fast, _reference(bars, bench, cfg))
    assert 0 < np.nanmean(fast) < 1  # both classes present


# ---------------------------------------------------------------- trade_outcome


def _long(bars: pd.DataFrame, ticker: str) -> pd.DataFrame:
    return bars.rename_axis("date").reset_index().assign(ticker=ticker)


@pytest.mark.parametrize("seed", [31, 32, 33])
def test_trade_outcome_matches_backtest_engine(trade_cfg, seed):
    """The label must equal the sign of the engine's own excess return for the same trade."""
    cfg = trade_cfg
    bars, bench = make_bars(n=260, seed=seed), make_bars(n=260, seed=seed + 100)
    labels = ticker_labels(bars, bench, cfg)["label"]
    atr = ta.atr(bars["high"], bars["low"], bars["close"], length=cfg.risk.atr_period)
    for t in range(20, 260 - horizon(cfg) - 2, 9):
        signal = pd.DataFrame(
            [{"date": bars.index[t], "ticker": "A", "sector": "S", "total_score": 1.0, "atr": atr.iloc[t], "regime": "bull"}]
        )
        trades = run_backtest(signal, _long(bars, "A"), cfg).trades
        trades = add_benchmark_returns(trades, _long(bench, "X"))
        expected = float(trades["excess_return"].iloc[0] > cfg.label.min_trade_excess)
        assert labels.iloc[t] == expected, (t, trades.iloc[0].to_dict())


def test_trade_outcome_both_classes_and_tail(trade_cfg):
    out = ticker_labels(make_bars(n=300, seed=40), make_bars(n=300, seed=41), trade_cfg)
    h = horizon(trade_cfg)
    assert out["label"].iloc[-h:].isna().all()
    assert 0 < out["label"].mean() < 1
    assert out["label_end"].iloc[50] == out.index[50 + h]


def test_trade_returns_equal_engine_returns(trade_cfg):
    from model.labels import ticker_trade_returns

    cfg = trade_cfg
    bars, bench = make_bars(n=200, seed=51), make_bars(n=200, seed=52)
    tr = ticker_trade_returns(bars, bench, cfg)
    atr = ta.atr(bars["high"], bars["low"], bars["close"], length=cfg.risk.atr_period)
    for t in range(20, 170, 13):
        signal = pd.DataFrame(
            [{"date": bars.index[t], "ticker": "A", "sector": "S", "total_score": 1.0, "atr": atr.iloc[t], "regime": "bull"}]
        )
        trade = add_benchmark_returns(run_backtest(signal, _long(bars, "A"), cfg).trades, _long(bench, "X")).iloc[0]
        assert tr["trade_return"].iloc[t] == pytest.approx(trade["return_net"])
        assert tr["bench_return"].iloc[t] == pytest.approx(trade["bench_return"])
        assert tr["exit_date"].iloc[t] == trade["exit_date"]
