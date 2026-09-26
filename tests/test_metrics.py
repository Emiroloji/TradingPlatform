import pandas as pd
import pytest

from backtest.metrics import breakdown, compute_metrics, max_drawdown

BENCH = pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=5), "close": [100.0, 110.0, 99.0, 105.0, 120.0]})


def _trades(returns: list[float], excess: list[float] | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "return_net": returns,
            "excess_return": excess if excess is not None else returns,
            "pnl": [r * 1000 for r in returns],
            "holding_days": [5] * len(returns),
            "exit_reason": ["stop"] * len(returns),
            "regime": ["bull"] * len(returns),
        }
    )


def test_expectancy_formula(cfg):
    m = compute_metrics(_trades([0.10, 0.10, -0.05, -0.05, -0.05]), BENCH, cfg=cfg)
    assert m["win_rate"] == pytest.approx(0.4)
    assert m["avg_win"] == pytest.approx(0.10)
    assert m["avg_loss"] == pytest.approx(-0.05)
    assert m["expectancy"] == pytest.approx(0.4 * 0.10 - 0.6 * 0.05)
    assert m["expectancy"] == pytest.approx(_trades([0.10, 0.10, -0.05, -0.05, -0.05])["return_net"].mean())
    assert m["profit_factor"] == pytest.approx(200 / 150)


def test_insufficient_data_flag(cfg):
    assert compute_metrics(_trades([0.01] * 29), BENCH, cfg=cfg)["insufficient_data"]
    assert not compute_metrics(_trades([0.01] * 30), BENCH, cfg=cfg)["insufficient_data"]


def test_max_drawdown():
    assert max_drawdown(pd.Series([100.0, 120.0, 90.0, 130.0])) == pytest.approx(-0.25)
    assert max_drawdown(pd.Series([1.0, 2.0, 3.0])) == 0


def test_relative_return_vs_benchmark(cfg):
    equity = pd.Series([100.0, 105.0, 110.0, 115.0, 130.0], index=BENCH["date"])
    m = compute_metrics(_trades([0.3]), BENCH, equity, cfg)
    assert m["total_return"] == pytest.approx(0.30)
    assert m["benchmark_return"] == pytest.approx(0.20)
    assert m["relative_return"] == pytest.approx(0.10)
    assert m["benchmark_max_drawdown"] == pytest.approx(99 / 110 - 1)


def test_breakdown_by_group(cfg):
    trades = _trades([0.1, -0.1, 0.2])
    trades["regime"] = ["bull", "bear", "bull"]
    table = breakdown(trades, BENCH, trades["regime"], cfg)
    assert table.loc["bull", "trades"] == 2
    assert table.loc["bear", "expectancy"] == pytest.approx(-0.1)
