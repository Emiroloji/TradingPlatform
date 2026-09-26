import pandas as pd
import pytest

from backtest.engine import add_benchmark_returns, run_backtest
from tests.conftest import make_bars

DATES = pd.bdate_range("2024-01-01", periods=40)


def _flat(ticker: str, price: float = 100.0, overrides: dict | None = None) -> pd.DataFrame:
    """Flat bars (open=close=price, high/low ±0.5%) with per-day overrides {day_index: {col: val}}."""
    df = pd.DataFrame(
        {"date": DATES, "ticker": ticker, "open": price, "high": price * 1.005, "low": price * 0.995, "close": price}
    )
    for i, cols in (overrides or {}).items():
        for c, v in cols.items():
            df.loc[i, c] = v
    return df


def _signal(ticker: str, day: int, atr: float = 2.0, sector: str = "S1", score: float = 80.0) -> dict:
    return {"date": DATES[day], "ticker": ticker, "sector": sector, "total_score": score, "atr": atr, "regime": "bull"}


def _run(prices, signals, cfg):
    return run_backtest(pd.DataFrame(signals), pd.concat(prices, ignore_index=True), cfg)


def test_fill_next_open_with_costs_and_stop_exit(cfg):
    bt = cfg.backtest
    # signal day 5 -> entry day 6 open; day 8 low touches stop (entry - 2*ATR = ~96)
    prices = [_flat("A", overrides={8: {"low": 90.0}})]
    res = _run(prices, [_signal("A", 5)], cfg)
    t = res.trades.iloc[0]
    entry = 100 * (1 + bt.slippage)
    stop = entry - cfg.risk.atr_stop_multiplier * 2.0
    assert t["entry_date"] == DATES[6]
    assert t["entry_price"] == pytest.approx(entry)
    assert t["exit_reason"] == "stop" and t["exit_date"] == DATES[8]
    exit_price = stop * (1 - bt.slippage)
    assert t["exit_price"] == pytest.approx(exit_price)
    expected = exit_price * (1 - bt.commission) / (entry * (1 + bt.commission)) - 1
    assert t["return_net"] == pytest.approx(expected)


def test_target_exit(cfg):
    prices = [_flat("A", overrides={10: {"high": 120.0}})]
    t = _run(prices, [_signal("A", 5)], cfg).trades.iloc[0]
    assert t["exit_reason"] == "target"
    assert t["target"] == pytest.approx(t["entry_price"] + cfg.backtest.target_r_multiple * (t["entry_price"] - t["stop"]))


def test_bar_touching_stop_and_target_counts_as_stop(cfg):
    prices = [_flat("A", overrides={9: {"high": 120.0, "low": 80.0}})]
    assert _run(prices, [_signal("A", 5)], cfg).trades.iloc[0]["exit_reason"] == "stop"


def test_gap_below_stop_exits_at_open(cfg):
    prices = [_flat("A", overrides={9: {"open": 85.0, "low": 84.0, "close": 86.0}})]
    t = _run(prices, [_signal("A", 5)], cfg).trades.iloc[0]
    assert t["exit_reason"] == "stop_gap"
    assert t["exit_price"] == pytest.approx(85.0 * (1 - cfg.backtest.slippage))


def test_time_exit_after_max_holding(cfg):
    t = _run([_flat("A")], [_signal("A", 5)], cfg).trades.iloc[0]
    assert t["exit_reason"] == "time"
    assert t["holding_days"] == cfg.backtest.max_holding_days
    assert t["exit_date"] == DATES[6 + cfg.backtest.max_holding_days - 1]


def test_signal_on_last_day_is_never_filled(cfg):
    res = _run([_flat("A")], [_signal("A", len(DATES) - 1)], cfg)
    assert res.trades.empty


def test_position_size_from_risk_rule(cfg):
    t = _run([_flat("A")], [_signal("A", 5)], cfg).trades.iloc[0]
    risk_per_share = t["entry_price"] - t["stop"]
    assert t["shares"] == int(cfg.risk.capital * cfg.risk.risk_per_trade // risk_per_share)


def test_max_positions_and_sector_limit(cfg):
    tickers = [f"T{i}" for i in range(8)]
    prices = [_flat(t) for t in tickers]
    # 3 in sector S1 (limit 2), 5 in other sectors; slots = 5
    # ATR 5 -> each position ~10% of capital, so cash never binds before the limits do
    signals = [_signal(t, 5, atr=5.0, sector="S1" if i < 3 else f"S{i}", score=90 - i) for i, t in enumerate(tickers)]
    res = _run(prices, signals, cfg)
    assert len(res.trades) == cfg.risk.max_positions
    assert (res.trades["sector"] == "S1").sum() == cfg.risk.max_per_sector
    assert "T2" not in set(res.trades["ticker"])  # third S1 name blocked by sector limit


def test_no_lookahead_trades_closed_before_cut_unchanged(cfg):
    bars = make_bars(n=300, seed=4).rename_axis("date").reset_index().assign(ticker="A")
    days = bars["date"]
    signals = [
        {"date": days[i], "ticker": "A", "sector": "S1", "total_score": 80.0, "atr": 3.0, "regime": "bull"}
        for i in range(10, 280, 7)
    ]
    full = run_backtest(pd.DataFrame(signals), bars, cfg)
    cut = days[200]
    part = run_backtest(pd.DataFrame([s for s in signals if s["date"] <= cut]), bars[bars["date"] <= cut], cfg)
    done = full.trades[full.trades["exit_date"] < cut].reset_index(drop=True)
    done_part = part.trades[part.trades["exit_date"] < cut].reset_index(drop=True)
    pd.testing.assert_frame_equal(done, done_part)
    pd.testing.assert_series_equal(full.equity[full.equity.index < cut], part.equity[part.equity.index < cut])


def test_benchmark_excess_return(cfg):
    prices = [_flat("A", overrides={10: {"high": 120.0}})]
    trades = _run(prices, [_signal("A", 5)], cfg).trades
    bench = _flat("XU100.IS", price=1000.0)
    out = add_benchmark_returns(trades, bench)
    assert out["bench_return"].iloc[0] == pytest.approx(0)
    assert out["excess_return"].iloc[0] == pytest.approx(out["return_net"].iloc[0])


def test_no_leverage_cash_limits_positions(cfg):
    # ATR 2 -> 4% stop -> each position ~25% of capital; only 4 fit in cash
    tickers = [f"T{i}" for i in range(5)]
    res = _run([_flat(t) for t in tickers], [_signal(t, 5, sector=f"S{i}") for i, t in enumerate(tickers)], cfg)
    assert len(res.trades) == 4
    assert res.equity.min() > 0
