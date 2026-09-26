import pandas as pd
import pytest

from model.labels import ticker_trade_returns
from signals.journal import live_performance, read_journal, record_signals, update_outcomes
from tests.conftest import make_bars


@pytest.fixture
def local(cfg, tmp_path):
    return cfg.model_copy(update={"data": cfg.data.model_copy(update={"storage_dir": str(tmp_path)})})


def _long(bars, ticker):
    return bars.rename_axis("date").reset_index().assign(ticker=ticker)


def _signal(day, ticker="A.IS"):
    return pd.DataFrame([{"date": day, "ticker": ticker, "total_score": 80.0, "stop": 1.0}])


def test_record_is_append_only(local):
    day = pd.Timestamp("2024-01-10")
    assert record_signals(_signal(day), local, "bist") == 1
    assert record_signals(_signal(day), local, "bist") == 0  # same day/ticker not duplicated
    assert len(read_journal(local, "bist")) == 1


def test_outcome_filled_only_after_holding_window(local):
    bars, bench = make_bars(n=120, seed=61), make_bars(n=120, seed=62)
    h = local.backtest.max_holding_days
    early, late = bars.index[40], bars.index[120 - h]  # `late` has no complete window yet
    record_signals(pd.concat([_signal(early), _signal(late)]), local, "bist")
    filled = update_outcomes(local, "bist", _long(bars, "A.IS"), _long(bench, "XU100.IS"))
    assert filled == 1

    j = read_journal(local, "bist").set_index("date")
    expected = ticker_trade_returns(bars, bench, local).loc[early]
    assert j.loc[early, "outcome_return"] == pytest.approx(expected["trade_return"])
    assert j.loc[early, "outcome_relative"] == pytest.approx(expected["trade_return"] - expected["bench_return"])
    assert pd.isna(j.loc[late, "outcome_return"])

    perf = live_performance(read_journal(local, "bist"))
    assert perf["signals"] == 2 and perf["closed"] == 1
