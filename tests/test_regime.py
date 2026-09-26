import numpy as np
import pandas as pd

from features.regime import BEAR, BULL, market_regime
from tests.conftest import assert_no_lookahead, make_bars


def _trend(direction: float) -> pd.DataFrame:
    bars = make_bars(n=400, seed=2)
    ramp = np.exp(np.linspace(0, direction, len(bars)))
    for c in ["open", "high", "low", "close"]:
        bars[c] = 100 * ramp * (bars[c] / bars["close"])
    return bars


def test_no_lookahead(benchmark, cfg):
    def fn(d):
        return market_regime(d, cfg).fillna("nan")

    assert_no_lookahead(fn, benchmark, [100, 230, 300, 399])


def test_warmup_is_unknown(benchmark, cfg):
    regime = market_regime(benchmark, cfg)
    assert regime.iloc[: cfg.features.regime.ema_period - 1].isna().all()
    assert set(regime.dropna().unique()) <= {"bull", "bear", "sideways"}


def test_uptrend_is_bull_and_downtrend_is_bear(cfg):
    assert market_regime(_trend(1.0), cfg).iloc[-1] == BULL
    assert market_regime(_trend(-1.0), cfg).iloc[-1] == BEAR
