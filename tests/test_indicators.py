import numpy as np
import pytest

from features import indicators
from tests.conftest import assert_no_lookahead

CUTS = [50, 199, 250, 300, 420]


@pytest.mark.parametrize("fn", [indicators.trend, indicators.momentum, indicators.volume, indicators.volatility])
def test_no_lookahead(fn, bars, cfg):
    assert_no_lookahead(lambda d: fn(d, cfg), bars, CUTS)


def test_scores_bounded_and_nan_during_warmup(bars, cfg):
    out = indicators.compute_indicators(bars, cfg)
    for col in ["trend_score", "momentum_score", "volume_score", "volatility_score"]:
        valid = out[col].dropna()
        assert valid.between(0, 100).all()
        assert len(valid) > 0
    # EMA200 not ready before 200 bars -> trend score must be unknown, not zero
    assert out["trend_score"].iloc[:150].isna().all()
    assert out["trend_score"].iloc[-1] == out["trend_score"].iloc[-1]  # not NaN


def test_trend_score_high_in_steady_uptrend(bars, cfg):
    up = bars.copy()
    ramp = np.linspace(1, 3, len(up))
    for c in ["open", "high", "low", "close"]:
        up[c] = up[c].iloc[0] * ramp * (up[c] / up["close"])
    assert indicators.trend(up, cfg)["trend_score"].iloc[-1] >= 80
