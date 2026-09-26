import numpy as np
import pandas as pd

from features.accumulation import CONDITIONS, accumulation_conditions, accumulation_score
from tests.conftest import assert_no_lookahead, make_bars

CUTS = [50, 160, 250, 300, 420]


def test_no_lookahead(bars, cfg):
    assert_no_lookahead(lambda d: accumulation_conditions(d, cfg), bars, CUTS)
    assert_no_lookahead(lambda d: accumulation_score(d, cfg), bars, CUTS)


def test_score_follows_condition_count(bars, cfg):
    cond = accumulation_conditions(bars, cfg)[CONDITIONS]
    score = accumulation_score(bars, cfg)
    table = cfg.features.accumulation.score_by_conditions
    valid = cond.dropna().index
    assert len(valid) > 0
    expected = cond.loc[valid].sum(axis=1).astype(int).map(lambda n: table[n])
    pd.testing.assert_series_equal(score.loc[valid], expected, check_names=False, check_dtype=False)
    assert score.isna().sum() > 0  # warm-up is unknown, not zero


def test_single_condition_cannot_reach_signal_threshold(cfg):
    table = cfg.features.accumulation.score_by_conditions
    assert table[1] < cfg.signal.min_accumulation_score


def test_quiet_band_with_rising_volume_scores_high(cfg):
    bars = make_bars(n=300, seed=3)
    w = cfg.features.accumulation.window_days
    tail = bars.index[-w:]
    last = bars["close"].iloc[-w - 1]
    rng = np.random.default_rng(5)
    flat = last * (1 + rng.normal(0, 0.002, w))
    bars.loc[tail, "close"] = flat
    bars.loc[tail, "open"] = flat
    bars.loc[tail, "high"] = flat * 1.003
    bars.loc[tail, "low"] = flat * 0.997
    bars.loc[tail, "volume"] = bars["volume"].median() * 2  # heavier volume in the band
    cond = accumulation_conditions(bars, cfg).iloc[-1]
    assert cond["acc_contraction"] == 1 and cond["acc_volume_up"] == 1
    assert accumulation_score(bars, cfg).iloc[-1] >= cfg.signal.min_accumulation_score
