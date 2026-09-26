from features.liquidity import liquidity
from tests.conftest import assert_no_lookahead


def test_no_lookahead(bars, cfg):
    assert_no_lookahead(lambda d: liquidity(d, cfg, "bist").astype(float), bars, [5, 19, 100, 449])


def test_threshold_and_warmup(bars, cfg):
    n = cfg.liquidity.lookback_days
    threshold = cfg.liquidity.min_avg_turnover_usd["bist"]
    liquid = bars.copy()
    liquid["volume"] = threshold * 1.001 / liquid["close"]  # turnover just above threshold every day
    out = liquidity(liquid, cfg, "bist")
    assert not out["liquidity_ok"].iloc[: n - 1].any()  # warm-up never passes
    assert out["liquidity_ok"].iloc[n - 1 :].all()

    thin = liquid.copy()
    thin["volume"] *= 0.5
    assert not liquidity(thin, cfg, "bist")["liquidity_ok"].any()


def test_turnover_converted_to_usd(bars, cfg):
    threshold = cfg.liquidity.min_avg_turnover_usd["bist"]
    local = bars.copy()
    local["volume"] = threshold * 1.001 * 40 / local["close"]  # passes only at <= 40 local per USD
    fx_cheap = local["close"] * 0 + 40.0
    fx_dear = local["close"] * 0 + 45.0
    assert liquidity(local, cfg, "bist", fx_cheap)["liquidity_ok"].iloc[-1]
    assert not liquidity(local, cfg, "bist", fx_dear)["liquidity_ok"].iloc[-1]
