import numpy as np
import pandas as pd
import pytest

from features import compute_features
from features.scoring import active_weights, total_score
from tests.conftest import make_bars


def _scores(**cols) -> pd.DataFrame:
    return pd.DataFrame({f"{k}_score": v for k, v in cols.items()})


def test_weights_renormalised_over_enabled_categories(cfg):
    scores = _scores(trend=[100.0], momentum=[100.0], volume=[100.0], volatility=[100.0], fundamental=[np.nan])
    w = active_weights(cfg)
    assert set(w) == {"trend", "momentum", "volume", "volatility"}
    assert sum(w.values()) == pytest.approx(1)
    assert w["trend"] == pytest.approx(25 / 80)
    assert total_score(scores, cfg).iloc[0] == pytest.approx(100)


def test_weighted_average(cfg):
    scores = _scores(trend=[100.0], momentum=[0.0], volume=[0.0], volatility=[0.0])
    assert total_score(scores, cfg).iloc[0] == pytest.approx(100 * 25 / 80)


def test_warmup_row_is_nan_not_reweighted(cfg):
    scores = _scores(trend=[np.nan, 50.0], momentum=[80.0, 50.0], volume=[80.0, 50.0], volatility=[80.0, 50.0])
    out = total_score(scores, cfg)
    assert np.isnan(out.iloc[0])
    assert out.iloc[1] == pytest.approx(50)


def _long(bars: pd.DataFrame, ticker: str) -> pd.DataFrame:
    return bars.rename_axis("date").reset_index().assign(ticker=ticker)


def test_compute_features_table_has_no_lookahead(cfg):
    prices = pd.concat([_long(make_bars(seed=10), "AAA.IS"), _long(make_bars(seed=11), "BBB.IS")])
    bench = _long(make_bars(seed=12), "XU100.IS")
    fx = _long(make_bars(seed=13), "USDTRY=X")
    full = compute_features(prices, bench, cfg, "bist", fx).set_index(["date", "ticker"])
    dates = sorted(prices["date"].unique())

    for t in [150, 260, 449]:
        cut = dates[t]
        part = compute_features(
            prices[prices["date"] <= cut], bench[bench["date"] <= cut], cfg, "bist", fx[fx["date"] <= cut]
        )
        part = part.set_index(["date", "ticker"])
        pd.testing.assert_frame_equal(part.xs(cut, level="date"), full.xs(cut, level="date"), rtol=1e-9)


def test_compute_features_columns(cfg):
    prices = _long(make_bars(seed=10), "AAA.IS")
    table = compute_features(prices, _long(make_bars(seed=12), "XU100.IS"), cfg, "bist")
    required = {
        "date", "ticker", "trend_score", "momentum_score", "volume_score", "volatility_score",
        "fundamental_score", "accumulation_score", "relative_strength", "regime", "liquidity_ok", "total_score",
    }
    assert required <= set(table.columns)
    assert table[["date", "ticker"]].duplicated().sum() == 0
    assert table["total_score"].dropna().between(0, 100).all()
