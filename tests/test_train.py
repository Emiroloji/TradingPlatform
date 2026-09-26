import numpy as np
import pandas as pd
import pytest

from model.predict import load_model, predict
from model.train import design_matrix, feature_names, save_artifact, train_walk_forward


def _dataset(cfg, seed: int = 0):
    """Synthetic features/labels 2019-2025 where label depends on rsi (learnable signal)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-07-01", "2025-06-30")
    tickers = [f"T{i}" for i in range(6)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    f = pd.DataFrame(index=idx).reset_index()
    for name in feature_names(cfg):
        f[name] = rng.normal(size=len(f))
    f["rsi"] = rng.uniform(0, 100, len(f))
    f["regime"] = rng.choice(["bull", "bear", "sideways"], len(f))
    f["liquidity_ok"] = True
    h = cfg.label.horizon_days
    date_pos = {d: i for i, d in enumerate(dates)}
    labels = f[["date", "ticker"]].copy()
    labels["label"] = (f["rsi"] + rng.normal(0, 15, len(f)) > 50).astype(float)
    labels["label_end"] = [dates[min(date_pos[d] + h, len(dates) - 1)] for d in f["date"]]
    labels.loc[labels["date"] > dates[-h - 1], "label"] = np.nan
    return f.drop(columns="regime_code"), labels


@pytest.fixture(scope="module")
def trained(request):
    from config import load_config

    cfg = load_config()
    f, labels = _dataset(cfg)
    return cfg, f, labels, train_walk_forward(f, labels, cfg)


def test_folds_are_purged_and_ordered(trained):
    cfg, f, labels, art = trained
    for _, fold in art.folds.iterrows():
        assert fold["train_last_label_end"] < fold["test_start"]  # purge
        assert pd.Timestamp(fold["train_start"]) == pd.Timestamp(fold["test_start"]) - pd.DateOffset(years=cfg.model.train_years)
    assert str(art.folds["test_start"].iloc[0]) == cfg.model.first_test_start


def test_each_oos_row_predicted_once_and_only_after_first_test(trained):
    cfg, f, labels, art = trained
    p = art.oos_predictions
    assert not p.duplicated(["date", "ticker"]).any()
    assert p["date"].min() >= pd.Timestamp(cfg.model.first_test_start)
    assert p["prob"].between(0, 1).all()


def test_learns_signal(trained):
    _, _, _, art = trained
    assert art.folds["auc"].min() > 0.75
    assert art.importance.index[0] == "rsi"


def test_future_data_does_not_change_earlier_fold(trained):
    cfg, f, labels, art = trained
    first_end = pd.Timestamp(art.folds["test_end"].iloc[0])
    f2, labels2 = f.copy(), labels.copy()
    later = f2["date"] >= first_end
    f2.loc[later, "rsi"] = 100 - f2.loc[later, "rsi"]  # invert the relationship after fold 0
    art2 = train_walk_forward(f2, labels2, cfg)
    a = art.oos_predictions.query("fold == 0").reset_index(drop=True)
    b = art2.oos_predictions.query("fold == 0").reset_index(drop=True)
    pd.testing.assert_series_equal(a["prob"], b["prob"])


def test_save_load_predict_roundtrip(trained, tmp_path):
    cfg, f, labels, art = trained
    save_artifact(art, root=tmp_path)
    loaded = load_model("bist", root=tmp_path)
    assert loaded.version == art.version
    today = f[f["date"] == f["date"].max()]
    out = predict(loaded, today)
    expected = art.final_model.predict_proba(design_matrix(today, cfg))[:, 1]
    np.testing.assert_allclose(out["prob"].to_numpy(), expected, rtol=1e-6)
    assert list(out.columns) == ["date", "ticker", "prob", "model_version"]
