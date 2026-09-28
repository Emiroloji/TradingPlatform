"""Walk-forward LightGBM (KURALLAR §5): no random splits, purged by label horizon.

Folds: train on [test_start - train_years, test_start) using only rows whose label is already
known before test_start (label_end < test_start), predict [test_start, test_start + test_months),
then move test_start forward by step_months. Every out-of-sample probability therefore comes
from a model that never saw that period.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import Config

logger = logging.getLogger(__name__)

REGIME_CODES = {"bear": -1, "sideways": 0, "bull": 1}
REGISTRY = Path(__file__).parent / "registry"


def registry_for(market: str) -> Path:
    """Each market keeps its own model versions and LATEST pointer."""
    return REGISTRY / market


def feature_names(cfg: Config) -> list[str]:
    """Scale-free features only: price/volume *levels* (EMA values, OBV, MACD, ATR, turnover)
    drift with inflation and would let the model learn the calendar instead of the setup."""
    return (
        [f"close_vs_ema_{n}" for n in cfg.features.trend.ema_periods]
        + ["adx", "dmp", "dmn", "rsi", "stoch_k", "stoch_d", "roc", "cmf", "volume_ratio", "atr_pct", "bb_width"]
        + ["acc_range_pct", "acc_volume_ratio", "acc_contraction", "acc_volume_up", "acc_divergence", "acc_spring"]
        + [f"rs_{n}" for n in cfg.features.relative.periods]
        + ["relative_strength", "trend_score", "momentum_score", "volume_score", "volatility_score"]
        + ["accumulation_score", "total_score", "regime_code"]
    )


def training_columns(cfg: Config) -> list[str]:
    """Feature-table columns train_walk_forward reads (market-specific ones may be absent)."""
    from features.insider import COLUMNS as INSIDER_COLUMNS
    from features.institutional import COLUMNS as INSTITUTIONAL_COLUMNS

    base = [c for c in feature_names(cfg) if c != "regime_code"]
    return ["date", "ticker", "liquidity_ok", "regime", *base, *INSIDER_COLUMNS, *INSTITUTIONAL_COLUMNS]


def design_matrix(features: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Base features plus market-specific ones present as columns (e.g. US insider features).
    Column presence is fixed per market, never per row, so it cannot leak future information."""
    from features.insider import COLUMNS as INSIDER_COLUMNS
    from features.institutional import COLUMNS as INSTITUTIONAL_COLUMNS

    X = features.assign(regime_code=features["regime"].map(REGIME_CODES))
    names = feature_names(cfg) + [c for c in INSIDER_COLUMNS + INSTITUTIONAL_COLUMNS if c in X.columns]
    return X[names].astype(float)


@dataclass
class ModelArtifact:
    version: str
    features: list[str]
    final_model: lgb.LGBMClassifier
    fold_models: list[lgb.LGBMClassifier]
    folds: pd.DataFrame  # one row per fold: dates, sizes, AUC, base rate
    oos_predictions: pd.DataFrame  # date, ticker, prob, fold
    importance: pd.Series  # mean normalised gain across folds
    meta: dict = field(default_factory=dict)


def fold_windows(first_test: pd.Timestamp, last_date: pd.Timestamp, cfg: Config) -> list[tuple]:
    m = cfg.model
    folds, start = [], first_test
    while start <= last_date:
        end = start + pd.DateOffset(months=m.test_months)
        folds.append((start - pd.DateOffset(years=m.train_years), start, end))
        start = start + pd.DateOffset(months=m.step_months)
    return folds


def _fit(X: pd.DataFrame, y: pd.Series, cfg: Config) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(**cfg.model.params)
    model.fit(X, y)
    return model


def _gain(model: lgb.LGBMClassifier, names: list[str]) -> pd.Series:
    gain = pd.Series(model.booster_.feature_importance(importance_type="gain"), index=names)
    return gain / gain.sum() if gain.sum() > 0 else gain


def train_walk_forward(features: pd.DataFrame, labels: pd.DataFrame, cfg: Config) -> ModelArtifact:
    """`features`: features table; `labels`: make_labels_frame output (date, ticker, label, label_end).

    Trains on liquid rows with a known label; predicts every row of each test window.
    """
    data = features.merge(labels, on=["date", "ticker"], how="left")
    del features  # frees the table when the caller passed it without keeping a reference
    trainable = data["liquidity_ok"].astype(bool) & data["label"].notna()
    X_all = design_matrix(data, cfg)
    names = list(X_all.columns)
    data = data[["date", "ticker", "label", "label_end"]]  # the feature columns now live in X_all

    fold_rows, preds, models, gains = [], [], [], []
    windows = fold_windows(pd.Timestamp(cfg.model.first_test_start), data["date"].max(), cfg)
    for i, (train_start, test_start, test_end) in enumerate(windows):
        train = trainable & (data["date"] >= train_start) & (data["label_end"] < test_start)
        test = (data["date"] >= test_start) & (data["date"] < test_end)
        model = _fit(X_all[train], data.loc[train, "label"], cfg)
        prob = model.predict_proba(X_all[test])[:, 1]
        preds.append(data.loc[test, ["date", "ticker"]].assign(prob=prob, fold=i))

        scored = test & trainable
        y_test = data.loc[scored, "label"]
        auc = roc_auc_score(y_test, prob[scored[test].to_numpy()]) if y_test.nunique() == 2 else np.nan
        fold_rows.append(
            {
                "fold": i,
                "train_start": train_start.date(),
                "test_start": test_start.date(),
                "test_end": min(test_end, data["date"].max() + pd.Timedelta(days=1)).date(),
                "train_rows": int(train.sum()),
                "train_last_label_end": data.loc[train, "label_end"].max().date(),
                "test_rows": int(test.sum()),
                "test_labelled": int(scored.sum()),
                "base_rate": float(y_test.mean()) if len(y_test) else np.nan,
                "auc": auc,
                "share_prob_ge_threshold": float((prob >= cfg.signal.min_probability).mean()),
            }
        )
        models.append(model)
        gains.append(_gain(model, names))
        logger.info("Fold %d: train %d rows -> test %s..%s AUC %.3f", i, train.sum(), test_start.date(), test_end.date(), auc)

    # model for live use: the most recent train_years with labels known today
    last = data["date"].max()
    final_rows = trainable & (data["date"] >= last - pd.DateOffset(years=cfg.model.train_years))
    final = _fit(X_all[final_rows], data.loc[final_rows, "label"], cfg)

    settings = {k: getattr(cfg, k).model_dump() for k in ["features", "label", "model"]}
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True, default=str).encode()).hexdigest()[:8]
    version = f"{datetime.now():%Y%m%d-%H%M%S}-{digest}"
    return ModelArtifact(
        version=version,
        features=names,
        final_model=final,
        fold_models=models,
        folds=pd.DataFrame(fold_rows),
        oos_predictions=pd.concat(preds, ignore_index=True).assign(model_version=version),
        importance=pd.concat(gains, axis=1).mean(axis=1).sort_values(ascending=False),
        meta={
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "data_end": str(last.date()),
            "final_train_rows": int(final_rows.sum()),
            "config_hash": digest,
            "settings": settings,
            "approved_for_live": False,  # set only after the backtest comparison (KURALLAR §5)
        },
    )


def save_artifact(art: ModelArtifact, root: Path) -> Path:
    """<root>/<version>/ (root = registry_for(market)): boosters, fold table, OOS predictions, importance, metadata."""
    path = root / art.version
    path.mkdir(parents=True, exist_ok=True)
    art.final_model.booster_.save_model(str(path / "final_model.txt"))
    for i, m in enumerate(art.fold_models):
        m.booster_.save_model(str(path / f"fold_{i}.txt"))
    art.folds.to_csv(path / "folds.csv", index=False)
    art.oos_predictions.to_parquet(path / "oos_predictions.parquet", index=False)
    art.importance.rename("gain_share").to_csv(path / "feature_importance.csv")
    (path / "metadata.json").write_text(
        json.dumps(art.meta | {"version": art.version, "features": art.features}, indent=2, default=str), encoding="utf-8"
    )
    (root / "LATEST").write_text(art.version, encoding="utf-8")
    logger.info("Saved model %s", path)
    return path


def set_approval(version: str, approved: bool, evidence: dict, root: Path) -> None:
    """Record the backtest verdict next to the model (every version keeps its metrics)."""
    meta_path = root / version / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["approved_for_live"] = approved
    meta["approval_evidence"] = evidence
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
