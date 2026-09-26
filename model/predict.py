"""Probabilities for the latest day from a saved model version."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from model.train import REGIME_CODES, registry_for


@dataclass(frozen=True)
class LoadedModel:
    version: str
    booster: lgb.Booster
    meta: dict


def load_model(market: str, version: str | None = None, root: Path | None = None) -> LoadedModel:
    root = root or registry_for(market)
    version = version or (root / "LATEST").read_text(encoding="utf-8").strip()
    path = root / version
    meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    return LoadedModel(version, lgb.Booster(model_file=str(path / "final_model.txt")), meta)


def load_oos_predictions(market: str, version: str | None = None, root: Path | None = None) -> pd.DataFrame:
    root = root or registry_for(market)
    version = version or (root / "LATEST").read_text(encoding="utf-8").strip()
    return pd.read_parquet(root / version / "oos_predictions.parquet")


def predict(model: LoadedModel, features_today: pd.DataFrame) -> pd.DataFrame:
    """MIMARI §4 predictions table: date, ticker, prob, model_version (features as saved with the model)."""
    X = features_today.assign(regime_code=features_today["regime"].map(REGIME_CODES))[model.meta["features"]].astype(float)
    return features_today[["date", "ticker"]].assign(prob=model.booster.predict(X), model_version=model.version)
