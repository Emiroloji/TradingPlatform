"""Weighted total score (0-100) from category scores, weights from config.

Only `scoring.enabled_categories` take part (fundamental data is not collected yet,
sentiment arrives in Faz 4); their weights are renormalised to sum to 1. The set is fixed
by config, never inferred from the data, so a day's score cannot depend on later bars.
A NaN category on a row (warm-up) makes that row's total NaN.
"""

from __future__ import annotations

import pandas as pd

from config import Config


def active_weights(cfg: Config) -> dict[str, float]:
    weights = {name: cfg.scoring.weights[name] for name in cfg.scoring.enabled_categories}
    total = sum(weights.values())
    return {name: w / total for name, w in weights.items()}


def total_score(scores: pd.DataFrame, cfg: Config) -> pd.Series:
    weights = active_weights(cfg)
    parts = [scores[f"{name}_score"] * w for name, w in weights.items()]
    return pd.concat(parts, axis=1).sum(axis=1, min_count=len(parts)).rename("total_score")
