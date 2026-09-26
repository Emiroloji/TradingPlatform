"""Model drift monitoring: feature distribution shift (PSI) and performance decay.

PSI compares each feature's recent distribution with the model's training window, using
quantile bins of the training data. Performance compares the AUC on the most recent labelled
rows with the walk-forward out-of-sample AUCs recorded for the model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import Config

_EPS = 1e-6


def psi(expected: pd.Series, actual: pd.Series, bins: int) -> float:
    """Population stability index of `actual` vs `expected` (NaNs dropped)."""
    e, a = expected.dropna().to_numpy(), actual.dropna().to_numpy()
    if len(e) == 0 or len(a) == 0:
        return float("nan")
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # (near-)constant feature: compare the share of the dominant value instead
        v = pd.Series(e).mode().iloc[0]
        pe, pa = max((e == v).mean(), _EPS), max((a == v).mean(), _EPS)
        return float((pa - pe) * np.log(pa / pe) + ((1 - pa) - (1 - pe)) * np.log(max(1 - pa, _EPS) / max(1 - pe, _EPS)))
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(e, edges)[0] / len(e)
    pa = np.histogram(a, edges)[0] / len(a)
    pe, pa = np.clip(pe, _EPS, None), np.clip(pa, _EPS, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def feature_drift(train: pd.DataFrame, recent: pd.DataFrame, features: list[str], cfg: Config) -> pd.Series:
    return pd.Series({f: psi(train[f], recent[f], cfg.model.drift.psi_bins) for f in features}).sort_values(ascending=False)


def recent_auc(y: pd.Series, prob: np.ndarray) -> float:
    return float(roc_auc_score(y, prob)) if y.nunique() == 2 else float("nan")


def drift_report(psi_values: pd.Series, importance: pd.Series, auc_recent: float, auc_oos: float, cfg: Config) -> tuple[str, list[str]]:
    """Markdown report and the list of alert messages (empty = no action needed)."""
    d = cfg.model.drift
    top = importance.head(d.top_features).index
    shown = psi_values.reindex(top)
    alerts = [f"{f}: PSI {v:.2f}" for f, v in shown.items() if v > d.psi_alert]
    if not np.isnan(auc_recent) and auc_recent < auc_oos - d.auc_drop_alert:
        alerts.append(f"AUC düştü: son dönem {auc_recent:.3f}, OOS ortalaması {auc_oos:.3f}")
    lines = [
        "## Özellik kayması (PSI, en önemli özellikler)", "",
        f"Eşik: PSI > {d.psi_alert} önemli kayma.", "", "| Özellik | PSI | Durum |", "|---|---|---|",
        *[f"| {f} | {v:.3f} | {'KAYMA' if v > d.psi_alert else 'normal'} |" for f, v in shown.items()],
        "", "## Performans", "",
        f"- Walk-forward OOS AUC ortalaması: {auc_oos:.3f}",
        f"- Son etiketli dönem AUC: {'—' if np.isnan(auc_recent) else f'{auc_recent:.3f}'}",
        "", f"**Uyarılar:** {'; '.join(alerts) if alerts else 'yok'}",
    ]
    return "\n".join(lines) + "\n", alerts
