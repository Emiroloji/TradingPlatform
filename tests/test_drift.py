import numpy as np
import pandas as pd

from model.drift import drift_report, psi


def test_psi_zero_for_same_distribution_and_large_for_shift(cfg):
    rng = np.random.default_rng(0)
    base = pd.Series(rng.normal(size=5000))
    assert psi(base, pd.Series(rng.normal(size=5000)), 10) < 0.02
    assert psi(base, pd.Series(rng.normal(loc=1.5, size=5000)), 10) > cfg.model.drift.psi_alert


def test_psi_handles_binary_features(cfg):
    a = pd.Series([0.0] * 90 + [1.0] * 10)
    assert psi(a, a, 10) < 1e-9
    assert psi(a, pd.Series([0.0] * 40 + [1.0] * 60), 10) > cfg.model.drift.psi_alert


def test_report_alerts(cfg):
    imp = pd.Series({"a": 0.5, "b": 0.3})
    text, alerts = drift_report(pd.Series({"a": 0.4, "b": 0.01}), imp, 0.45, 0.55, cfg)
    assert len(alerts) == 2 and "KAYMA" in text
    _, none = drift_report(pd.Series({"a": 0.01, "b": 0.01}), imp, 0.55, 0.55, cfg)
    assert none == []
