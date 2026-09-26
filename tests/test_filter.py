import numpy as np
import pandas as pd

from signals.filter import apply_conservative_filter, rule_based_signals, setup_ok


def test_every_rule_must_hold(cfg):
    s = cfg.signal
    base = {
        "liquidity_ok": True,
        "regime": s.allowed_regimes[0],
        "total_score": s.min_total_score,
        "accumulation_score": s.min_accumulation_score,
    }
    breaks = [
        {"liquidity_ok": False},
        {"regime": "bear"},
        {"regime": np.nan},  # warm-up regime is never allowed
        {"total_score": s.min_total_score - 0.1},
        {"total_score": np.nan},
        {"accumulation_score": s.min_accumulation_score - 1},
    ]
    rows = [base] + [base | b for b in breaks]
    out = rule_based_signals(pd.DataFrame(rows), cfg)
    assert list(out.index) == [0]


GOOD_SETUP = {"trades": 154, "expectancy_excess": 0.004}


def _rows(cfg, n=3, **over):
    s = cfg.signal
    base = {"ticker": "A", "liquidity_ok": True, "regime": s.allowed_regimes[0], "total_score": 90.0,
            "accumulation_score": 80.0, "prob": 0.9}
    return pd.DataFrame([base | over] * n)


def test_full_filter_passes_when_everything_holds(cfg):
    out, funnel = apply_conservative_filter(_rows(cfg), cfg, GOOD_SETUP, ml_approved=True)
    assert len(out) == 3
    assert funnel[0] == ("evren", 3) and funnel[-1][1] == 3


def test_ml_rule_only_with_approved_model(cfg):
    low_prob = _rows(cfg, prob=0.1)
    assert len(apply_conservative_filter(low_prob, cfg, GOOD_SETUP, ml_approved=True)[0]) == 0
    assert len(apply_conservative_filter(low_prob, cfg, GOOD_SETUP, ml_approved=False)[0]) == 3


def test_setup_track_record_gates_everything(cfg):
    for setup in [None, {"trades": 10, "expectancy_excess": 0.05}, {"trades": 154, "expectancy_excess": -0.0003}]:
        out, funnel = apply_conservative_filter(_rows(cfg), cfg, setup)
        assert out.empty
        assert funnel[-1][1] == 0 and "kurulum" in funnel[-1][0]
    assert "pozitif değil" in setup_ok({"trades": 154, "expectancy_excess": -0.0003}, cfg)[1]


def test_observation_setup_needs_baseline_and_13f_rule(cfg):
    from signals.filter import observation_candidates

    rows = _rows(cfg, n=4)
    rows["inst_filers_chg"] = [0.1, 0.1, -0.1, float("nan")]
    rows["inst_shares_chg"] = [0.2, -0.2, 0.2, 0.2]
    out = observation_candidates(rows, cfg, "us")
    assert list(out.index) == [0] and (out["setup"] == "kurumsal_toplama").all()
    assert observation_candidates(rows, cfg, "bist").empty
    # setup track record (rule 7) is not required in observation mode
    assert apply_conservative_filter(rows, cfg, None)[0].empty
