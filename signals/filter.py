"""Conservative-mode signal rules (KURALLAR §2): a stock is reported only if ALL hold.

  1. liquidity threshold                     (features.liquidity_ok)
  2. allowed market regime                   (signal.allowed_regimes)
  3. total score >= threshold
  4. accumulation score >= threshold
  5. ML probability >= threshold             - only with a model approved for live use (KURALLAR §5);
                                               without one the rule-based system runs alone (FAZLAR Faz 3)
  6. Gemini did not veto                     - applied after this filter, on candidates only (MIMARI §3)
  7. this setup's out-of-sample backtest has >= min_backtest_trades trades and positive expectancy
     measured against the benchmark (KURALLAR §4)
"""

from __future__ import annotations

from collections import OrderedDict

import pandas as pd

from config import Config


def rule_based_mask(features: pd.DataFrame, cfg: Config) -> pd.Series:
    s = cfg.signal
    return (
        features["liquidity_ok"].astype(bool)
        & features["regime"].isin(s.allowed_regimes)
        & (features["total_score"] >= s.min_total_score)
        & (features["accumulation_score"] >= s.min_accumulation_score)
    )


def rule_based_signals(features: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    return features[rule_based_mask(features, cfg)]


def ml_signals(features: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Rule-based signals that the model also backs (`prob` column required)."""
    return features[rule_based_mask(features, cfg) & (features["prob"] >= cfg.signal.min_probability)]


# Observation-mode setups (not live signals): extra conditions on top of the rule-based baseline.
OBSERVATION_RULES = {
    # 13F: more institutions AND more (split-adjusted) shares than last quarter
    "kurumsal_toplama": lambda df: (df["inst_filers_chg"] > 0) & (df["inst_shares_chg"] > 0),
}


def observation_candidates(df: pd.DataFrame, cfg: Config, market: str) -> pd.DataFrame:
    """Rows qualifying for each observation setup of `market`, with a `setup` column.

    Rule 7 (setup track record) is deliberately not applied: observation exists to measure it.
    """
    frames = []
    for name in cfg.signal.observation.get(market, []):
        mask = rule_based_mask(df, cfg) & OBSERVATION_RULES[name](df).fillna(False)
        frames.append(df[mask].assign(setup=name))
    return pd.concat(frames) if frames else df.iloc[0:0].assign(setup=pd.Series(dtype=str))


def setup_ok(setup: dict | None, cfg: Config) -> tuple[bool, str]:
    """Rule 7 on the stored out-of-sample backtest summary of the setup (None = never backtested)."""
    if not setup:
        return False, "kurulumun backtest sonucu yok"
    if setup["trades"] < cfg.signal.min_backtest_trades:
        return False, f"yetersiz veri ({setup['trades']} işlem < {cfg.signal.min_backtest_trades})"
    if not setup["expectancy_excess"] > 0:
        x = setup["expectancy_excess"]
        shown = f"{'+' if x >= 0 else '-'}%{abs(x) * 100:.2f}".replace(".", ",")
        return False, f"kurulumun endekse göre OOS expectancy'si pozitif değil ({shown})"
    return True, "kurulumun OOS expectancy'si pozitif"


def rule_steps(df: pd.DataFrame, cfg: Config, ml_approved: bool) -> "OrderedDict[str, pd.Series]":
    s = cfg.signal
    steps: OrderedDict[str, pd.Series] = OrderedDict()
    steps["likidite"] = df["liquidity_ok"].astype(bool)
    steps["rejim"] = df["regime"].isin(s.allowed_regimes)
    steps[f"toplam skor ≥ {s.min_total_score:.0f}"] = df["total_score"] >= s.min_total_score
    steps[f"toplama skoru ≥ {s.min_accumulation_score:.0f}"] = df["accumulation_score"] >= s.min_accumulation_score
    if ml_approved:
        steps[f"ML olasılığı ≥ {s.min_probability}"] = df["prob"] >= s.min_probability
    return steps


def apply_conservative_filter(
    df: pd.DataFrame, cfg: Config, setup: dict | None = None, ml_approved: bool = False
) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    """MIMARI §7: candidates passing rules 1-5 and 7, plus the funnel (step, remaining count).

    Rule 6 (Gemini veto) runs afterwards on these candidates only.
    """
    remaining = pd.Series(True, index=df.index)
    funnel = [("evren", len(df))]
    for name, mask in rule_steps(df, cfg, ml_approved).items():
        remaining &= mask
        funnel.append((name, int(remaining.sum())))
    ok, reason = setup_ok(setup, cfg)
    if not ok:
        remaining &= False
    funnel.append((f"kurulum backtest: {reason}", int(remaining.sum())))
    return df[remaining], funnel
