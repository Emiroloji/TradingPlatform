"""Typed loader for config.yaml (the `Config` type used by module interfaces)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MarketConfig(_Section):
    enabled: bool
    universe: str
    benchmark: str
    suffix: str
    currency: str
    fx: str | None
    news_locale: str


class QualityConfig(_Section):
    max_missing_ratio: float
    max_zero_volume_ratio: float
    max_abs_daily_return: float
    jump_excludes: dict[str, bool]
    exclude_on_issues: bool


class DataConfig(_Section):
    history_years: int
    interval: str
    storage_dir: str
    quality: QualityConfig


class LiquidityConfig(_Section):
    min_avg_turnover_usd: dict[str, float]
    lookback_days: int


class TrendFeatures(_Section):
    ema_periods: list[int]
    adx_period: int
    adx_threshold: float


class MomentumFeatures(_Section):
    rsi_period: int
    rsi_bull_min: float
    rsi_overbought: float
    macd_fast: int
    macd_slow: int
    macd_signal: int
    stoch_k: int
    stoch_d: int
    stoch_smooth_k: int
    stoch_overbought: float
    roc_period: int


class VolumeFeatures(_Section):
    obv_ema_period: int
    cmf_period: int
    avg_volume_period: int
    volume_ratio_min: float


class VolatilityFeatures(_Section):
    bb_period: int
    bb_std: float
    baseline_days: int


class AccumulationFeatures(_Section):
    window_days: int
    baseline_days: int
    range_contraction_ratio: float
    volume_increase_ratio: float
    divergence_max_price_return: float
    spring_lookback_days: int
    score_by_conditions: list[float]


class RegimeFeatures(_Section):
    ema_period: int
    slope_days: int
    adx_period: int
    adx_min: float


class RelativeFeatures(_Section):
    periods: list[int]


class FeaturesConfig(_Section):
    trend: TrendFeatures
    momentum: MomentumFeatures
    volume: VolumeFeatures
    volatility: VolatilityFeatures
    accumulation: AccumulationFeatures
    regime: RegimeFeatures
    relative: RelativeFeatures


class ScoringConfig(_Section):
    enabled_categories: list[str]
    weights: dict[str, float]


class SignalConfig(_Section):
    mode: str
    min_total_score: float
    min_accumulation_score: float
    min_probability: float
    allowed_regimes: list[str]
    min_backtest_trades: int


class LabelConfig(_Section):
    type: Literal["trade_outcome", "triple_barrier"]
    min_trade_excess: float
    horizon_days: int
    target_excess_return: float
    use_stop_barrier: bool


class ModelConfig(_Section):
    type: str
    train_years: int
    test_months: int
    step_months: int
    first_test_start: str
    params: dict[str, float | int | str]
    min_fold_win_ratio: float


class BacktestConfig(_Section):
    commission: float
    slippage: float
    entry: str
    target_r_multiple: float
    max_holding_days: int
    oos_start: str


class RiskConfig(_Section):
    capital: float
    risk_per_trade: float
    atr_period: int
    atr_stop_multiplier: float
    max_positions: int
    max_per_sector: int


class GeminiConfig(_Section):
    model: str
    fallback_models: list[str]
    timeout_seconds: int
    max_retries: int
    retry_backoff_seconds: float
    cache_days: int
    temperature: float
    veto_categories: list[str]
    banned_phrases: list[str]


class NewsConfig(_Section):
    sources: list[Literal["yfinance", "google_news_rss"]]
    lookback_days: int
    max_items: int
    timeout_seconds: int
    google_news_rss_url: str
    query_template: str


class TelegramConfig(_Section):
    max_message_chars: int
    timeout_seconds: int


class ScheduleConfig(_Section):
    bist_run_time: str
    us_run_time: str
    timezone: str


class Config(_Section):
    markets: dict[str, MarketConfig]
    data: DataConfig
    liquidity: LiquidityConfig
    features: FeaturesConfig
    scoring: ScoringConfig
    signal: SignalConfig
    label: LabelConfig
    model: ModelConfig
    backtest: BacktestConfig
    risk: RiskConfig
    gemini: GeminiConfig
    news: NewsConfig
    telegram: TelegramConfig
    schedule: ScheduleConfig

    @property
    def root(self) -> Path:
        return DEFAULT_CONFIG_PATH.parent

    @property
    def storage_path(self) -> Path:
        return self.root / self.data.storage_dir

    def enabled_markets(self) -> dict[str, MarketConfig]:
        return {name: m for name, m in self.markets.items() if m.enabled}


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Config:
    with open(path, encoding="utf-8") as f:
        return Config.model_validate(yaml.safe_load(f))
