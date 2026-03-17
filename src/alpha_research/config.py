from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import tomllib


DEFAULT_UNIVERSE = [
    "AAPL", "ABBV", "ABT", "AMGN", "AMZN", "AVGO", "AXP", "BA", "BAC", "CAT",
    "COST", "CRM", "CSCO", "CVX", "DIS", "GOOGL", "GS", "HD", "HON", "IBM",
    "INTC", "JNJ", "JPM", "KO", "LIN", "LLY", "LOW", "MA", "MCD", "META",
    "MMM", "MRK", "MS", "MSFT", "NEE", "NFLX", "NKE", "NVDA", "ORCL", "PEP",
    "PFE", "PG", "QCOM", "RTX", "SPY", "T", "TMO", "TSLA", "UNH", "UNP",
    "UPS", "USB", "V", "VZ", "WFC", "WMT", "XOM",
]


@dataclass(slots=True)
class DataConfig:
    source: str = "yfinance"
    start_date: str = "2018-01-01"
    end_date: str = "2024-12-31"
    benchmark: str = "SPY"
    tickers: list[str] = field(default_factory=lambda: [ticker for ticker in DEFAULT_UNIVERSE if ticker != "SPY"])
    cache_dir: str = "data/cache"
    min_history_days: int = 252
    min_price: float = 5.0
    min_median_dollar_volume: float = 10_000_000.0
    synthetic_seed: int = 7
    synthetic_tickers: int = 60


@dataclass(slots=True)
class FeatureConfig:
    winsorize_quantiles: tuple[float, float] = (0.02, 0.98)
    rank_features: list[str] = field(
        default_factory=lambda: [
            "reversal_1d",
            "reversal_5d",
            "momentum_20d",
            "momentum_60d",
            "vol_adj_momentum_20d",
            "abnormal_volume_20d",
            "turnover_ratio_20d",
            "beta_60d",
            "beta_instability_20d",
            "idio_vol_60d",
            "extreme_move_reversal_flag",
        ]
    )
    target_horizons: list[int] = field(default_factory=lambda: [5, 10])
    benchmark_window: int = 60
    volatility_window: int = 20


@dataclass(slots=True)
class SplitConfig:
    validation_start: str = "2022-01-03"
    test_start: str = "2023-01-03"
    retrain_frequency_days: int = 21
    min_train_observations: int = 252


@dataclass(slots=True)
class ModelConfig:
    model_type: str = "ridge"
    target_column: str = "target_return_5d"
    feature_columns: list[str] = field(
        default_factory=lambda: [
            "reversal_1d_rank",
            "reversal_5d_rank",
            "momentum_20d_rank",
            "momentum_60d_rank",
            "vol_adj_momentum_20d_rank",
            "abnormal_volume_20d_rank",
            "turnover_ratio_20d_rank",
            "beta_60d_z",
            "beta_instability_20d_z",
            "idio_vol_60d_z",
            "extreme_move_reversal_flag_rank",
        ]
    )
    ridge_alpha: float = 2.0
    composite_signal_weights: dict[str, float] = field(
        default_factory=lambda: {
            "momentum_20d_rank": 0.20,
            "momentum_60d_rank": 0.25,
            "vol_adj_momentum_20d_rank": 0.15,
            "reversal_5d_rank": -0.15,
            "abnormal_volume_20d_rank": 0.10,
            "idio_vol_60d_z": -0.10,
            "extreme_move_reversal_flag_rank": 0.05,
        }
    )


@dataclass(slots=True)
class PortfolioConfig:
    long_quantile: float = 0.10
    short_quantile: float = 0.10
    rebalance_frequency_days: int = 5
    sector_neutral: bool = True
    beta_neutral: bool = True
    min_names_per_side: int = 5
    beta_column: str = "beta_60d"
    sector_column: str = "sector"


@dataclass(slots=True)
class BacktestConfig:
    transaction_cost_bps: float = 10.0
    annualization_factor: int = 252
    benchmark_column: str = "benchmark_return"
    regime_vol_window: int = 20


@dataclass(slots=True)
class ResearchConfig:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)


def _merge_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    for key, value in values.items():
        if hasattr(instance, key):
            setattr(instance, key, value)
    return instance


def load_config(path: str | Path) -> ResearchConfig:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    config = ResearchConfig()
    for section_name, dataclass_type in (
        ("data", config.data),
        ("features", config.features),
        ("split", config.split),
        ("model", config.model),
        ("portfolio", config.portfolio),
        ("backtest", config.backtest),
    ):
        if section_name in raw:
            _merge_dataclass(dataclass_type, raw[section_name])
    return config
