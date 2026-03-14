"""Public interfaces for the cross-sectional equity alpha research platform."""

from .backtest import BacktestResult, run_backtest
from .config import (
    BacktestConfig,
    DataConfig,
    FeatureConfig,
    ModelConfig,
    PortfolioConfig,
    ResearchConfig,
    SplitConfig,
    load_config,
)
from .data import build_dataset
from .features import generate_features
from .modeling import fit_predict
from .portfolio import construct_portfolio

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "DataConfig",
    "FeatureConfig",
    "ModelConfig",
    "PortfolioConfig",
    "ResearchConfig",
    "SplitConfig",
    "build_dataset",
    "construct_portfolio",
    "fit_predict",
    "generate_features",
    "load_config",
    "run_backtest",
]
