"""Tests for the config module: default values, TOML loading, and field merging."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from alpha_research.config import (
    BacktestConfig,
    DataConfig,
    FeatureConfig,
    ModelConfig,
    PortfolioConfig,
    ResearchConfig,
    SplitConfig,
    load_config,
)


def test_research_config_default_instantiation() -> None:
    config = ResearchConfig()
    assert isinstance(config.data, DataConfig)
    assert isinstance(config.features, FeatureConfig)
    assert isinstance(config.split, SplitConfig)
    assert isinstance(config.model, ModelConfig)
    assert isinstance(config.portfolio, PortfolioConfig)
    assert isinstance(config.backtest, BacktestConfig)


def test_data_config_defaults() -> None:
    config = DataConfig()
    assert config.source == "yfinance"
    assert config.benchmark == "SPY"
    assert config.min_price == 5.0
    assert config.min_history_days == 252


def test_feature_config_winsorize_quantiles_are_valid() -> None:
    config = FeatureConfig()
    lower, upper = config.winsorize_quantiles
    assert 0.0 <= lower < upper <= 1.0


def test_model_config_feature_columns_are_nonempty() -> None:
    config = ModelConfig()
    assert len(config.feature_columns) > 0
    assert config.target_column.startswith("target_")


def test_portfolio_config_quantile_sum_is_lte_one() -> None:
    config = PortfolioConfig()
    assert config.long_quantile + config.short_quantile <= 1.0


def test_load_config_from_synthetic_toml() -> None:
    config = load_config("configs/synthetic.toml")
    assert config.data.source == "synthetic"
    assert config.split.validation_start == "2022-01-03"
    assert config.backtest.transaction_cost_bps == 10.0


def test_load_config_overrides_defaults(tmp_path: Path) -> None:
    toml_content = textwrap.dedent("""\
        [data]
        source = "synthetic"
        start_date = "2021-01-01"
        end_date = "2021-12-31"
        synthetic_tickers = 10

        [backtest]
        transaction_cost_bps = 50.0
    """)
    config_file = tmp_path / "test.toml"
    config_file.write_text(toml_content)
    config = load_config(config_file)

    assert config.data.source == "synthetic"
    assert config.data.synthetic_tickers == 10
    assert config.backtest.transaction_cost_bps == 50.0
    # Unspecified fields should retain defaults
    assert config.data.benchmark == "SPY"


def test_load_config_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nonexistent.toml")
