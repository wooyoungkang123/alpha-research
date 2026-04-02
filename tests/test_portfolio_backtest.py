from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_research.backtest import run_backtest
from alpha_research.config import BacktestConfig, PortfolioConfig
from alpha_research.portfolio import construct_portfolio


def test_sector_and_beta_neutral_portfolio_construction() -> None:
    predictions = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02")] * 8,
            "ticker": ["A", "B", "C", "D", "E", "F", "G", "H"],
            "sector": ["Tech", "Tech", "Health", "Health", "Tech", "Tech", "Health", "Health"],
            "beta_60d": [1.4, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6],
            "prediction": [0.9, 0.8, 0.7, 0.6, -0.6, -0.7, -0.8, -0.9],
            "split": ["test"] * 8,
        }
    )
    config = PortfolioConfig(long_quantile=0.5, short_quantile=0.5, rebalance_frequency_days=1, min_names_per_side=2)

    weights = construct_portfolio(predictions, exposures=None, portfolio_config=config)

    assert np.isclose(weights["weight"].sum(), 0.0, atol=1e-8)
    sector_exposure = weights.groupby("sector")["weight"].sum()
    assert np.allclose(sector_exposure.to_numpy(), 0.0, atol=1e-8)
    beta_exposure = (weights["weight"] * weights["beta_60d"]).sum()
    assert abs(beta_exposure) < 1e-8


def test_transaction_costs_reduce_net_returns() -> None:
    weights = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-03")],
            "ticker": ["A", "B", "A", "B"],
            "weight": [0.5, -0.5, -0.5, 0.5],
            "prediction": [1.0, -1.0, -1.0, 1.0],
            "sector": ["Tech", "Tech", "Tech", "Tech"],
            "beta_60d": [1.0, 1.0, 1.0, 1.0],
            "split": ["test", "test", "test", "test"],
        }
    )
    returns = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-03")],
            "ticker": ["A", "B", "A", "B"],
            "daily_return": [0.01, -0.01, -0.01, 0.01],
            "benchmark_return": [0.0, 0.0, 0.0, 0.0],
        }
    )
    config = BacktestConfig(transaction_cost_bps=100.0)

    result = run_backtest(weights, returns, config)

    assert (result.daily_results["net_return"] <= result.daily_results["gross_return"]).all()
    assert result.metrics.loc[0, "average_turnover"] > 0
