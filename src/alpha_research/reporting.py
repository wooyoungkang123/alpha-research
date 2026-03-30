from __future__ import annotations

import os
from pathlib import Path

_CACHE_ROOT = Path("data/cache")
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from .backtest import BacktestResult


def generate_report(
    result: BacktestResult,
    predictions: pd.DataFrame,
    output_dir: str | Path = "outputs",
    report_path: str | Path = "docs/research_summary.md",
) -> None:
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    _save_equity_curve(result.daily_results, figures_dir / "equity_curve.png")
    _save_long_short_contributions(result.leg_contributions, figures_dir / "leg_contributions.png")

    top_features = _feature_correlation_table(predictions)
    metrics_row = result.metrics.iloc[0]

    report = f"""# Research Summary

## Objective

This project evaluates whether cross-sectional signals derived from daily U.S. equity data can predict relative performance over the next few days and support a market-neutral long/short portfolio.

## What Was Tested

- Signal family: reversal, momentum, volatility, volume, rolling beta, beta instability, idiosyncratic volatility, and extreme-move mean reversion
- Prediction models: rank-composite baseline and ridge regression
- Portfolio: long top decile, short bottom decile, sector-aware construction, beta-neutral scaling, transaction costs applied

## Out-of-Sample Results

- Annualized return: {metrics_row['annualized_return']:.2%}
- Annualized volatility: {metrics_row['annualized_volatility']:.2%}
- Sharpe ratio: {metrics_row['sharpe_ratio']:.2f}
- Max drawdown: {metrics_row['max_drawdown']:.2%}
- Hit rate: {metrics_row['hit_rate']:.2%}
- Average turnover: {metrics_row['average_turnover']:.2f}

## Feature Diagnostics

Top absolute correlations between engineered features and realized returns in the evaluation window:

{top_features.to_markdown(index=False)}

## What Worked

- Cross-sectional ranking and basic preprocessing kept signals comparable across heterogeneous stocks.
- The portfolio construction layer explicitly controlled sector concentration and market beta drift.
- Transaction-cost-aware evaluation made the research output closer to a production-style alpha study.

## Limitations

- Free data introduces survivorship bias, stale sector labels, and occasional corporate-action quirks.
- The default universe is a liquid prototype universe rather than a full institutional coverage list.
- Targets are daily-bar based and do not model intraday execution or borrow constraints.

## Next Steps

- Add premium or point-in-time fundamentals and sector mappings.
- Extend the model set to boosted trees and residualized-return targets.
- Add richer risk-model neutralization and regime-aware allocation.
"""
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(report)


def _feature_correlation_table(predictions: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [column for column in predictions.columns if column.endswith("_rank") or column.endswith("_z")]
    if not feature_columns:
        return pd.DataFrame([{"feature": "n/a", "abs_correlation": 0.0}])
    correlations = []
    for column in feature_columns:
        series = predictions[column]
        target = predictions.get("target_return_5d")
        if target is None:
            continue
        corr = series.corr(target)
        correlations.append({"feature": column, "abs_correlation": abs(corr) if pd.notna(corr) else 0.0})
    if not correlations:
        return pd.DataFrame([{"feature": "n/a", "abs_correlation": 0.0}])
    return pd.DataFrame(correlations).sort_values("abs_correlation", ascending=False).head(5)


def _save_equity_curve(daily_results: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(daily_results["date"], daily_results["cumulative_return"], label="Strategy")
    axis.set_title("Cumulative Net Return")
    axis.set_ylabel("Return")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _save_long_short_contributions(leg_contributions: pd.DataFrame, path: Path) -> None:
    pivot = leg_contributions.pivot(index="date", columns="leg", values="gross_contribution").fillna(0.0)
    figure, axis = plt.subplots(figsize=(10, 4))
    for column in pivot.columns:
        axis.plot(pivot.index, pivot[column].cumsum(), label=column)
    axis.set_title("Cumulative Leg Contribution")
    axis.set_ylabel("Contribution")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
