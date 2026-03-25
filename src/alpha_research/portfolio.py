"""Sector-neutral, beta-neutral long/short portfolio construction.

Selects top/bottom quantile stocks by predicted return, assigns sector-budgeted
weights, then projects out residual sector and beta exposures via least-squares
constraint enforcement. Portfolios are rebalanced at a configurable frequency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PortfolioConfig


def construct_portfolio(
    predictions: pd.DataFrame,
    exposures: pd.DataFrame | None,
    portfolio_config: PortfolioConfig,
) -> pd.DataFrame:
    """Build a daily rebalanced long/short portfolio from model predictions.

    Parameters
    ----------
    predictions:
        Output of :func:`~alpha_research.modeling.fit_predict`. Must contain
        ``date``, ``ticker``, ``prediction``, ``sector``, and ``beta_60d``.
    exposures:
        Optional override frame for sector and beta columns. When provided,
        its ``sector`` / ``beta_60d`` values replace those in ``predictions``.
    portfolio_config:
        Long/short quantile thresholds, rebalance frequency, neutrality flags,
        and minimum names per side.

    Returns
    -------
    pd.DataFrame
        Columns: ``date``, ``ticker``, ``weight``, ``prediction``, ``sector``,
        ``beta_60d``, ``split``. Weights are normalized to gross exposure of 1
        (0.5 long / 0.5 short convention after normalization).
    """
    frame = predictions.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    if exposures is not None:
        keep = [column for column in [portfolio_config.sector_column, portfolio_config.beta_column] if column in exposures.columns]
        if keep:
            frame = frame.drop(columns=[column for column in keep if column in frame.columns], errors="ignore").merge(
                exposures[["date", "ticker", *keep]].drop_duplicates(["date", "ticker"]),
                on=["date", "ticker"],
                how="left",
            )

    rebalance_dates = _select_rebalance_dates(frame["date"].drop_duplicates().sort_values(), portfolio_config.rebalance_frequency_days)
    weights: list[pd.DataFrame] = []

    for rebalance_date in rebalance_dates:
        daily = frame.loc[frame["date"] == rebalance_date].copy()
        daily = daily.dropna(subset=["prediction"])
        if len(daily) < portfolio_config.min_names_per_side * 2:
            continue

        daily["prediction_rank"] = daily["prediction"].rank(method="first", pct=True)
        long_book = daily.loc[daily["prediction_rank"] >= 1.0 - portfolio_config.long_quantile].copy()
        short_book = daily.loc[daily["prediction_rank"] <= portfolio_config.short_quantile].copy()
        if len(long_book) < portfolio_config.min_names_per_side or len(short_book) < portfolio_config.min_names_per_side:
            continue

        if portfolio_config.sector_neutral:
            long_weights, short_weights = _sector_neutral_weights(
                long_book,
                short_book,
                sector_column=portfolio_config.sector_column,
            )
        else:
            long_weights = pd.Series(1.0 / len(long_book), index=long_book.index)
            short_weights = pd.Series(-1.0 / len(short_book), index=short_book.index)

        long_book["weight"] = long_weights
        short_book["weight"] = short_weights
        combined = pd.concat([long_book, short_book], ignore_index=True)
        combined["weight"] = _neutralize_weights(
            combined,
            sector_column=portfolio_config.sector_column,
            beta_column=portfolio_config.beta_column,
            enforce_sector_neutral=portfolio_config.sector_neutral,
            enforce_beta_neutral=portfolio_config.beta_neutral,
        )
        weights.append(combined)

    if not weights:
        return pd.DataFrame(columns=["date", "ticker", "weight", "prediction", "sector", "beta_60d"])

    result = pd.concat(weights, ignore_index=True)
    return result[["date", "ticker", "weight", "prediction", "sector", "beta_60d", "split"]]


def _select_rebalance_dates(dates: pd.Series, frequency_days: int) -> list[pd.Timestamp]:
    """Subsample sorted trading dates at every ``frequency_days`` interval."""
    ordered = list(pd.to_datetime(dates))
    return ordered[::frequency_days]


def _sector_neutral_weights(
    long_book: pd.DataFrame,
    short_book: pd.DataFrame,
    sector_column: str,
) -> tuple[pd.Series, pd.Series]:
    """Allocate equal sector budgets across sectors present in both long and short books.

    Each common sector receives ``0.5 / n_common_sectors`` of gross exposure on
    each side. Stocks within a sector receive equal within-sector weights. Falls
    back to uniform equal-weight if no sectors are common to both sides.
    """
    common_sectors = sorted(
        set(long_book[sector_column].dropna().unique()).intersection(short_book[sector_column].dropna().unique())
    )
    if not common_sectors:
        return (
            pd.Series(0.5 / len(long_book), index=long_book.index),
            pd.Series(-0.5 / len(short_book), index=short_book.index),
        )

    long_weights = pd.Series(0.0, index=long_book.index, dtype=float)
    short_weights = pd.Series(0.0, index=short_book.index, dtype=float)
    sector_budget = 0.5 / len(common_sectors)
    for sector in common_sectors:
        long_idx = long_book.index[long_book[sector_column] == sector]
        short_idx = short_book.index[short_book[sector_column] == sector]
        if len(long_idx) == 0 or len(short_idx) == 0:
            continue
        long_weights.loc[long_idx] = sector_budget / len(long_idx)
        short_weights.loc[short_idx] = -sector_budget / len(short_idx)

    if long_weights.abs().sum() == 0 or short_weights.abs().sum() == 0:
        return (
            pd.Series(0.5 / len(long_book), index=long_book.index),
            pd.Series(-0.5 / len(short_book), index=short_book.index),
        )
    return long_weights, short_weights


def _neutralize_weights(
    frame: pd.DataFrame,
    sector_column: str,
    beta_column: str,
    enforce_sector_neutral: bool,
    enforce_beta_neutral: bool,
) -> pd.Series:
    """Project out constraint violations from portfolio weights using least-squares.

    For each active constraint (sector membership vector or beta exposure
    vector), computes the residual violation ``c @ w`` and subtracts the
    minimum-norm correction ``C^T (C C^T)^{-1} violation`` via the
    Moore-Penrose pseudoinverse. Weights are then rescaled to unit gross
    exposure.
    """
    weights = frame["weight"].to_numpy(dtype=float)
    constraints: list[np.ndarray] = []

    if enforce_sector_neutral:
        for sector in sorted(frame[sector_column].dropna().unique()):
            constraints.append((frame[sector_column] == sector).astype(float).to_numpy())
    else:
        constraints.append(np.ones(len(frame), dtype=float))

    if enforce_beta_neutral:
        constraints.append(frame[beta_column].fillna(1.0).to_numpy(dtype=float))

    if constraints:
        matrix = np.vstack(constraints)
        violation = matrix @ weights
        adjustment = matrix.T @ np.linalg.pinv(matrix @ matrix.T) @ violation
        weights = weights - adjustment

    gross = np.abs(weights).sum()
    if gross == 0 or np.isnan(gross):
        return pd.Series(frame["weight"].to_numpy(dtype=float), index=frame.index)
    return pd.Series(weights / gross, index=frame.index)
