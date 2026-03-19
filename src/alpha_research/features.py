"""Cross-sectional feature engineering for the alpha research pipeline.

Computes momentum, reversal, volatility, volume, and beta-based signals on a
daily equity panel, then applies leakage-safe forward-return targets. All
features are cross-sectionally winsorized and z-scored per date to remove
market-level effects before modeling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FeatureConfig


def generate_features(panel: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Engineer cross-sectional features and forward-return targets from a raw panel.

    Parameters
    ----------
    panel:
        Raw daily panel as returned by :func:`~alpha_research.data.build_dataset`.
        Must contain columns ``ticker``, ``date``, ``adj_close``, ``volume``,
        ``dollar_volume``, ``daily_return``, ``benchmark_return``, and
        ``is_benchmark``.
    config:
        Feature configuration controlling volatility window, target horizons,
        rank features, and winsorize quantiles.

    Returns
    -------
    pd.DataFrame
        The equity-only subset of ``panel`` augmented with raw feature columns,
        cross-sectionally z-scored ``*_z`` variants, optional rank-scaled
        ``*_rank`` variants, and forward-return target columns.
    """
    equity_panel = panel.loc[~panel["is_benchmark"]].copy()
    equity_panel = equity_panel.sort_values(["ticker", "date"]).reset_index(drop=True)

    grouped = equity_panel.groupby("ticker", observed=True)
    returns = grouped["adj_close"].pct_change()
    equity_panel["reversal_1d"] = -returns
    equity_panel["reversal_5d"] = -grouped["adj_close"].pct_change(5)
    equity_panel["momentum_20d"] = grouped["adj_close"].pct_change(20)
    equity_panel["momentum_60d"] = grouped["adj_close"].pct_change(60)
    realized_vol = grouped["daily_return"].rolling(config.volatility_window).std().reset_index(level=0, drop=True)
    equity_panel["realized_vol_20d"] = realized_vol
    equity_panel["vol_adj_momentum_20d"] = equity_panel["momentum_20d"] / realized_vol.replace(0.0, np.nan)

    rolling_volume = grouped["volume"].rolling(20).mean().reset_index(level=0, drop=True)
    rolling_dollar = grouped["dollar_volume"].rolling(20).mean().reset_index(level=0, drop=True)
    equity_panel["abnormal_volume_20d"] = equity_panel["volume"] / rolling_volume.replace(0.0, np.nan) - 1.0
    equity_panel["turnover_ratio_20d"] = equity_panel["dollar_volume"] / rolling_dollar.replace(0.0, np.nan) - 1.0

    beta_frame = _compute_rolling_beta_features(equity_panel, config.benchmark_window)
    equity_panel = equity_panel.merge(beta_frame, on=["date", "ticker"], how="left")

    rolling_std = grouped["daily_return"].rolling(20).std().reset_index(level=0, drop=True)
    extreme_move = grouped["daily_return"].shift(1).abs() > (2.0 * rolling_std.shift(1))
    prior_direction = -np.sign(grouped["daily_return"].shift(1)).fillna(0.0)
    equity_panel["extreme_move_reversal_flag"] = extreme_move.astype(float) * prior_direction

    for horizon in config.target_horizons:
        equity_panel[f"target_return_{horizon}d"] = (
            grouped["adj_close"].shift(-horizon) / equity_panel["adj_close"] - 1.0
        )
    equity_panel["target_top_quintile_5d"] = _cross_sectional_top_quintile(
        equity_panel, "target_return_5d"
    )

    feature_columns = [
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
    equity_panel = _cross_sectional_preprocess(
        equity_panel,
        feature_columns=feature_columns,
        rank_features=config.rank_features,
        winsorize_quantiles=config.winsorize_quantiles,
    )
    return equity_panel


def _compute_rolling_beta_features(panel: pd.DataFrame, window: int) -> pd.DataFrame:
    """Compute rolling OLS beta, beta instability, and idiosyncratic volatility.

    For each ticker, regresses daily returns on benchmark returns over a rolling
    ``window``-day window to produce:

    - ``beta_60d``: rolling market beta (covariance / benchmark variance)
    - ``beta_instability_20d``: 20-day rolling std of ``beta_60d``
    - ``idio_vol_60d``: rolling std of the market-residual return
    """
    frame = panel[["date", "ticker", "daily_return", "benchmark_return"]].copy()

    def per_ticker(group: pd.DataFrame) -> pd.DataFrame:
        ticker = group.name
        cov = group["daily_return"].rolling(window).cov(group["benchmark_return"])
        bench_var = group["benchmark_return"].rolling(window).var()
        beta = cov / bench_var.replace(0.0, np.nan)
        residual = group["daily_return"] - beta * group["benchmark_return"]
        return pd.DataFrame(
            {
                "date": group["date"].to_numpy(),
                "ticker": np.repeat(ticker, len(group)),
                "beta_60d": beta.to_numpy(),
                "beta_instability_20d": beta.rolling(20).std().to_numpy(),
                "idio_vol_60d": residual.rolling(window).std().to_numpy(),
            },
            index=group.index,
        )

    computed = (
        frame.groupby("ticker", observed=True, group_keys=False)
        .apply(per_ticker, include_groups=False)
        .reset_index(drop=True)
    )
    return computed[["date", "ticker", "beta_60d", "beta_instability_20d", "idio_vol_60d"]]


def _cross_sectional_top_quintile(panel: pd.DataFrame, column: str) -> pd.Series:
    """Return a float flag (1.0 / 0.0) marking the top-quintile stocks per date."""
    return (
        panel.groupby("date", observed=True)[column]
        .transform(lambda series: (series.rank(pct=True, method="average") >= 0.80).astype(float))
    )


def _cross_sectional_preprocess(
    panel: pd.DataFrame,
    feature_columns: list[str],
    rank_features: list[str],
    winsorize_quantiles: tuple[float, float],
) -> pd.DataFrame:
    """Winsorize, z-score, and optionally rank-scale features cross-sectionally.

    For every date independently:
    1. Winsorize each feature to ``winsorize_quantiles`` to clip outliers.
    2. Z-score to zero mean and unit std (``feature_z`` columns).
    3. For features in ``rank_features``, also add a rank-scaled column in
       ``[-1, 1]`` (``feature_rank`` columns).
    """
    processed = panel.copy()
    lower_q, upper_q = winsorize_quantiles

    for feature in feature_columns:
        processed[feature] = processed.groupby("date", observed=True)[feature].transform(
            lambda series: _winsorize(series, lower_q, upper_q)
        )
        processed[f"{feature}_z"] = processed.groupby("date", observed=True)[feature].transform(_zscore)
        if feature in rank_features:
            processed[f"{feature}_rank"] = processed.groupby("date", observed=True)[feature].transform(
                _rank_to_unit_interval
            )
    return processed


def _winsorize(series: pd.Series, lower_q: float, upper_q: float) -> pd.Series:
    """Clip a series to its ``lower_q`` and ``upper_q`` quantiles. Returns unchanged if fewer than 5 non-null values."""
    if series.notna().sum() < 5:
        return series
    lower = series.quantile(lower_q)
    upper = series.quantile(upper_q)
    return series.clip(lower=lower, upper=upper)


def _zscore(series: pd.Series) -> pd.Series:
    """Return population z-scores (ddof=0). Returns zeros if std is 0 or NaN."""
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def _rank_to_unit_interval(series: pd.Series) -> pd.Series:
    """Map percentile ranks to ``[-1, 1]`` via ``2 * (rank_pct - 0.5)``."""
    ranks = series.rank(method="average", pct=True)
    return 2.0 * (ranks - 0.5)
