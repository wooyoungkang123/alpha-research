"""Walk-forward modeling: composite rank signal and ridge regression.

Implements two prediction strategies:

- **composite**: weighted sum of pre-specified z-scored features (no fitting).
- **ridge**: scikit-learn Ridge regression retrained on a rolling expanding
  window at a configurable frequency, producing strictly out-of-sample
  predictions for the validation and test splits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .config import ModelConfig, SplitConfig


@dataclass(slots=True)
class ModelArtifact:
    train_end_date: pd.Timestamp
    predict_from_date: pd.Timestamp
    predict_to_date: pd.Timestamp
    model_name: str


def fit_predict(
    feature_panel: pd.DataFrame,
    split_config: SplitConfig,
    model_config: ModelConfig,
) -> pd.DataFrame:
    """Produce strictly out-of-sample predictions via walk-forward validation.

    The dataset is divided into train / validation / test splits by date.
    Ridge models are retrained every ``split_config.retrain_frequency_days``
    trading days on an expanding window of training data. Predictions are only
    generated for validation and test dates — no training-period rows are
    returned, preventing look-ahead bias.

    Parameters
    ----------
    feature_panel:
        Feature panel as returned by :func:`~alpha_research.features.generate_features`.
    split_config:
        Date boundaries and retrain cadence.
    model_config:
        Feature columns, target column, ridge alpha, and model type selection.

    Returns
    -------
    pd.DataFrame
        Rows for validation/test dates with columns ``prediction``,
        ``prediction_composite``, ``prediction_ridge``, ``split``, and the
        original feature columns.

    Raises
    ------
    ValueError
        If no rows remain after dropping NaNs on features and target.
    """
    frame = feature_panel.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["date", "ticker"]).reset_index(drop=True)
    evaluation_frame = frame.dropna(subset=model_config.feature_columns + [model_config.target_column]).copy()
    if evaluation_frame.empty:
        raise ValueError("No feature rows remain after dropping missing data.")

    evaluation_frame["split"] = np.where(
        evaluation_frame["date"] >= pd.Timestamp(split_config.test_start),
        "test",
        np.where(evaluation_frame["date"] >= pd.Timestamp(split_config.validation_start), "validation", "train"),
    )
    evaluation_frame["prediction_composite"] = _composite_signal(evaluation_frame, model_config)
    evaluation_frame["prediction_ridge"] = np.nan

    unique_dates = sorted(date for date in evaluation_frame["date"].unique() if date >= np.datetime64(split_config.validation_start))
    retrain_idx = list(range(0, len(unique_dates), split_config.retrain_frequency_days))
    model = None
    active_window_end = None

    for offset, start_idx in enumerate(retrain_idx):
        retrain_date = pd.Timestamp(unique_dates[start_idx])
        predict_dates = unique_dates[start_idx : start_idx + split_config.retrain_frequency_days]
        train_mask = evaluation_frame["date"] < retrain_date
        train_frame = evaluation_frame.loc[train_mask]
        if len(train_frame) < split_config.min_train_observations:
            continue

        model = Ridge(alpha=model_config.ridge_alpha)
        model.fit(train_frame[model_config.feature_columns], train_frame[model_config.target_column])
        active_window_end = pd.Timestamp(predict_dates[-1])

        predict_mask = evaluation_frame["date"].isin(predict_dates)
        evaluation_frame.loc[predict_mask, "prediction_ridge"] = model.predict(
            evaluation_frame.loc[predict_mask, model_config.feature_columns]
        )

    prediction_column = "prediction_ridge" if model_config.model_type == "ridge" else "prediction_composite"
    predictions = evaluation_frame.loc[evaluation_frame["split"].isin(["validation", "test"])].copy()
    predictions["prediction"] = predictions[prediction_column]
    predictions["prediction_model"] = model_config.model_type
    predictions["active_window_end"] = active_window_end
    selected_columns = [
        "date",
        "ticker",
        "sector",
        "beta_60d",
        "benchmark_return",
        "daily_return",
        model_config.target_column,
        "prediction",
        "prediction_composite",
        "prediction_ridge",
        "prediction_model",
        "split",
    ]
    selected_columns.extend(
        [
            column
            for column in model_config.feature_columns
            if column in predictions.columns and column not in selected_columns
        ]
    )
    return predictions[selected_columns].dropna(subset=["prediction"])


def _composite_signal(frame: pd.DataFrame, config: ModelConfig) -> pd.Series:
    """Compute a weighted linear combination of pre-specified features.

    Weights come from ``config.composite_signal_weights`` (feature_name -> float).
    Missing features are silently skipped; NaN values are filled with 0.
    """
    signal = pd.Series(0.0, index=frame.index, dtype=float)
    for feature_name, weight in config.composite_signal_weights.items():
        if feature_name not in frame.columns:
            continue
        signal = signal + weight * frame[feature_name].fillna(0.0)
    return signal
