from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_research.config import ResearchConfig
from alpha_research.data import build_dataset
from alpha_research.features import generate_features


def test_forward_target_matches_future_price_ratio() -> None:
    config = ResearchConfig()
    config.data.source = "synthetic"
    config.data.start_date = "2020-01-01"
    config.data.end_date = "2021-12-31"

    panel = build_dataset(config.data)
    features = generate_features(panel, config.features)
    ticker_frame = features.loc[features["ticker"] == features["ticker"].iloc[0]].reset_index(drop=True)

    row_idx = 120
    expected = ticker_frame.loc[row_idx + 5, "adj_close"] / ticker_frame.loc[row_idx, "adj_close"] - 1.0
    observed = ticker_frame.loc[row_idx, "target_return_5d"]

    assert np.isclose(observed, expected, atol=1e-12)


def test_cross_sectional_zscores_are_centered_by_date() -> None:
    config = ResearchConfig()
    config.data.source = "synthetic"
    config.data.start_date = "2020-01-01"
    config.data.end_date = "2021-12-31"

    panel = build_dataset(config.data)
    features = generate_features(panel, config.features)
    sample_date = pd.Timestamp("2021-06-01")
    sample = features.loc[features["date"] == sample_date, "momentum_20d_z"].dropna()

    assert len(sample) > 10
    assert abs(sample.mean()) < 1e-8
