from __future__ import annotations

from alpha_research.config import ResearchConfig
from alpha_research.data import build_dataset
from alpha_research.features import generate_features
from alpha_research.modeling import fit_predict


def test_walk_forward_predictions_start_after_validation_boundary() -> None:
    config = ResearchConfig()
    config.data.source = "synthetic"
    config.data.start_date = "2020-01-01"
    config.data.end_date = "2024-12-31"

    panel = build_dataset(config.data)
    features = generate_features(panel, config.features)
    predictions = fit_predict(features, config.split, config.model)

    assert not predictions.empty
    assert predictions["date"].min().strftime("%Y-%m-%d") >= config.split.validation_start
    assert set(predictions["split"].unique()) <= {"validation", "test"}
