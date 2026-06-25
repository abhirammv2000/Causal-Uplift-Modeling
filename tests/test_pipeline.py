"""Smoke + correctness tests for the end-to-end pipeline and data utilities."""

import numpy as np
import pytest

from uplift import config, data, pipeline
from uplift.config import ExperimentConfig


def test_stratified_sample_preserves_cells(synthetic_df):
    s = data.stratified_sample(synthetic_df, 2000, random_state=0)
    assert 1500 < len(s) < 2500
    # treatment ratio preserved within a few points
    assert abs(s[config.TREATMENT_COL].mean()
               - synthetic_df[config.TREATMENT_COL].mean()) < 0.05


def test_split_is_stratified(synthetic_df):
    cfg = ExperimentConfig(outcome=config.VISIT_COL)
    X_tr, X_te, y_tr, y_te, t_tr, t_te = data.train_test_split_uplift(synthetic_df, cfg)
    assert len(X_tr) + len(X_te) == len(synthetic_df)
    assert abs(t_tr.mean() - t_te.mean()) < 0.02


def test_run_experiment(synthetic_df):
    cfg = ExperimentConfig(outcome=config.VISIT_COL, base_learner="gbdt", test_size=0.3)
    result = pipeline.run_experiment(synthetic_df, cfg, model_names=["t-learner", "x-learner"])
    assert list(result.metrics.index) == sorted(
        result.metrics.index, key=lambda m: -result.metrics.loc[m, "qini"]
    )
    assert {"qini", "auuc"}.issubset(result.metrics.columns)
    assert np.isfinite(result.ate)


def test_config_validation():
    with pytest.raises(ValueError):
        ExperimentConfig(outcome="not_a_column")
    with pytest.raises(ValueError):
        ExperimentConfig(test_size=1.5)
