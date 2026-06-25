"""Tests that each uplift estimator fits, predicts, and recovers signal."""

import numpy as np
import pytest

from uplift import config, evaluation as ev, models
from uplift.config import ExperimentConfig
from uplift.data import make_xyt

MODEL_NAMES = ["s-learner", "t-learner", "x-learner", "class-transform"]

# Small, fast base learners keep the suite quick while preserving signal.
FAST = dict(base_learner="gbdt", max_iter=60)


@pytest.fixture
def xyt(synthetic_df):
    cfg = ExperimentConfig(outcome=config.VISIT_COL, base_learner="gbdt")
    return make_xyt(synthetic_df, cfg)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_model_fits_and_predicts(name, xyt):
    X, y, t = xyt
    model = models.build_model(name, **FAST)
    model.fit(X, y, t)
    u = model.predict_uplift(X)
    assert u.shape == (len(X),)
    assert np.isfinite(u).all()


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_model_recovers_signal(name, xyt):
    """On a dataset with a real heterogeneous effect, in-sample Qini should be
    clearly positive (sanity that the estimator learns *something* causal)."""
    X, y, t = xyt
    model = models.build_model(name, **FAST).fit(X, y, t)
    u = model.predict_uplift(X)
    assert ev.qini_auc_score(y, t, u) > 0.02


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        models.build_model("q-learner")


def test_build_classifier_kinds():
    for kind in ["gbdt", "logreg"]:
        clf = models.build_classifier(kind)
        assert hasattr(clf, "fit")
