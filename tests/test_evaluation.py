"""Property tests for the uplift metrics.

The key guarantees: a perfect ranking beats a random one; a random ranking
scores ~0 Qini; and the metrics are sign-aware (ranking by *negative* uplift is
worse than random).
"""

import numpy as np
import pytest

from uplift import evaluation as ev


@pytest.fixture
def trial():
    """A randomized trial where true uplift is known per unit."""
    rng = np.random.default_rng(7)
    n = 6000
    t = (rng.random(n) < 0.5).astype(int)
    true_uplift = rng.normal(0, 1, n)            # heterogeneous true effect
    base = 0.2
    p = np.clip(base + t * true_uplift * 0.1, 0.01, 0.99)
    y = (rng.random(n) < p).astype(int)
    return y, t, true_uplift


def test_qini_perfect_beats_random(trial):
    y, t, true_uplift = trial
    rng = np.random.default_rng(0)
    random_scores = rng.normal(size=len(y))
    assert ev.qini_auc_score(y, t, true_uplift) > ev.qini_auc_score(y, t, random_scores)


def test_random_qini_near_zero(trial):
    y, t, _ = trial
    rng = np.random.default_rng(1)
    scores = rng.normal(size=len(y))
    assert abs(ev.qini_auc_score(y, t, scores)) < 0.15


def test_reversed_scores_are_worse(trial):
    y, t, true_uplift = trial
    good = ev.qini_auc_score(y, t, true_uplift)
    bad = ev.qini_auc_score(y, t, -true_uplift)
    assert bad < good
    assert bad < 0  # ranking by anti-uplift should lose money vs random


def test_curves_start_and_end(trial):
    y, t, u = trial
    q = ev.qini_curve(y, t, u)
    assert q["qini"].iloc[0] == 0.0
    assert q["fraction"].iloc[-1] == pytest.approx(1.0)
    # at full population qini and random baseline must coincide
    assert q["qini"].iloc[-1] == pytest.approx(q["random"].iloc[-1])


def test_uplift_at_k_matches_manual(trial):
    y, t, u = trial
    k = 0.25
    val = ev.uplift_at_k(y, t, u, k)
    order = np.argsort(-u, kind="mergesort")[: int(round(len(y) * k))]
    yk, tk = y[order], t[order]
    expected = yk[tk == 1].mean() - yk[tk == 0].mean()
    assert val == pytest.approx(expected)


def test_evaluate_bundle_keys(trial):
    y, t, u = trial
    d = ev.evaluate(y, t, u).as_dict()
    assert set(d) == {"qini", "auuc", "uplift@10%", "uplift@30%", "weighted_avg_uplift"}


def test_input_validation():
    with pytest.raises(ValueError):
        ev.qini_curve([1, 0], [1], [0.5, 0.5])
    with pytest.raises(ValueError):
        ev.uplift_at_k([1, 0], [1, 0], [0.1, 0.2], k=1.5)


def test_bootstrap_metric_brackets_point(trial):
    y, t, u = trial
    ci = ev.bootstrap_metric(y, t, u, "qini", n_boot=100, random_state=0)
    assert ci["lo"] <= ci["point"] <= ci["hi"]
    assert ci["se"] > 0
    with pytest.raises(ValueError):
        ev.bootstrap_metric(y, t, u, "not_a_metric")


def test_calibration_shape_and_signal(trial):
    y, t, u = trial
    calib = ev.uplift_calibration(y, t, u, bins=8)
    assert len(calib) == 8
    assert {"predicted_uplift", "observed_uplift", "n"}.issubset(calib.columns)
    # predicted uplift must be monotonically non-decreasing across sorted bins
    assert calib["predicted_uplift"].is_monotonic_increasing
