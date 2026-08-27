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


def test_matches_scikit_uplift(trial):
    """Cross-check the from-scratch metrics against the reference sklift library."""
    sklift_metrics = pytest.importorskip("sklift.metrics")
    y, t, u = trial
    # uplift@k is an exact, unambiguous quantity - it must match to the bit.
    assert ev.uplift_at_k(y, t, u, 0.3) == pytest.approx(
        sklift_metrics.uplift_at_k(y, u, t, strategy="overall", k=0.3), abs=1e-9
    )
    # Normalized Qini agrees with sklift's to a tight tolerance (tiny differences
    # come only from trapezoid endpoint / tie handling).
    assert ev.qini_auc_score(y, t, u) == pytest.approx(
        sklift_metrics.qini_auc_score(y, u, t), abs=0.01
    )


def test_compare_models_bootstrap_flags_ties(trial):
    y, t, true_uplift = trial
    rng = np.random.default_rng(3)
    preds = {
        "good": true_uplift,
        "good_copy": true_uplift + rng.normal(0, 1e-9, len(y)),  # ~identical
        "random": rng.normal(size=len(y)),
    }
    cmp = ev.compare_models_bootstrap(y, t, preds, "qini", n_boot=100, random_state=0)
    # best model is one of the two near-identical "good" models
    assert cmp.index[0] in {"good", "good_copy"}
    # the near-identical twin is statistically tied with the best...
    twin = "good_copy" if cmp.index[0] == "good" else "good"
    assert bool(cmp.loc[twin, "tied"])
    # ...while a random scorer is significantly worse (diff CI excludes 0).
    assert not bool(cmp.loc["random", "tied"])
    assert cmp.loc["random", "diff_lo"] > 0
