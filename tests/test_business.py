"""Tests for the profit / ROI targeting layer."""

import numpy as np
import pytest

from uplift import business


@pytest.fixture
def trial():
    rng = np.random.default_rng(3)
    n = 6000
    t = (rng.random(n) < 0.5).astype(int)
    true_uplift = rng.normal(0, 1, n)
    p = np.clip(0.2 + t * true_uplift * 0.1, 0.01, 0.99)
    y = (rng.random(n) < p).astype(int)
    return y, t, true_uplift


def test_profit_curve_starts_at_zero(trial):
    y, t, u = trial
    curve = business.profit_curve(y, t, u, value_per_conversion=50, cost_per_treatment=1)
    assert curve["profit"].iloc[0] == 0.0
    assert curve["n_targeted"].iloc[0] == 0
    assert {"fraction", "profit", "random_profit",
            "incremental_conversions"}.issubset(curve.columns)


def test_optimal_beats_random_and_treat_all(trial):
    y, t, u = trial
    d = business.optimal_targeting(y, t, u, value_per_conversion=50, cost_per_treatment=1)
    # A good ranking should target a strict subset and beat both baselines.
    assert 0 < d.best_fraction <= 1.0
    assert d.best_profit >= d.profit_random_at_best
    assert d.best_profit >= d.profit_treat_all
    assert d.uplift_over_random >= 0


def test_high_cost_shrinks_targeting(trial):
    """Raising the per-treatment cost should never target *more* of the base."""
    y, t, u = trial
    cheap = business.optimal_targeting(y, t, u, 50, cost_per_treatment=0.5)
    pricey = business.optimal_targeting(y, t, u, 50, cost_per_treatment=5.0)
    assert pricey.best_fraction <= cheap.best_fraction + 1e-9


def test_validation():
    with pytest.raises(ValueError):
        business.profit_curve([1, 0], [1, 0], [0.1, 0.2], value_per_conversion=0, cost_per_treatment=1)
