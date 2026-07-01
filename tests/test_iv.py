"""Tests for the Wald / CACE estimator.

The guarantee that matters: under one-sided non-compliance, the Wald estimator
recovers the *complier* effect (which the diluted intent-to-treat effect
understates), and the Wald identity ``CACE * first_stage == ITT`` holds.
"""

import numpy as np
import pytest

from uplift import iv


@pytest.fixture
def noncompliance():
    """An RCT with one-sided non-compliance where the true complier effect is known."""
    rng = np.random.default_rng(11)
    n = 60_000
    z = (rng.random(n) < 0.85).astype(int)      # randomized assignment (instrument)
    is_complier = rng.random(n) < 0.30          # latent type, independent of z
    d = (z * is_complier).astype(int)           # one-sided: only assigned compliers exposed
    base, effect = 0.20, 0.12
    p = base + d * effect                        # exclusion: only exposure moves the outcome
    y = (rng.random(n) < p).astype(int)
    return y, z, d, effect


def test_wald_recovers_complier_effect(noncompliance):
    y, z, d, effect = noncompliance
    res = iv.estimate_cace(y, z, d, n_boot=200)
    assert res.cace == pytest.approx(effect, abs=0.03)


def test_cace_exceeds_diluted_itt(noncompliance):
    y, z, d, _ = noncompliance
    res = iv.estimate_cace(y, z, d, n_boot=100)
    # The ITT is the complier effect diluted by non-compliance, so it is smaller.
    assert abs(res.itt_y) < abs(res.cace)


def test_wald_identity(noncompliance):
    y, z, d, _ = noncompliance
    res = iv.estimate_cace(y, z, d, n_boot=50)
    assert res.cace * res.itt_d == pytest.approx(res.itt_y, rel=1e-9)


def test_first_stage_equals_compliance(noncompliance):
    y, z, d, _ = noncompliance
    res = iv.estimate_cace(y, z, d, n_boot=50)
    assert res.itt_d == pytest.approx(0.30, abs=0.02)


def test_ci_brackets_point(noncompliance):
    y, z, d, _ = noncompliance
    res = iv.estimate_cace(y, z, d, n_boot=200, random_state=0)
    assert res.lo <= res.cace <= res.hi
    assert res.se > 0


def test_validation():
    with pytest.raises(ValueError):
        iv.estimate_cace([1, 0], [1], [0, 1])           # mismatched lengths
    with pytest.raises(ValueError):
        iv.estimate_cace([1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0])  # dead first stage
