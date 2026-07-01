"""Instrumental-variables / CACE estimation for the non-compliance problem.

Criteo randomized *assignment* (``treatment``), but only a fraction of assigned
users were actually *exposed* to the ad (``exposure``). So everything the uplift
models estimate is an **intent-to-treat (ITT)** effect: the effect of being
*targeted*, diluted by everyone who was targeted but never saw an ad.

The effect on the users the ad actually reached — the **complier average causal
effect (CACE)**, a.k.a. LATE — is recovered with the **Wald estimator**, using
randomized assignment as an instrument for exposure:

    CACE = ITT_Y / ITT_D
         = (effect of assignment on the outcome) / (effect of assignment on exposure)

Identification rests on the standard IV assumptions, all plausible here:

* **Randomization** — assignment is random (true by design in an RCT).
* **Exclusion** — assignment moves the outcome *only* through exposure.
* **Monotonicity / no defiers** — being assigned never makes a user *less* likely
  to be exposed.
* **Relevance** — a non-zero first stage (assignment actually shifts exposure).

Criteo has *one-sided* non-compliance — a control user can never be exposed — so
the first stage ``ITT_D`` is simply the exposure rate among the assigned, and the
CACE is the ITT rescaled by that compliance rate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_arrays(y, z, d) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()
    d = np.asarray(d, dtype=float).ravel()
    if not (len(y) == len(z) == len(d)):
        raise ValueError("y, z (assignment), and d (exposure) must have the same length")
    if len(y) == 0:
        raise ValueError("empty input")
    return y, z, d


def _wald(y, z, d) -> tuple[float, float, float]:
    """Return ``(itt_y, itt_d, cace)`` for one sample."""
    treated, control = z == 1, z == 0
    if not treated.any() or not control.any():
        raise ValueError("need both assigned (z=1) and unassigned (z=0) units")
    itt_y = y[treated].mean() - y[control].mean()
    itt_d = d[treated].mean() - d[control].mean()
    if abs(itt_d) < 1e-12:
        raise ValueError(
            "first stage is ~0: assignment does not move exposure, so the CACE "
            "is unidentified (the instrument is irrelevant)"
        )
    return float(itt_y), float(itt_d), float(itt_y / itt_d)


@dataclass
class CACEResult:
    """ITT, first stage, and the Wald CACE with a bootstrap confidence interval."""

    itt_y: float       # intent-to-treat effect on the outcome
    itt_d: float       # first stage: effect of assignment on exposure (= compliance)
    cace: float        # complier average causal effect (LATE)
    lo: float          # CACE lower confidence bound
    hi: float          # CACE upper confidence bound
    se: float          # bootstrap standard error of the CACE

    def as_dict(self) -> dict[str, float]:
        return {
            "itt_y": self.itt_y,
            "first_stage": self.itt_d,
            "cace": self.cace,
            "lo": self.lo,
            "hi": self.hi,
            "se": self.se,
        }


def estimate_cace(
    y, z, d,
    *,
    n_boot: int = 500,
    alpha: float = 0.05,
    random_state: int = 42,
) -> CACEResult:
    """Estimate the CACE (LATE) via the Wald estimator with a bootstrap CI.

    Parameters
    ----------
    y : outcome (e.g. ``visit``).
    z : randomized assignment / instrument (e.g. ``treatment``).
    d : actual exposure / treatment received (e.g. ``exposure``).
    n_boot, alpha, random_state : bootstrap controls for the percentile CI.
    """
    y, z, d = _as_arrays(y, z, d)
    itt_y, itt_d, cace = _wald(y, z, d)

    rng = np.random.default_rng(random_state)
    n = len(y)
    samples = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            samples[b] = _wald(y[idx], z[idx], d[idx])[2]
        except ValueError:
            samples[b] = np.nan
    samples = samples[np.isfinite(samples)]
    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return CACEResult(itt_y, itt_d, cace, float(lo), float(hi), float(samples.std(ddof=1)))
