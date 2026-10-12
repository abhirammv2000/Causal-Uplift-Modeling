"""Create the small committed sample at data/sample/criteo_sample.parquet.

This always generates a **synthetic** Criteo-like dataset - same schema,
imbalanced ~85/15 treatment assignment, low visit / conversion rates, and
heterogeneous treatment effects (persuadables and sleeping dogs) so
uplift models have signal to find. It guarantees the repo runs end-to-end for
anyone who clones it.

We deliberately do *not* commit a slice of the real Criteo data: Criteo releases
it for research without redistribution rights. For real results, download the
full file (`scripts/download_data.py`) and run `run_experiment.py --use-raw`.

Usage:
    python scripts/make_sample.py [--n 50000] [--seed 42]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uplift import config  # noqa: E402


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def synthesize(n: int, seed: int = 42) -> pd.DataFrame:
    """Generate a Criteo-like RCT with heterogeneous treatment effects."""
    rng = np.random.default_rng(seed)
    F = config.FEATURES
    X = rng.standard_normal((n, len(F))).astype("float32")

    # Randomized, imbalanced assignment (matches Criteo's ~85% treated). This is
    # the *instrument*: it is random, so it is a valid handle on exposure.
    t = (rng.random(n) < 0.85).astype("int8")

    # Exposure is the true mediator: only the "compliers" among the assigned
    # actually see the ad. One-sided non-compliance - a control user is never
    # exposed - exactly as in the real Criteo campaign (where exposure is low).
    compliance_rate = 0.35
    exposure = ((t == 1) & (rng.random(n) < compliance_rate)).astype("int8")

    # Baseline visit propensity (~5%) driven by a few features.
    base = -3.0 + 0.6 * X[:, 0] + 0.4 * X[:, 3] - 0.3 * X[:, 7]

    # Heterogeneous effect: persuadables where f1 high, sleeping dogs where f5
    # high. The effect flows through *exposure*, not mere assignment (exclusion
    # restriction), so being targeted helps only via actually seeing the ad.
    tau = 0.9 * X[:, 1] - 0.7 * X[:, 5] + 0.3 * X[:, 2]

    p_visit = _sigmoid(base + exposure * tau)
    visit = (rng.random(n) < p_visit).astype("int8")

    # Conversion is rarer and only happens given a visit.
    p_conv = _sigmoid(-2.0 + 1.5 * (visit == 1) + 0.4 * X[:, 4] + 0.5 * exposure * X[:, 1])
    conversion = ((rng.random(n) < p_conv) & (visit == 1)).astype("int8")

    out = pd.DataFrame(X, columns=F)
    out[config.TREATMENT_COL] = t
    out[config.EXPOSURE_COL] = exposure
    out[config.CONVERSION_COL] = conversion
    out[config.VISIT_COL] = visit
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=50_000, help="approx number of rows")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Generating a synthetic Criteo-like sample (no real data is redistributed).")
    sample = synthesize(args.n, seed=args.seed)
    source = "synthetic (Criteo-like DGP)"

    config.SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.SAMPLE_DIR / config.SAMPLE_FILENAME
    sample.to_parquet(out_path, index=False)

    rates = {
        "rows": len(sample),
        "treated_%": round(100 * sample[config.TREATMENT_COL].mean(), 1),
        "visit_%": round(100 * sample[config.VISIT_COL].mean(), 2),
        "conversion_%": round(100 * sample[config.CONVERSION_COL].mean(), 2),
        "ate_visit": round(
            sample.loc[sample[config.TREATMENT_COL] == 1, config.VISIT_COL].mean()
            - sample.loc[sample[config.TREATMENT_COL] == 0, config.VISIT_COL].mean(),
            4,
        ),
    }
    print(f"Wrote {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")
    print(f"Source: {source}")
    print(f"Summary: {rates}")


if __name__ == "__main__":
    main()
