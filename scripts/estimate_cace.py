"""Estimate the campaign's effect on the *exposed* (CACE / LATE), not just the
intent-to-treat effect, using randomized assignment as an instrument for exposure.

Only a fraction of targeted users actually saw an ad, so the uplift models'
numbers are intent-to-treat (the effect of being *targeted*). This rescales by
the compliance rate to recover the effect on users the ad actually reached.

Usage:
    python scripts/estimate_cace.py                 # committed synthetic sample
    python scripts/estimate_cace.py --use-raw       # full real Criteo data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uplift import config, data, iv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcome", default=config.VISIT_COL, choices=config.OUTCOME_COLS)
    parser.add_argument("--use-raw", action="store_true",
                        help="use the full downloaded dataset instead of the sample")
    args = parser.parse_args()

    df = data.load_raw() if args.use_raw else data.load_sample()
    y = df[args.outcome].to_numpy()
    z = df[config.TREATMENT_COL].to_numpy()
    d = df[config.EXPOSURE_COL].to_numpy()

    res = iv.estimate_cace(y, z, d)

    print(f"Rows: {len(df):,}   outcome: {args.outcome!r}\n")
    print(f"First stage (exposure rate among the assigned): {res.itt_d:6.3f}")
    print(f"ITT  - effect of being targeted:                {res.itt_y:+7.4f}")
    print(f"CACE - effect on the exposed (compliers):       {res.cace:+7.4f}"
          f"   95% CI [{res.lo:+.4f}, {res.hi:+.4f}]")
    print(
        "\nThe ITT is diluted by everyone targeted but never shown the ad. The CACE\n"
        "rescales by the compliance rate to recover the effect on users the ad\n"
        "actually reached - identified by the Wald estimator (assignment as an\n"
        "instrument for exposure), valid under one-sided non-compliance."
    )


if __name__ == "__main__":
    main()
