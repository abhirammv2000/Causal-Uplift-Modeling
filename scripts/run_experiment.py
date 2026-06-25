"""Run the full uplift experiment and write a metrics table + figures.

Usage:
    python scripts/run_experiment.py --outcome visit --base-learner lgbm
    python scripts/run_experiment.py --use-raw --n-rows 1000000   # on full data

Outputs:
    reports/metrics.csv          headline metrics per model
    reports/figures/qini.png     Qini curves
    reports/figures/uplift.png   cumulative uplift curves
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uplift import config, data, evaluation, pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcome", default=config.VISIT_COL, choices=config.OUTCOME_COLS)
    parser.add_argument("--base-learner", default="lgbm",
                        choices=["lgbm", "xgb", "gbdt", "logreg"])
    parser.add_argument("--use-raw", action="store_true",
                        help="use the full downloaded dataset instead of the sample")
    parser.add_argument("--n-rows", type=int, default=None,
                        help="subsample to this many rows (stratified)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = config.ExperimentConfig(
        outcome=args.outcome,
        base_learner=args.base_learner,
        n_rows=args.n_rows,
        random_state=args.seed,
    )

    if args.use_raw:
        # Load the full file with memory-efficient dtypes, then let the pipeline
        # take a *stratified* subsample (cfg.n_rows) — unbiased, unlike reading
        # the first N rows of a possibly-ordered file.
        dtypes = {f: "float32" for f in config.FEATURES}
        for c in (config.TREATMENT_COL, config.EXPOSURE_COL,
                  config.CONVERSION_COL, config.VISIT_COL):
            dtypes[c] = "int8"
        print("Loading full raw dataset (memory-efficient dtypes)...")
        df = data.load_raw().astype(dtypes)
    else:
        df = data.load_sample()
    print(f"Loaded {len(df):,} rows. Running models on outcome={cfg.outcome!r} "
          f"with base learner={cfg.base_learner!r} ...")

    result = pipeline.run_experiment(df, cfg)

    print(f"\nObserved ATE (test) on {cfg.outcome}: {result.ate:+.4f}\n")
    print(result.metrics.round(4).to_string())

    reports_dir = config.REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    result.metrics.round(5).to_csv(reports_dir / "metrics.csv")

    ax = evaluation.plot_qini(result.qini_curves)
    ax.figure.savefig(config.FIGURES_DIR / "qini.png", dpi=120, bbox_inches="tight")
    plt.close(ax.figure)

    ax = evaluation.plot_uplift_curve(result.uplift_curves)
    ax.figure.savefig(config.FIGURES_DIR / "uplift.png", dpi=120, bbox_inches="tight")
    plt.close(ax.figure)

    print(f"\nSaved metrics -> {reports_dir / 'metrics.csv'}")
    print(f"Saved figures -> {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
