"""Project paths, dataset schema, and run configuration.

Centralising these constants keeps notebooks, scripts, and tests in sync and
makes the whole pipeline reproducible from a single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SAMPLE_DIR = DATA_DIR / "sample"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

RAW_FILENAME = "criteo-uplift-v2.1.csv.gz"
SAMPLE_FILENAME = "criteo_sample.parquet"

# Public Criteo AI Lab uplift dataset (v2.1). ~290 MB compressed, ~13.9M rows.
CRITEO_URL = "http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz"

# --------------------------------------------------------------------------- #
# Dataset schema
# --------------------------------------------------------------------------- #
FEATURES = [f"f{i}" for i in range(12)]  # f0 .. f11, dense anonymised floats
TREATMENT_COL = "treatment"              # 1 = targeted by campaign, 0 = control
EXPOSURE_COL = "exposure"                # 1 = actually saw the ad (a mediator)
# Outcomes: `visit` is the denser, higher-signal label; `conversion` is rarer.
VISIT_COL = "visit"
CONVERSION_COL = "conversion"
OUTCOME_COLS = [VISIT_COL, CONVERSION_COL]

ALL_COLUMNS = FEATURES + [TREATMENT_COL, EXPOSURE_COL, CONVERSION_COL, VISIT_COL]


@dataclass
class ExperimentConfig:
    """Configuration for a single uplift experiment run."""

    outcome: str = VISIT_COL          # which label to model
    features: list[str] = field(default_factory=lambda: list(FEATURES))
    treatment_col: str = TREATMENT_COL
    test_size: float = 0.3
    random_state: int = 42
    # Sub-sampling for fast, reproducible runs on a laptop. None = use all rows.
    n_rows: int | None = None
    base_learner: str = "lgbm"        # one of: lgbm, xgb, logreg, gbdt
    qini_bins: int = 100              # resolution of qini/uplift curves

    def __post_init__(self) -> None:
        if self.outcome not in OUTCOME_COLS:
            raise ValueError(f"outcome must be one of {OUTCOME_COLS}, got {self.outcome!r}")
        if not 0.0 < self.test_size < 1.0:
            raise ValueError("test_size must be in (0, 1)")
