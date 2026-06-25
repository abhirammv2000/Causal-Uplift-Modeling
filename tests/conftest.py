import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uplift import config  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    """Small Criteo-like dataset with a real, recoverable uplift signal."""
    rng = np.random.default_rng(0)
    n = 5000
    X = rng.standard_normal((n, len(config.FEATURES))).astype("float32")
    t = (rng.random(n) < 0.7).astype(int)
    base = -2.0 + 0.5 * X[:, 0]
    tau = 1.2 * X[:, 1]  # heterogeneous effect driven by f1
    p = 1.0 / (1.0 + np.exp(-(base + t * tau)))
    y = (rng.random(n) < p).astype(int)
    df = pd.DataFrame(X, columns=config.FEATURES)
    df[config.TREATMENT_COL] = t
    df[config.EXPOSURE_COL] = 0
    df[config.VISIT_COL] = y
    df[config.CONVERSION_COL] = (y & (rng.random(n) < 0.3)).astype(int)
    return df
