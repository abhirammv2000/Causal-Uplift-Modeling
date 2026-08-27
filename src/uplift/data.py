"""Data acquisition, loading, sampling, and splitting for the Criteo dataset.

The real dataset (~13.9M rows) is downloaded on demand; a small stratified
sample is committed to the repo so notebooks, the test suite, and CI all run in
seconds without the multi-hundred-MB download.
"""

from __future__ import annotations

import gzip
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def download_criteo(dest: Path | None = None, *, force: bool = False) -> Path:
    """Download the raw Criteo uplift file to ``data/raw``.

    Returns the path to the gzipped CSV. Skips the download if it already
    exists unless ``force=True``. Requires the ``requests`` and ``tqdm``
    packages (both in requirements.txt).
    """
    import requests
    from tqdm import tqdm

    dest = dest or (config.RAW_DIR / config.RAW_FILENAME)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return dest

    with requests.get(config.CRITEO_URL, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as fh, tqdm(
            total=total, unit="B", unit_scale=True, desc="criteo"
        ) as bar:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                bar.update(len(chunk))
        shutil.move(tmp, dest)
    return dest


def load_raw(
    path: Path | None = None,
    *,
    n_rows: int | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load the raw (gzipped) Criteo CSV into a DataFrame.

    ``n_rows`` limits how many rows are read (useful for quick experiments);
    note this reads the *first* n rows, so shuffle before relying on it for
    unbiased estimates - :func:`stratified_sample` is preferred for that.
    """
    path = path or (config.RAW_DIR / config.RAW_FILENAME)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}. Run `python scripts/download_data.py` first."
        )
    return pd.read_csv(path, nrows=n_rows, usecols=columns)


def load_sample(path: Path | None = None) -> pd.DataFrame:
    """Load the small committed sample shipped with the repo."""
    path = path or (config.SAMPLE_DIR / config.SAMPLE_FILENAME)
    if not path.exists():
        raise FileNotFoundError(
            f"Sample not found at {path}. Run `python scripts/make_sample.py`."
        )
    return pd.read_parquet(path)


def stratified_sample(
    df: pd.DataFrame,
    n: int,
    *,
    strata: tuple[str, str] = (config.TREATMENT_COL, config.VISIT_COL),
    random_state: int = 42,
) -> pd.DataFrame:
    """Return a sample of ~``n`` rows preserving the joint (treatment, outcome)
    distribution. Critical for uplift: naive sampling can wipe out the rare
    treated-and-converted cell that carries most of the signal.
    """
    rng = np.random.RandomState(random_state)
    frac = min(1.0, n / len(df))
    parts = []
    for _, grp in df.groupby(list(strata)):
        k = max(1, int(round(len(grp) * frac)))
        parts.append(grp.sample(n=min(k, len(grp)), random_state=rng.randint(1 << 31)))
    out = pd.concat(parts).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return out


def make_xyt(
    df: pd.DataFrame,
    cfg: config.ExperimentConfig,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Split a DataFrame into features ``X``, outcome ``y``, treatment ``t``."""
    X = df[cfg.features].copy()
    y = df[cfg.outcome].astype(int)
    t = df[cfg.treatment_col].astype(int)
    return X, y, t


def train_test_split_uplift(
    df: pd.DataFrame,
    cfg: config.ExperimentConfig,
):
    """Stratified train/test split on the (treatment, outcome) cell.

    Returns ``(X_tr, X_te, y_tr, y_te, t_tr, t_te)``.
    """
    from sklearn.model_selection import train_test_split

    X, y, t = make_xyt(df, cfg)
    strat = t.astype(str) + "_" + y.astype(str)
    X_tr, X_te, y_tr, y_te, t_tr, t_te = train_test_split(
        X, y, t,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=strat,
    )
    return X_tr, X_te, y_tr, y_te, t_tr, t_te


def _gzip_csv(src_csv: Path, dest_gz: Path) -> None:
    with open(src_csv, "rb") as fi, gzip.open(dest_gz, "wb") as fo:
        shutil.copyfileobj(fi, fo)
