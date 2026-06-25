"""End-to-end uplift experiment: load -> split -> fit models -> evaluate.

This is the single entry point used by ``scripts/run_experiment.py`` and the
notebooks, so a run is fully reproducible from an :class:`ExperimentConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import data as datamod
from . import evaluation as evalmod
from . import models as modelmod
from .config import ExperimentConfig

DEFAULT_MODELS = ["s-learner", "t-learner", "x-learner", "class-transform"]


@dataclass
class ExperimentResult:
    """Outputs of a run: a metrics table, fitted models, and curve data."""

    metrics: pd.DataFrame
    qini_curves: dict[str, pd.DataFrame]
    uplift_curves: dict[str, pd.DataFrame]
    fitted: dict[str, modelmod.BaseUpliftModel]
    predictions: dict[str, np.ndarray]
    ate: float


def average_treatment_effect(y, t) -> float:
    """Observed ATE from the randomized arms: P(y|t=1) - P(y|t=0)."""
    y, t = np.asarray(y), np.asarray(t)
    return float(y[t == 1].mean() - y[t == 0].mean())


def run_experiment(
    df: pd.DataFrame,
    cfg: ExperimentConfig,
    model_names: list[str] | None = None,
) -> ExperimentResult:
    """Fit and evaluate a set of uplift models on a single dataset."""
    model_names = model_names or DEFAULT_MODELS

    if cfg.n_rows is not None and cfg.n_rows < len(df):
        df = datamod.stratified_sample(df, cfg.n_rows, random_state=cfg.random_state)

    X_tr, X_te, y_tr, y_te, t_tr, t_te = datamod.train_test_split_uplift(df, cfg)
    ate = average_treatment_effect(y_te, t_te)

    rows, qini_curves, uplift_curves, fitted, preds = [], {}, {}, {}, {}
    for name in model_names:
        model = modelmod.build_model(name, base_learner=cfg.base_learner,
                                     random_state=cfg.random_state)
        model.fit(X_tr, y_tr, t_tr)
        uplift = model.predict_uplift(X_te)

        report = evalmod.evaluate(y_te, t_te, uplift)
        rows.append({"model": model.name, **report.as_dict()})
        qini_curves[model.name] = evalmod.qini_curve(y_te, t_te, uplift)
        uplift_curves[model.name] = evalmod.uplift_curve(y_te, t_te, uplift)
        fitted[model.name] = model
        preds[model.name] = uplift

    metrics = (
        pd.DataFrame(rows).set_index("model").sort_values("qini", ascending=False)
    )
    return ExperimentResult(metrics, qini_curves, uplift_curves, fitted, preds, ate)
