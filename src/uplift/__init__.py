"""Causal uplift modeling toolkit for the Criteo incrementality dataset.

Submodules
----------
config      : paths, column names, and run configuration.
data        : download, load, sample, and train/test splitting.
models      : uplift estimators (S-, T-, X-learners and class transformation).
evaluation  : uplift-specific metrics (Qini, AUUC, uplift@k) and plots.
business    : profit curves, ROI, and optimal-targeting decisions.
iv          : instrumental-variables / CACE (Wald) for non-compliance.
pipeline    : end-to-end experiment runner tying the pieces together.
"""

from . import business, config, data, evaluation, iv, models  # noqa: F401

__version__ = "0.1.0"
