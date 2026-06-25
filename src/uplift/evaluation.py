"""Uplift-specific evaluation metrics and plots.

Classification metrics (accuracy, ROC-AUC) are meaningless for uplift because
the true individual treatment effect is never observed. Instead we rank units
by predicted uplift and measure how much *incremental* outcome we capture in the
top of that ranking, against the treatment/control groups of a randomized trial.

Implementations here are written from scratch for transparency and are unit
tested for the properties that matter (a perfect ranking beats random; a random
ranking scores ~0). ``scikit-uplift`` is used independently in the notebooks to
cross-check these numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _as_arrays(y, t, uplift) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float).ravel()
    t = np.asarray(t, dtype=float).ravel()
    uplift = np.asarray(uplift, dtype=float).ravel()
    if not (len(y) == len(t) == len(uplift)):
        raise ValueError("y, t, and uplift must have the same length")
    if len(y) == 0:
        raise ValueError("empty input")
    return y, t, uplift


def _cumulative(y, t, uplift):
    """Sort by predicted uplift (desc) and return cumulative group counts/outcomes."""
    order = np.argsort(-uplift, kind="mergesort")  # stable, deterministic ties
    y, t = y[order], t[order]
    treat, ctrl = (t == 1), (t == 0)
    n_t = np.cumsum(treat)
    n_c = np.cumsum(ctrl)
    y_t = np.cumsum(y * treat)
    y_c = np.cumsum(y * ctrl)
    return n_t, n_c, y_t, y_c


def qini_curve(y, t, uplift) -> pd.DataFrame:
    """Cumulative Qini curve.

    At each prefix of the uplift-sorted population the Qini value is
    ``Y_t - Y_c * (N_t / N_c)`` — the incremental positive outcomes attributable
    to treatment, with the control response scaled to the treated group size.
    Returns a DataFrame with ``n_targeted``, ``fraction``, and ``qini``, plus a
    ``random`` straight-line baseline.
    """
    y, t, uplift = _as_arrays(y, t, uplift)
    n_t, n_c, y_t, y_c = _cumulative(y, t, uplift)
    n = len(y)
    ratio = np.divide(n_t, n_c, out=np.zeros(n, dtype=float), where=n_c > 0)
    qini = y_t - y_c * ratio

    n_targeted = np.arange(1, n + 1)
    qini = np.r_[0.0, qini]
    n_targeted = np.r_[0, n_targeted]
    fraction = n_targeted / n
    random = fraction * qini[-1]  # diagonal: same total uplift, spread evenly
    return pd.DataFrame(
        {"n_targeted": n_targeted, "fraction": fraction, "qini": qini, "random": random}
    )


def uplift_curve(y, t, uplift) -> pd.DataFrame:
    """Cumulative uplift (AUUC) curve.

    At each prefix the cumulative incremental outcome is
    ``(Y_t/N_t - Y_c/N_c) * (N_t + N_c)``. Returns ``n_targeted``, ``fraction``,
    ``uplift`` and the ``random`` baseline.
    """
    y, t, uplift = _as_arrays(y, t, uplift)
    n_t, n_c, y_t, y_c = _cumulative(y, t, uplift)
    n = len(y)
    rate_t = np.divide(y_t, n_t, out=np.zeros(n, dtype=float), where=n_t > 0)
    rate_c = np.divide(y_c, n_c, out=np.zeros(n, dtype=float), where=n_c > 0)
    curve = (rate_t - rate_c) * (n_t + n_c)

    n_targeted = np.r_[0, np.arange(1, n + 1)]
    curve = np.r_[0.0, curve]
    fraction = n_targeted / n
    random = fraction * curve[-1]
    return pd.DataFrame(
        {"n_targeted": n_targeted, "fraction": fraction, "uplift": curve, "random": random}
    )


def qini_auc_score(y, t, uplift, *, normalize: bool = True) -> float:
    """Qini coefficient: area between the Qini curve and the random baseline.

    With ``normalize=True`` the result is divided by the area of a perfect
    (oracle) Qini curve, yielding a value that is ~0 for random ranking and
    1.0 for a perfect ranking. Negative means worse than random.
    """
    df = qini_curve(y, t, uplift)
    area_model = np.trapezoid(df["qini"], df["fraction"])
    area_random = np.trapezoid(df["random"], df["fraction"])
    coeff = area_model - area_random
    if not normalize:
        return float(coeff)

    # Oracle: rank treated-positives and control-negatives first (best possible).
    y_arr, t_arr, _ = _as_arrays(y, t, uplift)
    oracle_score = np.where(t_arr == 1, y_arr, -y_arr)
    df_perfect = qini_curve(y_arr, t_arr, oracle_score)
    area_perfect = np.trapezoid(df_perfect["qini"], df_perfect["fraction"]) - area_random
    if abs(area_perfect) < 1e-12:
        return 0.0
    return float(coeff / area_perfect)


def auuc_score(y, t, uplift, *, normalize: bool = True) -> float:
    """Area under the uplift curve, relative to the random baseline."""
    df = uplift_curve(y, t, uplift)
    area_model = np.trapezoid(df["uplift"], df["fraction"])
    area_random = np.trapezoid(df["random"], df["fraction"])
    coeff = area_model - area_random
    if not normalize:
        return float(coeff)
    y_arr, t_arr, _ = _as_arrays(y, t, uplift)
    oracle = np.where(t_arr == 1, y_arr, -y_arr)
    df_p = uplift_curve(y_arr, t_arr, oracle)
    area_perfect = np.trapezoid(df_p["uplift"], df_p["fraction"]) - area_random
    if abs(area_perfect) < 1e-12:
        return 0.0
    return float(coeff / area_perfect)


def uplift_at_k(y, t, uplift, k: float = 0.3) -> float:
    """Estimated uplift among the top ``k`` fraction ranked by predicted uplift.

    This is the difference in outcome rates between treated and control units
    within the highest-scoring ``k`` of the population — the quantity a
    marketer actually cares about when targeting the top decile/quantile.
    """
    if not 0.0 < k <= 1.0:
        raise ValueError("k must be in (0, 1]")
    y, t, uplift = _as_arrays(y, t, uplift)
    cut = max(1, int(round(len(y) * k)))
    order = np.argsort(-uplift, kind="mergesort")[:cut]
    yk, tk = y[order], t[order]
    treat, ctrl = tk == 1, tk == 0
    if treat.sum() == 0 or ctrl.sum() == 0:
        return float("nan")
    return float(yk[treat].mean() - yk[ctrl].mean())


def weighted_average_uplift(y, t, uplift, bins: int = 10) -> float:
    """Average per-bin uplift weighted by bin size (sklift-style WAU)."""
    y, t, uplift = _as_arrays(y, t, uplift)
    order = np.argsort(-uplift, kind="mergesort")
    y, t = y[order], t[order]
    edges = np.linspace(0, len(y), bins + 1).astype(int)
    total_w, acc = 0.0, 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            continue
        yb, tb = y[lo:hi], t[lo:hi]
        treat, ctrl = tb == 1, tb == 0
        if treat.sum() == 0 or ctrl.sum() == 0:
            continue
        u = yb[treat].mean() - yb[ctrl].mean()
        w = hi - lo
        acc += u * w
        total_w += w
    return float(acc / total_w) if total_w else float("nan")


@dataclass
class UpliftReport:
    """Bundle of headline uplift metrics for one model."""

    qini: float
    auuc: float
    uplift_at_10: float
    uplift_at_30: float
    weighted_avg_uplift: float

    def as_dict(self) -> dict[str, float]:
        return {
            "qini": self.qini,
            "auuc": self.auuc,
            "uplift@10%": self.uplift_at_10,
            "uplift@30%": self.uplift_at_30,
            "weighted_avg_uplift": self.weighted_avg_uplift,
        }


def evaluate(y, t, uplift) -> UpliftReport:
    """Compute the standard uplift metric bundle for a set of predictions."""
    return UpliftReport(
        qini=qini_auc_score(y, t, uplift),
        auuc=auuc_score(y, t, uplift),
        uplift_at_10=uplift_at_k(y, t, uplift, 0.10),
        uplift_at_30=uplift_at_k(y, t, uplift, 0.30),
        weighted_avg_uplift=weighted_average_uplift(y, t, uplift),
    )


# --------------------------------------------------------------------------- #
# Uncertainty: bootstrap confidence intervals
# --------------------------------------------------------------------------- #
def bootstrap_metric(
    y, t, uplift,
    metric: str = "qini",
    *,
    n_boot: int = 200,
    alpha: float = 0.05,
    random_state: int = 42,
) -> dict[str, float]:
    """Bootstrap a percentile confidence interval for an uplift metric.

    A single Qini number is nearly meaningless without a sense of its sampling
    variability — uplift signal is noisy. Resampling units with replacement and
    recomputing the metric gives an honest interval. Returns ``point``, ``lo``,
    ``hi`` (the (1-alpha) percentile CI), and ``se``.
    """
    fns = {
        "qini": qini_auc_score,
        "auuc": auuc_score,
        "uplift@10%": lambda yy, tt, uu: uplift_at_k(yy, tt, uu, 0.10),
        "uplift@30%": lambda yy, tt, uu: uplift_at_k(yy, tt, uu, 0.30),
    }
    if metric not in fns:
        raise ValueError(f"metric must be one of {list(fns)}, got {metric!r}")
    fn = fns[metric]
    y, t, uplift = _as_arrays(y, t, uplift)

    rng = np.random.default_rng(random_state)
    n = len(y)
    samples = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        samples[b] = fn(y[idx], t[idx], uplift[idx])
    samples = samples[np.isfinite(samples)]
    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "point": float(fn(y, t, uplift)),
        "lo": float(lo),
        "hi": float(hi),
        "se": float(samples.std(ddof=1)),
    }


# --------------------------------------------------------------------------- #
# Calibration: do predicted uplifts match realized uplifts?
# --------------------------------------------------------------------------- #
def uplift_calibration(y, t, uplift, bins: int = 10) -> pd.DataFrame:
    """Compare predicted vs. realized uplift across bins of predicted uplift.

    Units are grouped into ``bins`` by predicted uplift; within each bin we
    report the mean *predicted* uplift and the *observed* uplift (treated rate
    minus control rate). A well-calibrated model tracks the diagonal — a much
    stronger statement than merely ranking units correctly.
    """
    y, t, uplift = _as_arrays(y, t, uplift)
    order = np.argsort(uplift, kind="mergesort")
    y, t, u = y[order], t[order], uplift[order]
    edges = np.linspace(0, len(y), bins + 1).astype(int)
    rows = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if hi <= lo:
            continue
        yb, tb, ub = y[lo:hi], t[lo:hi], u[lo:hi]
        treat, ctrl = tb == 1, tb == 0
        observed = (yb[treat].mean() - yb[ctrl].mean()
                    if treat.sum() and ctrl.sum() else np.nan)
        rows.append({"bin": i, "n": hi - lo,
                     "predicted_uplift": float(ub.mean()),
                     "observed_uplift": float(observed)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Plotting (imported lazily so the metrics have no hard matplotlib dependency)
# --------------------------------------------------------------------------- #
def plot_qini(curves: dict[str, pd.DataFrame], ax=None):
    """Plot one or more Qini curves. ``curves`` maps model name -> qini_curve df."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    first = next(iter(curves.values()))
    ax.plot(first["fraction"], first["random"], "k--", lw=1, label="Random")
    for name, df in curves.items():
        ax.plot(df["fraction"], df["qini"], lw=2, label=name)
    ax.set_xlabel("Fraction of population targeted")
    ax.set_ylabel("Cumulative incremental conversions (Qini)")
    ax.set_title("Qini curves")
    ax.legend(loc="upper left")
    return ax


def plot_uplift_curve(curves: dict[str, pd.DataFrame], ax=None):
    """Plot one or more cumulative uplift (AUUC) curves."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    first = next(iter(curves.values()))
    ax.plot(first["fraction"], first["random"], "k--", lw=1, label="Random")
    for name, df in curves.items():
        ax.plot(df["fraction"], df["uplift"], lw=2, label=name)
    ax.set_xlabel("Fraction of population targeted")
    ax.set_ylabel("Cumulative uplift")
    ax.set_title("Uplift curves")
    ax.legend(loc="upper left")
    return ax


def plot_calibration(calib: pd.DataFrame, ax=None, title: str | None = None):
    """Scatter predicted vs. observed uplift (from :func:`uplift_calibration`)."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(calib["predicted_uplift"], calib["observed_uplift"], s=40, zorder=3)
    both = pd.concat([calib["predicted_uplift"], calib["observed_uplift"]])
    lims = [both.min() - 0.005, both.max() + 0.005]
    ax.plot(lims, lims, "k--", lw=1, label="perfect calibration")
    ax.set_xlabel("Mean predicted uplift (bin)")
    ax.set_ylabel("Observed uplift (bin)")
    ax.set_title(title or "Uplift calibration")
    ax.legend()
    return ax
