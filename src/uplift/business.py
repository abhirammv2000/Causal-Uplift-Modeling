"""Turn uplift rankings into money: profit curves, ROI, and optimal targeting.

A model's Qini is necessary but not what a stakeholder signs off on. The
decision is: *how many users do we treat, and what does it earn?* Given a value
per incremental conversion and a cost per treatment, targeting the top-k by
predicted uplift yields

    profit(k) = value * (incremental conversions in top-k)
              - cost  * (number of users treated in top-k)

The incremental-conversions term is read straight off the uplift curve, which is
estimated from the randomized treated/control arms — so the profit is causal,
not a correlational guess.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import evaluation


def profit_curve(
    y, t, uplift,
    value_per_conversion: float,
    cost_per_treatment: float,
) -> pd.DataFrame:
    """Profit as a function of how deep we target the uplift-ranked population.

    Returns a DataFrame with ``fraction``, ``n_targeted``,
    ``incremental_conversions``, ``profit`` (uplift targeting) and
    ``random_profit`` (targeting the same fraction at random).
    """
    if value_per_conversion <= 0 or cost_per_treatment < 0:
        raise ValueError("value_per_conversion must be > 0 and cost_per_treatment >= 0")
    uc = evaluation.uplift_curve(y, t, uplift)
    inc = uc["uplift"].to_numpy()
    n_targeted = uc["n_targeted"].to_numpy()
    rand_inc = uc["random"].to_numpy()
    return pd.DataFrame({
        "fraction": uc["fraction"].to_numpy(),
        "n_targeted": n_targeted,
        "incremental_conversions": inc,
        "profit": value_per_conversion * inc - cost_per_treatment * n_targeted,
        "random_profit": value_per_conversion * rand_inc - cost_per_treatment * n_targeted,
    })


@dataclass
class TargetingDecision:
    """The profit-maximising targeting depth and what it earns."""

    best_fraction: float
    best_profit: float
    profit_treat_all: float
    profit_random_at_best: float
    uplift_over_random: float   # extra profit vs. random targeting at best depth
    uplift_over_treat_all: float  # extra profit vs. treating everyone

    def as_dict(self) -> dict[str, float]:
        return {
            "best_fraction": self.best_fraction,
            "best_profit": self.best_profit,
            "profit_treat_all": self.profit_treat_all,
            "profit_random_at_best": self.profit_random_at_best,
            "gain_vs_random": self.uplift_over_random,
            "gain_vs_treat_all": self.uplift_over_treat_all,
        }


def optimal_targeting(
    y, t, uplift,
    value_per_conversion: float,
    cost_per_treatment: float,
) -> TargetingDecision:
    """Find the targeting depth that maximises profit and compare to baselines."""
    curve = profit_curve(y, t, uplift, value_per_conversion, cost_per_treatment)
    i = int(curve["profit"].to_numpy().argmax())
    best_profit = float(curve["profit"].iloc[i])
    treat_all = float(curve["profit"].iloc[-1])
    rand_at_best = float(curve["random_profit"].iloc[i])
    return TargetingDecision(
        best_fraction=float(curve["fraction"].iloc[i]),
        best_profit=best_profit,
        profit_treat_all=treat_all,
        profit_random_at_best=rand_at_best,
        uplift_over_random=best_profit - rand_at_best,
        uplift_over_treat_all=best_profit - treat_all,
    )


def plot_profit_curve(curve: pd.DataFrame, decision: TargetingDecision | None = None, ax=None):
    """Plot the profit curve with the random baseline and the optimal depth."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(curve["fraction"], curve["profit"], lw=2, label="Uplift targeting")
    ax.plot(curve["fraction"], curve["random_profit"], "k--", lw=1, label="Random targeting")
    ax.axhline(0, color="grey", lw=0.8)
    if decision is not None:
        ax.axvline(decision.best_fraction, color="C3", ls=":", lw=1.5,
                   label=f"Optimal @ {decision.best_fraction:.0%}")
    ax.set_xlabel("Fraction of population targeted")
    ax.set_ylabel("Expected profit")
    ax.set_title("Profit curve: who should we treat?")
    ax.legend(loc="lower center")
    return ax
