# Causal Uplift Modeling on the Criteo Incrementality Dataset

[![CI](https://github.com/abhirammv2000/Causal-Uplift-Modeling/actions/workflows/ci.yml/badge.svg)](https://github.com/abhirammv2000/Causal-Uplift-Modeling/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%E2%80%933.13-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

Estimating **who to treat**, not just **who will convert**. This project builds
and rigorously evaluates individual treatment-effect (uplift) models on the
[Criteo AI Lab](https://ailab.criteo.com/criteo-uplift-prediction-dataset/)
incrementality benchmark — a ~13.9M-user randomized controlled trial — to find
the customers whose behavior *changes* because of a marketing intervention.

> Classical response models rank people by `P(convert)`. But many of them would
> have converted anyway ("sure things"), and some are actively annoyed by being
> targeted ("sleeping dogs"). Uplift modeling ranks by **incremental**
> conversions — `P(convert | treated) − P(convert | not treated)` — which is the
> quantity that actually drives marketing ROI.

This repo is structured as a reproducible Python package (`src/uplift`), not just
notebooks: four uplift estimators behind a common interface, from-scratch
uplift-specific metrics with a unit-tested correctness suite, a one-command
experiment runner, and a synthetic data fallback so it runs the moment you clone
it.

---

## Headline results — on the real, full dataset

Four uplift estimators trained on the **real Criteo data** (13.98M users; a 2M
stratified subsample for modeling), evaluated on a held-out test set with proper
uplift metrics. Base learner: LightGBM.

| Model | Qini ↑ | AUUC ↑ | Uplift @ 10% | Uplift @ 30% |
|-------|:------:|:------:|:------------:|:------------:|
| **S-learner**           | **0.090** | **0.091** | **0.064** | 0.030 |
| X-learner               | 0.082 | 0.084 | 0.064 | 0.030 |
| T-learner               | 0.076 | 0.078 | 0.062 | 0.027 |
| Class transformation    | 0.059 | 0.060 | 0.054 | 0.025 |

*Qini / AUUC are normalized so 0 = random targeting and 1 = a perfect oracle.*

**The signal is real, not noise.** Bootstrapped 95% CIs (S-learner, 300
resamples): **Qini = 0.090 [0.075, 0.104]**, uplift@10% = 0.064 [0.055, 0.075] —
both comfortably above zero.

**The business takeaway:** the average treatment effect on `visit` is just
**+0.0103** (a +27% relative lift over the 3.8% control rate). But the model's
**top 10% of users show an uplift of +0.064 — about 6× the average.**
Concentrating spend on that decile, instead of treating everyone, is exactly
where uplift modeling pays for itself.

<p align="center">
  <img src="reports/figures/qini.png" width="32%" alt="Qini curves"/>
  <img src="reports/figures/uplift.png" width="32%" alt="Uplift curves"/>
  <img src="reports/figures/calibration.png" width="32%" alt="Uplift calibration"/>
</p>

> These numbers come from the full real dataset:
> `python scripts/download_data.py` then
> `python scripts/run_experiment.py --use-raw`. The repo also ships a small
> **synthetic** Criteo-like sample so everything (notebooks, tests, the default
> `run_experiment.py`) runs in seconds on clone without the ~290 MB download —
> see [data/README.md](data/README.md). Real data is not redistributed.

---

## Why this is a *causal* problem (and how I keep it honest)

The fundamental problem of causal inference is that we never observe both
outcomes for the same person — only treated-or-not. The Criteo data sidesteps
confounding because treatment was **randomized**, so:

- **Evaluation is unbiased by construction.** Ranking quality is measured against
  the real treated/control arms via the **Qini** and **uplift (AUUC)** curves —
  not classification accuracy, which is meaningless here.
- **The metrics are sign-aware and tested.** `tests/test_evaluation.py` asserts
  the properties that must hold: a perfect ranking beats random, a random
  ranking scores ≈ 0 Qini, and ranking by *anti*-uplift scores **below** random.
- **Uncertainty and calibration, not just a point score.** Every headline metric
  ships with a **bootstrap confidence interval**, and an **uplift calibration**
  curve checks that *predicted* uplift matches *realized* uplift per bin — a
  stronger claim than merely ranking units in the right order.
- **I'm precise about scope.** Because Criteo is an RCT, this is *heterogeneous
  treatment-effect estimation*, not de-confounding of observational data. The
  X-learner and class-transformation models still carry the propensity machinery
  needed for the observational case (Criteo's assignment is a strongly imbalanced
  ~85/15), so the same code generalizes.

---

## Methods

Four estimators spanning the standard uplift taxonomy, all behind one
`fit(X, y, t) → predict_uplift(X)` interface (`src/uplift/models.py`):

| Estimator | Idea | Notes |
|-----------|------|-------|
| **S-learner** | One model on `[X, t]`; uplift = `μ(x,1) − μ(x,0)` | Simple, strong baseline |
| **T-learner** | Separate treated/control outcome models | Clean, can overfit small arms |
| **X-learner** | Impute per-unit effects, regress them, blend by propensity | Robust under treatment imbalance (Künzel et al. 2019) |
| **Class transformation** | Revert the label so one classifier learns uplift directly | Inverse-propensity reweighted for the 85/15 split (Jaśkowski & Jaroszewicz 2012) |

Base learners are pluggable: LightGBM, XGBoost, histogram GBDT, or
logistic/linear regression.

**Evaluation** (`src/uplift/evaluation.py`, all from scratch): Qini curve &
coefficient, uplift/AUUC curve & score, uplift@k, and weighted-average uplift.

---

## Repository layout

```
src/uplift/            Installable package — the real work lives here
  config.py            Paths, dataset schema, ExperimentConfig
  data.py              Download, load, stratified sample, uplift-aware split
  models.py            S- / T- / X-learner + class transformation
  evaluation.py        Qini, AUUC, uplift@k, bootstrap CIs, calibration
  business.py          profit curves, ROI, optimal targeting depth
  pipeline.py          load → split → fit → evaluate, in one call
scripts/
  download_data.py     Fetch the full ~290 MB Criteo file
  make_sample.py       Build the committed sample (subsample real, or synthesize)
  run_experiment.py    Train all models, write reports/metrics.csv + figures
notebooks/             Narrative analysis (EDA → meta-learners → business case)
tests/                 pytest suite for metrics, models, and the pipeline
data/sample/           Small parquet sample so the repo runs out of the box
reports/figures/       Generated Qini / uplift curves
```

---

## Quickstart

```bash
pip install -r requirements.txt        # see docs/INSTALL.md (note on causalml)

python scripts/make_sample.py          # build data/sample (synthetic if no raw file)
python scripts/run_experiment.py       # train 4 models → reports/metrics.csv + figures
pytest -q                              # run the test suite
```

Run on the full dataset:

```bash
python scripts/download_data.py
python scripts/run_experiment.py --use-raw --n-rows 2000000 --base-learner lgbm
```

Use the package directly:

```python
from uplift import data, pipeline
from uplift.config import ExperimentConfig

df = data.load_sample()
result = pipeline.run_experiment(df, ExperimentConfig(outcome="visit"))
print(result.metrics)          # ranked metrics table
print(result.ate)              # observed average treatment effect
```

---

## Notebooks

Three notebooks, **all runnable** against the `uplift` package and the committed
sample (no Kaggle paths, no manual setup — each is executed with outputs):

0. [**Flagship analysis**](notebooks/00_flagship_uplift_analysis.ipynb) — the
   end-to-end story: data → ATE → 4 models → Qini/AUUC → bootstrap CI →
   calibration → targeting payoff.
1. [**Exploratory data analysis**](notebooks/01_exploratory_data_analysis.ipynb)
   — base rates, two-proportion significance tests, a covariate-balance check
   that randomization holds, and the four uplift segments.
2. [**Business case**](notebooks/02_business_case_profit_curves.ipynb) — profit
   curves and ROI: the optimal targeting depth and what uplift targeting earns
   vs. treating everyone or targeting at random (`src/uplift/business.py`).

---

## Dataset

Criteo AI Lab Uplift v2.1: ~13.9M users, 12 anonymized features (`f0`–`f11`),
randomized `treatment` (~85% treated), an `exposure` mediator, and `visit` /
`conversion` outcomes. See [data/README.md](data/README.md) for schema and
download details.

## References

- Diemert, Betlei, Renaudin, Amini. *A Large Scale Benchmark for Uplift Modeling*, AdKDD 2018.
- Künzel, Sekhon, Bickel, Yu. *Metalearners for estimating heterogeneous treatment effects*, PNAS 2019.
- Jaśkowski & Jaroszewicz. *Uplift modeling for clinical trial data*, ICML 2012.
- Gutierrez & Gérardy. *Causal Inference and Uplift Modelling: A Review*, 2017.

## License

MIT — see [LICENSE](LICENSE).
