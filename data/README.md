# Data

This project uses the **Criteo AI Lab Uplift Prediction** dataset (v2.1) - a
large-scale randomized controlled trial released for incrementality / uplift
research.

| | |
|---|---|
| Rows | ~13.9 million users |
| Features | `f0`-`f11`, dense anonymized floats |
| Treatment | `treatment` (1 = targeted, 0 = control); ~85% treated |
| Mediator | `exposure` (actually saw the ad) |
| Outcomes | `visit` (denser), `conversion` (rare) |
| Design | Randomized - supports unbiased uplift evaluation |

## Layout

```
data/
├── raw/        # full Criteo file (gitignored, ~290 MB compressed)
└── sample/     # small committed parquet sample so the repo runs out-of-the-box
```

## Getting the data

```bash
# 1. Full dataset (optional, large)
python scripts/download_data.py

# 2. (Re)build the committed sample
python scripts/make_sample.py            # subsamples raw if present, else synthesizes
```

The committed `sample/criteo_sample.parquet` lets the notebooks, tests, and the
experiment runner work immediately. **It is intentionally *synthetic*** - a
Criteo-like dataset (`scripts/make_sample.py`, `synthesize()`) matching the real
schema, ~85/15 treatment imbalance, and low base rates, with a deliberately
stronger heterogeneous effect than the real data so the offline demo is legible.

Why synthetic and not a real subsample? Criteo releases this data for research
but does not grant redistribution rights, so re-committing even a slice of it to
a public repo is avoided here. The **headline results in the top-level README are
computed on the full, real 13.9M-row dataset** - download it, then run
`python scripts/run_experiment.py --use-raw`. Only *derived* artifacts -
aggregated Qini/uplift/calibration curves and the metrics table - are committed.

Source: <http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz>
Citation: Diemert et al., *A Large Scale Benchmark for Uplift Modeling* (AdKDD 2018).
