# Installation

```bash
git clone <your-fork-url>
cd Causal-Uplift-Modeling

python -m venv .venv
# Windows:  .venv\Scripts\activate
# Unix:     source .venv/bin/activate

pip install -r requirements.txt
# or, to import `uplift` from anywhere as a package:
pip install -e .
```

Tested on **Python 3.11–3.13**.

## A note on `causalml`

The notebooks reference `causalml` for cross-checking, but it is **optional and
intentionally not a hard dependency**. It ships native (Cython) extensions that
frequently fail to build on Python 3.13 / Windows. The core of this project —
the `uplift` package, every metric, and all four estimators — depends only on
`scikit-learn`, `scikit-uplift`, `xgboost`, and `lightgbm`, which install
cleanly everywhere.

If you want `causalml` and are on Python ≤ 3.11:

```bash
pip install causalml
```

## Verify the install

```bash
python scripts/make_sample.py        # builds data/sample/criteo_sample.parquet
python scripts/run_experiment.py     # trains 4 uplift models, writes reports/
pytest -q                            # runs the test suite
```
