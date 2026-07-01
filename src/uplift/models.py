"""Uplift estimators with a common ``fit`` / ``predict_uplift`` interface.

Four families, spanning the standard taxonomy:

* **S-learner**  - one model on ``[X, t]``; uplift = p(y|X,t=1) - p(y|X,t=0).
* **T-learner**  - separate treated/control outcome models; uplift = mu1 - mu0.
* **X-learner**  - Kunzel et al. (2019); imputes individual effects then models
  them, blended by propensity. Strong when treatment groups are imbalanced
  (exactly the Criteo case: ~85% treated).
* **Class transformation** - Jaskowski & Jaroszewicz (2012); reverts the label
  so a single classifier estimates uplift directly.

Base learners are pluggable (LightGBM, XGBoost, gradient-boosted trees, or
logistic/linear regression) via :func:`build_classifier` / :func:`build_regressor`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Base-learner factories
# --------------------------------------------------------------------------- #
def build_classifier(kind: str = "lgbm", random_state: int = 42, **overrides):
    """Return an unfitted sklearn-compatible probabilistic classifier."""
    kind = kind.lower()
    if kind == "lgbm":
        from lightgbm import LGBMClassifier

        params = dict(n_estimators=200, num_leaves=31, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                      random_state=random_state, verbose=-1)
        params.update(overrides)
        return LGBMClassifier(**params)
    if kind == "xgb":
        from xgboost import XGBClassifier

        params = dict(n_estimators=200, max_depth=5, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                      eval_metric="logloss", random_state=random_state)
        params.update(overrides)
        return XGBClassifier(**params)
    if kind == "gbdt":
        from sklearn.ensemble import HistGradientBoostingClassifier

        params = dict(max_iter=200, learning_rate=0.05, random_state=random_state)
        params.update(overrides)
        return HistGradientBoostingClassifier(**params)
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        params = dict(max_iter=1000, random_state=random_state)
        params.update(overrides)
        return make_pipeline(StandardScaler(), LogisticRegression(**params))
    raise ValueError(f"unknown classifier kind: {kind!r}")


def build_regressor(kind: str = "lgbm", random_state: int = 42, **overrides):
    """Return an unfitted sklearn-compatible regressor (for X-learner stages)."""
    kind = kind.lower()
    if kind == "lgbm":
        from lightgbm import LGBMRegressor

        params = dict(n_estimators=200, num_leaves=31, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                      random_state=random_state, verbose=-1)
        params.update(overrides)
        return LGBMRegressor(**params)
    if kind == "xgb":
        from xgboost import XGBRegressor

        params = dict(n_estimators=200, max_depth=5, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                      random_state=random_state)
        params.update(overrides)
        return XGBRegressor(**params)
    if kind == "gbdt":
        from sklearn.ensemble import HistGradientBoostingRegressor

        params = dict(max_iter=200, learning_rate=0.05, random_state=random_state)
        params.update(overrides)
        return HistGradientBoostingRegressor(**params)
    if kind == "linreg":
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(StandardScaler(), Ridge(**overrides))
    raise ValueError(f"unknown regressor kind: {kind!r}")


def _proba(model, X) -> np.ndarray:
    """Positive-class probability from a fitted classifier."""
    p = model.predict_proba(X)
    return p[:, 1] if p.ndim == 2 else np.asarray(p).ravel()


def _as_frame(X) -> pd.DataFrame:
    """Coerce to a DataFrame with stable column names.

    Keeping a named DataFrame through fit *and* predict avoids the sklearn /
    LightGBM 'feature names' mismatch warning and keeps column order pinned.
    """
    if isinstance(X, pd.DataFrame):
        return X
    X = np.asarray(X)
    return pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])


def _mask(a) -> np.ndarray:
    return np.asarray(a).ravel()


# --------------------------------------------------------------------------- #
# Estimators
# --------------------------------------------------------------------------- #
class BaseUpliftModel(ABC):
    """Common interface: ``fit(X, y, t)`` then ``predict_uplift(X)``."""

    name: str = "base"

    @abstractmethod
    def fit(self, X, y, t) -> BaseUpliftModel: ...

    @abstractmethod
    def predict_uplift(self, X) -> np.ndarray: ...


class SLearner(BaseUpliftModel):
    name = "S-learner"

    _T = "__treatment__"

    def __init__(self, base_learner: str = "lgbm", random_state: int = 42, **kw):
        self.model = build_classifier(base_learner, random_state, **kw)

    def fit(self, X, y, t):
        X = _as_frame(X)
        Xt = X.assign(**{self._T: _mask(t)})
        self.model.fit(Xt, _mask(y))
        return self

    def predict_uplift(self, X):
        X = _as_frame(X)
        p1 = _proba(self.model, X.assign(**{self._T: 1}))
        p0 = _proba(self.model, X.assign(**{self._T: 0}))
        return p1 - p0


class TLearner(BaseUpliftModel):
    name = "T-learner"

    def __init__(self, base_learner: str = "lgbm", random_state: int = 42, **kw):
        self.m_treat = build_classifier(base_learner, random_state, **kw)
        self.m_ctrl = build_classifier(base_learner, random_state, **kw)

    def fit(self, X, y, t):
        X, y, t = _as_frame(X), _mask(y), _mask(t)
        self.m_treat.fit(X[t == 1], y[t == 1])
        self.m_ctrl.fit(X[t == 0], y[t == 0])
        return self

    def predict_uplift(self, X):
        X = _as_frame(X)
        return _proba(self.m_treat, X) - _proba(self.m_ctrl, X)


class XLearner(BaseUpliftModel):
    """X-learner (Kunzel et al., 2019) for binary outcomes."""

    name = "X-learner"

    def __init__(self, base_learner: str = "lgbm", random_state: int = 42, **kw):
        self.mu1 = build_classifier(base_learner, random_state, **kw)
        self.mu0 = build_classifier(base_learner, random_state, **kw)
        self.tau1 = build_regressor(base_learner, random_state, **kw)
        self.tau0 = build_regressor(base_learner, random_state, **kw)
        self._base, self._rs = base_learner, random_state

    def fit(self, X, y, t):
        X, y, t = _as_frame(X), _mask(y).astype(float), _mask(t)
        Xt, yt = X[t == 1], y[t == 1]
        Xc, yc = X[t == 0], y[t == 0]

        # Stage 1: outcome models per arm.
        self.mu1.fit(Xt, yt)
        self.mu0.fit(Xc, yc)

        # Stage 2: imputed individual treatment effects, regressed per arm.
        d1 = yt - _proba(self.mu0, Xt)          # treated: actual - counterfactual control
        d0 = _proba(self.mu1, Xc) - yc          # control: counterfactual treated - actual
        self.tau1.fit(Xt, d1)
        self.tau0.fit(Xc, d0)

        # Propensity for blending (constant suffices for an RCT, but learn it
        # so the estimator also behaves on observational data).
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self.propensity = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=self._rs)
        ).fit(X, t)
        return self

    def predict_uplift(self, X):
        X = _as_frame(X)
        g = _proba(self.propensity, X)
        return g * self.tau0.predict(X) + (1 - g) * self.tau1.predict(X)


class ClassTransformation(BaseUpliftModel):
    """Revert-label / class-transformation uplift model.

    Defines z = 1 when treatment and outcome 'agree' (treated-positive or
    control-negative). A classifier predicting ``P(z=1)`` recovers uplift via
    ``2 * P(z=1) - 1`` when treatment is balanced. For imbalanced assignment
    (Criteo is ~85/15) we apply inverse-propensity sample weights so the
    estimate stays unbiased without throwing away data.
    """

    name = "Class transformation"

    def __init__(self, base_learner: str = "lgbm", random_state: int = 42,
                 reweight: bool = True, **kw):
        self.model = build_classifier(base_learner, random_state, **kw)
        self.reweight = reweight

    def fit(self, X, y, t):
        X, y, t = _as_frame(X), _mask(y), _mask(t)
        z = ((t == 1) & (y == 1)) | ((t == 0) & (y == 0))
        sample_weight = None
        if self.reweight:
            p = t.mean()                       # marginal treatment probability
            sample_weight = np.where(t == 1, 0.5 / p, 0.5 / (1 - p))
        self.model.fit(X, z.astype(int), sample_weight=sample_weight)
        return self

    def predict_uplift(self, X):
        return 2.0 * _proba(self.model, _as_frame(X)) - 1.0


MODEL_REGISTRY: dict[str, type[BaseUpliftModel]] = {
    "s-learner": SLearner,
    "t-learner": TLearner,
    "x-learner": XLearner,
    "class-transform": ClassTransformation,
}


def build_model(
    name: str, base_learner: str = "lgbm", random_state: int = 42, **kw
) -> BaseUpliftModel:
    """Instantiate an uplift model by registry key."""
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"unknown model {name!r}; choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[key](base_learner=base_learner, random_state=random_state, **kw)
