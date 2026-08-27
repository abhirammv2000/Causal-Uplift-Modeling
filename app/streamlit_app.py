"""Interactive uplift targeting & ROI explorer.

Trains the uplift models on the committed sample (cached), then lets you set the
value of an incremental conversion and the cost per treatment and see, live:

* the profit curve vs. targeting depth, with the profit-maximising cut-off,
* what uplift targeting earns over treating everyone and over random targeting,
* the Qini curve and how uplift concentrates in the top ranks.

Run locally:
    pip install streamlit
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uplift import business, data, evaluation, pipeline  # noqa: E402
from uplift.config import ExperimentConfig  # noqa: E402

st.set_page_config(page_title="Uplift Targeting Explorer", layout="wide")


@st.cache_resource(show_spinner="Training uplift models on the sample...")
def _train(outcome: str, base_learner: str):
    """Fit the models once and cache them across reruns."""
    df = data.load_sample()
    cfg = ExperimentConfig(outcome=outcome, base_learner=base_learner)
    result = pipeline.run_experiment(df, cfg)
    return result


st.title(" Uplift Targeting & ROI Explorer")
st.caption(
    "Who should we treat? Ranking by **incremental** conversions, not raw "
    "propensity. Trained on the committed synthetic Criteo-like sample - the "
    "methodology is identical on the full real dataset."
)

with st.sidebar:
    st.header("Setup")
    outcome = st.selectbox("Outcome", ["visit", "conversion"], index=0)
    base_learner = st.selectbox("Base learner", ["lgbm", "gbdt", "logreg"], index=0)
    st.markdown("---")
    st.header("Economics")
    value = st.number_input("Value per incremental conversion ($)", 0.1, 1000.0, 10.0, 0.5)
    cost = st.number_input("Cost per treatment ($)", 0.0, 100.0, 0.10, 0.05)

result = _train(outcome, base_learner)
best_name = result.metrics.index[0]
model_name = st.sidebar.selectbox(
    "Model", list(result.metrics.index),
    index=list(result.metrics.index).index(best_name),
)

y_te, t_te = result.y_test, result.t_test
uplift = result.predictions[model_name]

curve = business.profit_curve(y_te, t_te, uplift, value, cost)
decision = business.optimal_targeting(y_te, t_te, uplift, value, cost)

# --- Headline numbers ----------------------------------------------------- #
c1, c2, c3, c4 = st.columns(4)
c1.metric("Observed ATE", f"{result.ate:+.4f}")
c2.metric("Best targeting depth", f"{decision.best_fraction:.0%}")
c3.metric("Profit vs. treat-all", f"${decision.uplift_over_treat_all:,.0f}")
c4.metric("Profit vs. random", f"${decision.uplift_over_random:,.0f}")

left, right = st.columns(2)

with left:
    st.subheader("Profit curve")
    ax = business.plot_profit_curve(curve, decision)
    st.pyplot(ax.figure)
    plt.close(ax.figure)

with right:
    st.subheader("Qini curve")
    ax = evaluation.plot_qini({model_name: result.qini_curves[model_name]})
    st.pyplot(ax.figure)
    plt.close(ax.figure)

st.subheader("Model leaderboard (uplift metrics)")
st.dataframe(result.metrics.round(4))

st.info(
    f"At ${value:.2f}/conversion and ${cost:.2f}/treatment, **{model_name}** is "
    f"most profitable targeting the top **{decision.best_fraction:.0%}** of users "
    f" - earning **${decision.uplift_over_treat_all:,.0f}** more than treating "
    f"everyone. Profit is read off the uplift curve, estimated from the "
    f"randomized treated/control arms, so it is causal - not a correlational guess."
)
