"""
app.py — Supply Chain Delay Risk Dashboard
-----------------------------------------------------------------------------
A Streamlit dashboard covering:
    1. Dataset overview
    2. Predictions & insights (interactive what-if form)
    3. SHAP explainability
    4. Model performance metrics
    5. Data drift checks (reference vs. current)
    6. Responsible AI checklist (fairness, privacy, consent, explainability)

Expects the artifacts produced by train_model.py in ./artifacts/
Run:
    streamlit run app.py
-----------------------------------------------------------------------------
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
from scipy import stats
from sklearn.metrics import confusion_matrix, roc_curve

ARTIFACTS = Path("artifacts")

st.set_page_config(
    page_title="Supply Chain Delay Risk Dashboard",
    page_icon="📦",
    layout="wide",
)


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_resource
def load_model():
    return joblib.load(ARTIFACTS / "model.pkl")


@st.cache_data
def load_metrics():
    with open(ARTIFACTS / "metrics.json") as f:
        return json.load(f)


@st.cache_data
def load_feature_importance():
    return pd.read_csv(ARTIFACTS / "feature_importance.csv")


@st.cache_resource
def load_shap():
    return joblib.load(ARTIFACTS / "shap_values.pkl")


@st.cache_data
def load_shap_sample():
    return pd.read_parquet(ARTIFACTS / "shap_sample.parquet")


@st.cache_data
def load_reference():
    return pd.read_parquet(ARTIFACTS / "reference_data.parquet")


@st.cache_data
def load_current():
    return pd.read_parquet(ARTIFACTS / "current_data.parquet")


@st.cache_data
def load_fairness():
    return pd.read_parquet(ARTIFACTS / "fairness_slice.parquet")


@st.cache_data
def load_schema():
    with open(ARTIFACTS / "feature_schema.json") as f:
        return json.load(f)


if not (ARTIFACTS / "model.pkl").exists():
    st.error(
        "No artifacts found in ./artifacts/. Run `python train_model.py --data "
        "<your_csv>` first — see the README."
    )
    st.stop()

model = load_model()
metrics = load_metrics()
schema = load_schema()
NUMERIC = schema["numeric"]
BINARY = schema["binary"]
CATEGORICAL = schema["categorical"]
FEATURES = NUMERIC + BINARY + CATEGORICAL

st.title("📦 Supply Chain Delay Risk Dashboard")
st.caption(
    "Predicts whether an order is likely to be delivered late, and explains why."
)

tabs = st.tabs(
    [
        "🗂️ Overview",
        "🔮 Predictions & Insights",
        "🧠 SHAP Explainability",
        "📊 Model Metrics",
        "🌊 Data Drift",
        "✅ Responsible AI Checklist",
    ]
)

# --------------------------------------------------------------------------- #
# 1. Overview
# --------------------------------------------------------------------------- #
with tabs[0]:
    st.subheader("Dataset Overview")
    ref = load_reference()
    cur = load_current()
    full = pd.concat([ref, cur], ignore_index=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total orders", f"{len(full):,}")
    c2.metric("Delayed orders", f"{int(full['is_delayed'].sum()):,}")
    c3.metric("Delay rate", f"{full['is_delayed'].mean():.1%}")
    c4.metric("Avg. quantity / order", f"{full['quantity'].mean():.1f}")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            full, x="quantity", nbins=50, title="Quantity Distribution", log_y=True
        )
        st.plotly_chart(fig, width='stretch')
    with col2:
        by_cat = (
            full.groupby("category")["is_delayed"]
            .agg(["mean", "count"])
            .query("count > 200")
            .sort_values("mean", ascending=False)
            .head(15)
            .reset_index()
        )
        fig = px.bar(
            by_cat,
            x="mean",
            y="category",
            orientation="h",
            title="Top 15 Categories by Delay Rate (n > 200)",
            labels={"mean": "Delay rate"},
        )
        st.plotly_chart(fig, width='stretch')

    by_country = (
        full.groupby("country")["is_delayed"].agg(["mean", "count"]).reset_index()
    )
    fig = px.bar(
        by_country.sort_values("count", ascending=False).head(15),
        x="country",
        y="mean",
        title="Delay Rate by Country (top 15 by volume)",
        labels={"mean": "Delay rate"},
    )
    st.plotly_chart(fig, width='stretch')

# --------------------------------------------------------------------------- #
# 2. Predictions & Insights
# --------------------------------------------------------------------------- #
with tabs[1]:
    st.subheader("What-if Prediction")
    st.write("Adjust the order attributes below to see the predicted delay risk.")

    left, right = st.columns([1, 1.3])
    with left:
        with st.form("predict_form"):
            quantity = st.number_input("Quantity", min_value=1, value=2)
            sales = st.number_input("Sales value ($)", min_value=0.0, value=150.0)
            freight_value = st.number_input("Freight value ($)", min_value=0.0, value=18.0)
            month = st.slider("Month", 1, 12, 6)
            day_of_week = st.slider("Day of week (0=Mon)", 0, 6, 2)
            is_weekend = st.checkbox("Weekend order")
            is_holiday = st.checkbox("Near a holiday")
            country = st.selectbox("Country", schema["categories"]["country"])
            region = st.selectbox("Region", schema["categories"]["region"])
            state = st.selectbox("State", schema["categories"]["state"])
            category = st.selectbox("Product category", schema["categories"]["category"])
            source = st.selectbox("Data source", schema["categories"]["source"])
            submitted = st.form_submit_button("Predict delay risk")

    with right:
        if submitted:
            row = pd.DataFrame(
                [
                    {
                        "quantity": quantity,
                        "sales": sales,
                        "freight_value": freight_value,
                        "sales_per_quantity": sales / max(quantity, 1),
                        "freight_per_quantity": freight_value / max(quantity, 1),
                        "year": 2018,
                        "month": month,
                        "quarter": (month - 1) // 3 + 1,
                        "week": month * 4,
                        "day_of_week": day_of_week,
                        "is_weekend": int(is_weekend),
                        "is_holiday": int(is_holiday),
                        "is_month_start": 0,
                        "is_month_end": 0,
                        "is_quarter_start": 0,
                        "is_quarter_end": 0,
                        "has_freight_value": int(freight_value > 0),
                        "country": country,
                        "region": region,
                        "state": state,
                        "category": category,
                        "source": source,
                    }
                ]
            )[FEATURES]

            proba = model.predict_proba(row)[0, 1]
            pred = "🔴 Likely DELAYED" if proba >= 0.5 else "🟢 Likely ON TIME"

            st.metric("Predicted delay probability", f"{proba:.1%}")
            st.write(f"### {pred}")

            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=proba * 100,
                    title={"text": "Delay risk (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "darkred" if proba >= 0.5 else "seagreen"},
                        "steps": [
                            {"range": [0, 50], "color": "#d4f4dd"},
                            {"range": [50, 100], "color": "#f8d7da"},
                        ],
                    },
                )
            )
            st.plotly_chart(gauge, width='stretch')

            # local SHAP explanation for this single prediction
            explainer = shap.TreeExplainer(model.named_steps["model"])
            transformed = model.named_steps["preprocess"].transform(row)
            sv = explainer.shap_values(transformed)
            sv_pos = sv[1][0] if isinstance(sv, list) else (sv[0, :, 1] if sv.ndim == 3 else sv[0])

            local_df = pd.DataFrame(
                {"feature": FEATURES, "impact": sv_pos}
            ).sort_values("impact", key=abs, ascending=False).head(8)
            fig = px.bar(
                local_df,
                x="impact",
                y="feature",
                orientation="h",
                title="Top factors driving this prediction",
                color="impact",
                color_continuous_scale=["seagreen", "lightgray", "darkred"],
            )
            st.plotly_chart(fig, width='stretch')
            st.caption(
                "Positive (red) values push the prediction toward **delayed**; "
                "negative (green) values push toward **on time**."
            )
        else:
            st.info("Fill in the form and click **Predict delay risk** to see a result.")

# --------------------------------------------------------------------------- #
# 3. SHAP Explainability
# --------------------------------------------------------------------------- #
with tabs[2]:
    st.subheader("Global Model Explainability (SHAP)")
    shap_data = load_shap()
    shap_sample = load_shap_sample()
    shap_values = shap_data["shap_values"]

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Beeswarm summary plot**")
        fig, ax = plt.subplots()
        shap.summary_plot(
            shap_values, shap_sample[FEATURES], show=False, plot_size=(7, 6)
        )
        st.pyplot(fig, clear_figure=True)
    with col2:
        st.write("**Mean |SHAP value| — feature impact ranking**")
        mean_abs = np.abs(shap_values).mean(axis=0)
        imp_df = pd.DataFrame({"feature": FEATURES, "mean_abs_shap": mean_abs}).sort_values(
            "mean_abs_shap", ascending=True
        )
        fig = px.bar(imp_df, x="mean_abs_shap", y="feature", orientation="h")
        st.plotly_chart(fig, width='stretch')

    st.write("**Dependence plot**")
    feat = st.selectbox("Feature", FEATURES, index=FEATURES.index("freight_value"))
    fig, ax = plt.subplots()
    shap.dependence_plot(
        feat, shap_values, shap_sample[FEATURES], show=False, ax=ax, interaction_index=None
    )
    st.pyplot(fig, clear_figure=True)

# --------------------------------------------------------------------------- #
# 4. Model Metrics
# --------------------------------------------------------------------------- #
with tabs[3]:
    st.subheader("Model Performance")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", metrics["accuracy"])
    c2.metric("Precision", metrics["precision"])
    c3.metric("Recall", metrics["recall"])
    c4.metric("F1", metrics["f1"])
    c5.metric("ROC AUC", metrics["roc_auc"])

    col1, col2 = st.columns(2)
    with col1:
        cm = np.array(metrics["confusion_matrix"])
        fig = px.imshow(
            cm,
            text_auto=True,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["On time", "Delayed"],
            y=["On time", "Delayed"],
            title="Confusion Matrix",
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig, width='stretch')
    with col2:
        fi = load_feature_importance().head(15)
        fig = px.bar(
            fi.sort_values("importance"),
            x="importance",
            y="feature",
            orientation="h",
            title="Top 15 Feature Importances (Random Forest)",
        )
        st.plotly_chart(fig, width='stretch')

    st.caption(
        f"Trained on {metrics['n_train']:,} orders, evaluated on {metrics['n_test']:,} "
        f"held-out orders. Actual delay rate in test set: {metrics['positive_rate_actual']:.1%}, "
        f"predicted: {metrics['positive_rate_predicted']:.1%}."
    )

# --------------------------------------------------------------------------- #
# 5. Data Drift
# --------------------------------------------------------------------------- #
with tabs[4]:
    st.subheader("Data Drift: Reference vs. Current")
    st.caption(
        "Reference = earliest 70% of orders by date (training-time distribution). "
        "Current = most recent 30% (stand-in for freshly incoming production data)."
    )
    ref = load_reference()
    cur = load_current()

    rows = []
    for col in NUMERIC:
        stat, pvalue = stats.ks_2samp(ref[col].dropna(), cur[col].dropna())
        # Population Stability Index for extra robustness
        bins = np.histogram_bin_edges(ref[col].dropna(), bins=10)
        ref_pct = np.histogram(ref[col].dropna(), bins=bins)[0] / len(ref[col].dropna())
        cur_pct = np.histogram(cur[col].dropna(), bins=bins)[0] / max(len(cur[col].dropna()), 1)
        ref_pct = np.where(ref_pct == 0, 1e-6, ref_pct)
        cur_pct = np.where(cur_pct == 0, 1e-6, cur_pct)
        psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
        rows.append(
            {
                "feature": col,
                "ks_statistic": round(stat, 4),
                "p_value": round(pvalue, 4),
                "psi": round(psi, 4),
                "drift_detected": bool(pvalue < 0.05 or psi > 0.2),
            }
        )
    drift_df = pd.DataFrame(rows).sort_values("psi", ascending=False)

    n_drifted = drift_df["drift_detected"].sum()
    if n_drifted:
        st.warning(f"⚠️ Drift detected in {n_drifted} of {len(drift_df)} numeric features.")
    else:
        st.success("✅ No significant drift detected in numeric features.")

    st.dataframe(
        drift_df.style.apply(
            lambda r: ["background-color: #fddede" if r.drift_detected else "" for _ in r],
            axis=1,
        ),
        width='stretch',
    )
    st.caption(
        "**KS test**: p < 0.05 suggests the distributions differ. "
        "**PSI** (Population Stability Index): >0.2 = notable shift, >0.25 = major shift."
    )

    feat = st.selectbox("Inspect a feature's distribution", NUMERIC, key="drift_feat")
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=ref[feat], name="Reference", opacity=0.6, histnorm="probability"))
    fig.add_trace(go.Histogram(x=cur[feat], name="Current", opacity=0.6, histnorm="probability"))
    fig.update_layout(barmode="overlay", title=f"Distribution shift — {feat}")
    st.plotly_chart(fig, width='stretch')

    st.write("**Categorical feature drift** (share of top categories)")
    cat_feat = st.selectbox("Categorical feature", CATEGORICAL, key="drift_cat")
    ref_share = ref[cat_feat].value_counts(normalize=True).head(10)
    cur_share = cur[cat_feat].value_counts(normalize=True).reindex(ref_share.index).fillna(0)
    comp = pd.DataFrame({"Reference": ref_share, "Current": cur_share}).reset_index()
    comp.columns = [cat_feat, "Reference", "Current"]
    fig = px.bar(comp.melt(id_vars=cat_feat), x=cat_feat, y="value", color="variable", barmode="group")
    st.plotly_chart(fig, width='stretch')

# --------------------------------------------------------------------------- #
# 6. Responsible AI Checklist
# --------------------------------------------------------------------------- #
with tabs[5]:
    st.subheader("✅ Responsible AI Checklist")

    st.markdown("### 1. Fairness & Bias")
    st.write(
        "The model uses regional fields (`country`, `region`, `state`). We check "
        "whether the false positive rate (flagging on-time orders as delayed) and "
        "accuracy are consistent across regions, since a skew would deprioritize "
        "those routes unfairly."
    )
    fdf = load_fairness()
    slice_by = st.selectbox("Slice fairness metrics by", ["country", "region"])
    group_stats = []
    for g, gdf in fdf.groupby(slice_by):
        if len(gdf) < 30:
            continue
        cm = confusion_matrix(gdf["y_true"], gdf["y_pred"], labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) else np.nan
        acc = (tp + tn) / len(gdf)
        group_stats.append({slice_by: g, "n": len(gdf), "accuracy": acc, "false_positive_rate": fpr})
    gstats = pd.DataFrame(group_stats).sort_values("n", ascending=False).head(15)
    fig = px.bar(
        gstats.melt(id_vars=[slice_by, "n"], value_vars=["accuracy", "false_positive_rate"]),
        x=slice_by,
        y="value",
        color="variable",
        barmode="group",
        title=f"Accuracy & False Positive Rate by {slice_by} (groups with n≥30)",
    )
    st.plotly_chart(fig, width='stretch')
    spread = gstats["false_positive_rate"].max() - gstats["false_positive_rate"].min()
    st.caption(
        f"False positive rate spread across shown groups: {spread:.1%}. "
        "Large spreads warrant investigation before using predictions to deprioritize routes."
    )

    st.markdown("### 2. Privacy & Data Security")
    st.write(
        "- The dataset contains **aggregate** demand, sales and delivery records — no "
        "customer names, addresses, or other direct personal identifiers.\n"
        "- Fine-grained geographic aggregation (city/state level) could still indirectly "
        "reveal supplier volumes or business strategy, so access to raw exports is restricted.\n"
        "- No personally identifiable information (PII) is fed into this dashboard or any "
        "production API built on it."
    )

    st.markdown("### 3. Explainability & Transparency")
    st.write(
        "- The underlying model is a Random Forest, which is not inherently interpretable "
        "on its own.\n"
        "- SHAP (SHapley Additive exPlanations) is integrated throughout this dashboard — "
        "both globally (**SHAP Explainability** tab) and per-prediction (**Predictions & "
        "Insights** tab) — so warehouse managers and logistics coordinators can see exactly "
        "which factors (e.g. `freight_value`, `quantity`, `category`) drove a specific delay "
        "prediction, rather than trusting a black box."
    )

    st.markdown("### 4. Consent & Compliance")
    st.write(
        "- This dataset (merged from DataCo, Olist and M5) is sourced from openly published, "
        "research-oriented datasets — it was not collected under a production consent flow.\n"
        "- **Before any real-world deployment**, data collection and use must comply with "
        "applicable data protection regulation (e.g. GDPR, CCPA), and supply chain partners "
        "must explicitly consent to their delivery metrics being used for predictive analytics.\n"
        "- Predictions from this model should support human decision-making, not replace it — "
        "especially for decisions that could deprioritize service to specific regions."
    )

    st.info(
        "This checklist is a living document. Re-run it whenever the model, features, or "
        "data sources change."
    )

st.sidebar.header("About")
st.sidebar.write(
    "This dashboard predicts delivery delay risk for supply-chain orders and explains "
    "predictions with SHAP. Built with scikit-learn + SHAP + Streamlit."
)
st.sidebar.write(f"Model artifacts loaded from `{ARTIFACTS.resolve()}`")
