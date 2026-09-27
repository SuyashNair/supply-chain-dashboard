# 📦 Supply Chain Delay Risk Dashboard

An end-to-end, responsible-AI-aware dashboard that predicts whether a supply
chain order will be delivered late, explains *why* with SHAP, tracks model
health, and checks incoming data for drift.

Built for the merged DataCo + Olist + M5 `supply_chain_clean_final.csv`
dataset, but the pipeline generalizes to any tabular delay/ops dataset with
minor column renaming in `train_model.py`.

## What's inside

| File | Purpose |
|---|---|
| `train_model.py` | Loads the CSV, engineers features, trains a `RandomForestClassifier`, evaluates it, computes SHAP values, and writes every artifact the dashboard needs into `artifacts/`. |
| `app.py` | The Streamlit dashboard: Overview, Predictions & Insights, SHAP Explainability, Model Metrics, Data Drift, Responsible AI Checklist. |
| `requirements.txt` | Pinned dependencies. |
| `tests/test_smoke.py` | Fast unit tests (no dataset needed) run in CI. |
| `.github/workflows/ci.yml` | Lints, tests, and sanity-checks the app on every push/PR. |

## Model & features

- **Target:** `is_delayed` (binary). Rows from the `M5` source are excluded
  from training because they carry no real delivery outcome (`is_delayed=0`
  there is a placeholder, not a label) — see the comment in
  `engineer_features()`.
- **Features:** order-level numerics (`quantity`, `sales`, `freight_value`,
  derived ratios), calendar features (`month`, `day_of_week`, `is_weekend`,
  `is_holiday`, …), and categoricals (`country`, `region`, `state`,
  `category`, `source`), ordinal-encoded for the tree model.
- **Model:** `RandomForestClassifier` (150 trees, max depth 8,
  `class_weight="balanced_subsample"`) inside a single `sklearn.Pipeline`
  (imputation + encoding + model), so `model.pkl` is self-contained.

## Quickstart — local or Colab

```bash
pip install -r requirements.txt

# 1) Train the model and generate all dashboard artifacts
python train_model.py --data supply_chain_clean_final.csv

# 2) Launch the dashboard
streamlit run app.py
```

For the full copy-paste Colab flow (including a live in-notebook preview via
`localtunnel` and pushing the repo to GitHub), see
[`Supply_Chain_Dashboard_Colab.ipynb`](./Supply_Chain_Dashboard_Colab.ipynb).

## Responsible AI

The dashboard's **Responsible AI Checklist** tab covers:

1. **Fairness & Bias** — false-positive-rate and accuracy parity across
   `country` / `region` slices, computed live from held-out predictions.
2. **Privacy & Data Security** — no PII in the dataset; access-control and
   aggregation notes.
3. **Explainability & Transparency** — SHAP integrated globally (beeswarm,
   feature-importance, dependence plots) and per-prediction (local
   explanation on every what-if prediction).
4. **Consent & Compliance** — notes on GDPR/CCPA obligations before any
   production deployment using real partner data.

## Data drift checks

The **Data Drift** tab splits the dataset chronologically (earliest 70% =
reference / training-time distribution, most recent 30% = "current"
production-like data) and runs, per numeric feature:

- **Kolmogorov–Smirnov test** (`p < 0.05` ⇒ distributions differ)
- **Population Stability Index (PSI)** (`> 0.2` ⇒ notable shift, `> 0.25` ⇒
  major shift)

plus a categorical top-category share comparison. In production, swap
`current_data.parquet` for a rolling window of live traffic to get ongoing
drift monitoring.

## Deploying

- **Streamlit Community Cloud** (free): push this repo to GitHub, then on
  [share.streamlit.io](https://share.streamlit.io) point it at `app.py`. It
  auto-redeploys on every push to `main` — no extra CI step needed for that.
- **GitHub Actions** (`.github/workflows/ci.yml`) lints and tests the code on
  every push/PR so a broken commit never reaches `main`.

## Repository

GitHub: https://github.com/SuyashNair/supply-chain-dashboard
