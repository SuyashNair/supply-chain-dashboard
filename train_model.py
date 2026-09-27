"""
train_model.py
-----------------------------------------------------------------------------
Trains a delivery-delay classifier on the supply chain dataset and saves
every artifact the Streamlit dashboard (app.py) needs:

    artifacts/
        model.pkl              trained RandomForestClassifier (in a Pipeline)
        metrics.json            accuracy / precision / recall / f1 / roc_auc / confusion matrix
        feature_importance.csv  RF impurity-based feature importance
        shap_values.pkl         SHAP values computed on a background sample
        shap_sample.parquet     the rows the SHAP values correspond to
        reference_data.parquet  "training-time" data distribution (drift baseline)
        current_data.parquet    most-recent slice of data ("production" data)
        feature_schema.json     which columns are numeric vs categorical
        fairness_slice.parquet  held-out test predictions + sensitive attributes

Run:
    python train_model.py --data supply_chain_clean_final.csv
-----------------------------------------------------------------------------
"""

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore")

NUMERIC_FEATURES = [
    "quantity",
    "sales",
    "freight_value",
    "sales_per_quantity",
    "freight_per_quantity",
    "year",
    "month",
    "quarter",
    "week",
    "day_of_week",
]
BINARY_FEATURES = [
    "is_weekend",
    "is_holiday",
    "is_month_start",
    "is_month_end",
    "is_quarter_start",
    "is_quarter_end",
    "has_freight_value",
]
CATEGORICAL_FEATURES = ["country", "region", "state", "category", "source"]
TARGET = "is_delayed"
SENSITIVE_ATTRS = ["country", "region"]  # used later for the fairness tab


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Light cleanup shared by training and the dashboard."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # M5 rows never carry a real delivery outcome (no delivery_days at all),
    # so is_delayed==0 there is a placeholder, not a label. Drop them so the
    # model isn't trained against fake negatives.
    df = df[df["source"].isin(["DataCo", "Olist"])].reset_index(drop=True)

    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("Unknown").astype(str)

    return df


def build_pipeline() -> Pipeline:
    numeric_transformer = SimpleImputer(strategy="median")
    categorical_transformer = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )
    binary_transformer = SimpleImputer(strategy="constant", fill_value=0)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("bin", binary_transformer, BINARY_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        min_samples_leaf=25,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )

    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def main(data_path: str, out_dir: str, sample_for_shap: int, test_size: float):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {data_path} ...")
    df = pd.read_csv(data_path, low_memory=False)
    df = engineer_features(df)
    print(f"Rows after filtering to labeled sources: {len(df):,}")

    features = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
    X = df[features]
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=test_size, random_state=42, stratify=y
    )

    print("Training RandomForestClassifier ...")
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    # ---- Metrics ---------------------------------------------------------
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "positive_rate_actual": round(float(y_test.mean()), 4),
        "positive_rate_predicted": round(float(y_pred.mean()), 4),
    }
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Metrics:", metrics)

    # ---- Feature importance ----------------------------------------------
    importances = pipe.named_steps["model"].feature_importances_
    fi = pd.DataFrame({"feature": features, "importance": importances}).sort_values(
        "importance", ascending=False
    )
    fi.to_csv(out / "feature_importance.csv", index=False)

    # ---- SHAP --------------------------------------------------------------
    print(f"Computing SHAP values on a sample of {sample_for_shap} test rows ...")
    shap_sample = X_test.sample(n=min(sample_for_shap, len(X_test)), random_state=42)
    transformed = pipe.named_steps["preprocess"].transform(shap_sample)
    explainer = shap.TreeExplainer(pipe.named_steps["model"])
    shap_values = explainer.shap_values(transformed)

    # RandomForestClassifier -> list [class0, class1] for older SHAP versions;
    # normalize to a single 2D array for the positive class.
    if isinstance(shap_values, list):
        shap_values_pos = shap_values[1]
    elif shap_values.ndim == 3:
        shap_values_pos = shap_values[:, :, 1]
    else:
        shap_values_pos = shap_values

    joblib.dump(
        {
            "shap_values": shap_values_pos,
            "feature_names": features,
            "expected_value": (
                explainer.expected_value[1]
                if isinstance(explainer.expected_value, (list, np.ndarray))
                else explainer.expected_value
            ),
        },
        out / "shap_values.pkl",
    )
    shap_sample.reset_index(drop=True).to_parquet(out / "shap_sample.parquet")

    # ---- Drift baselines ----------------------------------------------------
    # Reference = earliest 70% of dates, Current = most recent 30%.
    # This lets the dashboard demonstrate a real drift check without needing
    # a second, freshly-collected dataset.
    df_sorted = df.sort_values("date")
    cut = int(len(df_sorted) * 0.7)
    df_sorted.iloc[:cut][features + [TARGET, "date"]].to_parquet(out / "reference_data.parquet")
    df_sorted.iloc[cut:][features + [TARGET, "date"]].to_parquet(out / "current_data.parquet")

    # ---- Fairness slice (test set predictions + sensitive attrs) -----------
    fairness_df = df_test[SENSITIVE_ATTRS].copy().reset_index(drop=True)
    fairness_df["y_true"] = y_test.reset_index(drop=True)
    fairness_df["y_pred"] = y_pred
    fairness_df["y_proba"] = y_proba
    fairness_df.to_parquet(out / "fairness_slice.parquet")

    # ---- Schema + model -----------------------------------------------------
    with open(out / "feature_schema.json", "w") as f:
        json.dump(
            {
                "numeric": NUMERIC_FEATURES,
                "binary": BINARY_FEATURES,
                "categorical": CATEGORICAL_FEATURES,
                "target": TARGET,
                "categories": {c: sorted(df[c].unique().tolist())[:200] for c in CATEGORICAL_FEATURES},
            },
            f,
            indent=2,
        )

    joblib.dump(pipe, out / "model.pkl")
    print(f"\nAll artifacts written to {out.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="supply_chain_clean_final.csv")
    parser.add_argument("--out_dir", default="artifacts")
    parser.add_argument("--shap_sample", type=int, default=500)
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()
    main(args.data, args.out_dir, args.shap_sample, args.test_size)
