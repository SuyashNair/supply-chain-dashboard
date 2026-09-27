"""
Lightweight smoke tests that run in CI without needing the full dataset
or trained artifacts. They check that the code imports cleanly and that
core feature-engineering logic behaves as expected.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_model import (  # noqa: E402
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_pipeline,
    engineer_features,
)


def _fake_df(n=50):
    return pd.DataFrame(
        {
            "date": pd.date_range("2018-01-01", periods=n, freq="D"),
            "year": 2018,
            "month": [((i % 12) + 1) for i in range(n)],
            "quarter": 1,
            "week": 1,
            "day_of_week": [i % 7 for i in range(n)],
            "country": "United States",
            "region": "West",
            "state": "CA",
            "city": "LA",
            "product": "Widget",
            "category": ["A", "B"] * (n // 2),
            "sales": 100.0,
            "quantity": 2,
            "freight_value": 15.0,
            "sales_per_quantity": 50.0,
            "freight_per_quantity": 7.5,
            "is_weekend": 0,
            "is_holiday": 0,
            "is_month_start": 0,
            "is_month_end": 0,
            "is_quarter_start": 0,
            "is_quarter_end": 0,
            "has_freight_value": 1,
            "source": ["DataCo", "Olist"] * (n // 2),
            "is_delayed": [0, 1] * (n // 2),
        }
    )


def test_engineer_features_drops_m5_and_fills_categoricals():
    df = _fake_df()
    df = pd.concat(
        [df, pd.DataFrame([{**df.iloc[0].to_dict(), "source": "M5"}])], ignore_index=True
    )
    out = engineer_features(df)
    assert "M5" not in out["source"].unique()
    for col in CATEGORICAL_FEATURES:
        assert out[col].isna().sum() == 0


def test_pipeline_fits_and_predicts():
    df = engineer_features(_fake_df())
    features = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
    X, y = df[features], df["is_delayed"]
    pipe = build_pipeline()
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == len(y)
    assert set(preds).issubset({0, 1})
