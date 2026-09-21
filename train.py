import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import os

os.makedirs("model", exist_ok=True)

CATEGORICAL = ["country", "state", "category", "source"]
NUMERIC = ["quantity", "month", "day_of_week", "is_holiday", "is_weekend"]
TARGET = "is_delayed"

# We assume your CSV is in your Downloads folder. If you moved it, change this path!
df = pd.read_csv("../supply_chain_clean_final.csv", low_memory=False)
df = df.dropna(subset=[TARGET])
df[CATEGORICAL] = df[CATEGORICAL].fillna("Unknown")
df[NUMERIC] = df[NUMERIC].fillna(0)

X, y = df[CATEGORICAL + NUMERIC], df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
                         ("num", "passthrough", NUMERIC)])
model = Pipeline([("pre", pre), ("clf", RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42))])
model.fit(X_train, y_train)

joblib.dump(model, "model/delay_model.joblib")
X_test.to_csv("model/test_data.csv", index=False)
print("Model retrained successfully for your local version!")
