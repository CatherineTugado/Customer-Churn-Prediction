"""Train the final churn model and save it with its metrics.

Usage:
    python -m churn.train --data data/raw/telco_churn.csv --out models/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from churn.features import make_preprocessor, prepare

RANDOM_STATE = 42

# Chosen in notebooks/01_churn_analysis.ipynb (randomized search, 5-fold CV)
XGB_PARAMS = dict(
    n_estimators=500, learning_rate=0.02, max_depth=2, subsample=0.7,
    colsample_bytree=1.0, min_child_weight=3, reg_lambda=5.0,
    eval_metric="logloss", random_state=RANDOM_STATE,
)


def build_model(**overrides) -> Pipeline:
    params = {**XGB_PARAMS, **overrides}
    return Pipeline([
        ("prep", make_preprocessor(scale_numeric=False)),
        ("model", XGBClassifier(**params)),
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/telco_churn.csv")
    parser.add_argument("--out", default="models")
    args = parser.parse_args()

    X, y = prepare(args.data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    model = build_model().fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "pr_auc": round(average_precision_score(y_test, proba), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "churn_xgb.joblib")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
