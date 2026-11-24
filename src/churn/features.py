"""Data loading, cleaning and feature engineering for the Telco churn dataset."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Churn"
ID_COL = "customerID"

SERVICE_COLS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

NUMERIC_FEATURES = [
    "tenure", "MonthlyCharges", "TotalCharges",
    "n_services", "avg_monthly_spend", "charge_increase_ratio",
]
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", *SERVICE_COLS, "Contract",
    "PaperlessBilling", "PaymentMethod", "tenure_band",
]


def load_raw(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix types and the 11 blank TotalCharges rows (brand-new customers, tenure 0)."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # Customers with tenure 0 have not been billed yet, so total charges = 0.
    df.loc[df["TotalCharges"].isna() & (df["tenure"] == 0), "TotalCharges"] = 0.0
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    if TARGET in df:
        df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Business-driven features."""
    df = df.copy()
    df["n_services"] = (df[SERVICE_COLS] == "Yes").sum(axis=1)
    # Average historical spend; for tenure 0 fall back to the current monthly charge.
    df["avg_monthly_spend"] = np.where(
        df["tenure"] > 0, df["TotalCharges"] / df["tenure"].clip(lower=1), df["MonthlyCharges"]
    )
    # > 1 means the customer now pays more than their historical average (price increase).
    df["charge_increase_ratio"] = df["MonthlyCharges"] / df["avg_monthly_spend"].replace(0, np.nan)
    df["charge_increase_ratio"] = df["charge_increase_ratio"].fillna(1.0)
    df["tenure_band"] = pd.cut(
        df["tenure"], bins=[-1, 6, 12, 24, 48, 72],
        labels=["0-6m", "7-12m", "13-24m", "25-48m", "49-72m"],
    ).astype(str)
    return df


def prepare(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    df = add_features(clean(load_raw(path)))
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    return X, y


def make_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    num = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        [
            ("num", num, NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
