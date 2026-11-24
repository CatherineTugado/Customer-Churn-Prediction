"""Turn churn probabilities into a retention-campaign decision.

Assumptions (edit to match your business):
- Every targeted customer receives a retention offer that costs OFFER_COST.
- A true churner who receives the offer is saved with probability SAVE_RATE.
- A saved customer is worth MONTHS_OF_VALUE x their MonthlyCharges.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OFFER_COST = 50.0       # USD per targeted customer
SAVE_RATE = 0.30        # share of contacted churners who stay
MONTHS_OF_VALUE = 12    # revenue horizon of a saved customer


def campaign_profit(y_true, proba, monthly_charges, threshold: float) -> float:
    y_true = np.asarray(y_true)
    target = np.asarray(proba) >= threshold
    value_saved = SAVE_RATE * MONTHS_OF_VALUE * np.asarray(monthly_charges)
    gains = (target & (y_true == 1)) * value_saved
    return float(gains.sum() - target.sum() * OFFER_COST)


def profit_curve(y_true, proba, monthly_charges, thresholds=None) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)
    rows = []
    for t in thresholds:
        target = np.asarray(proba) >= t
        rows.append({
            "threshold": t,
            "customers_targeted": int(target.sum()),
            "churners_caught": int((target & (np.asarray(y_true) == 1)).sum()),
            "profit": campaign_profit(y_true, proba, monthly_charges, t),
        })
    return pd.DataFrame(rows)


def lift_table(y_true, proba, n_bins: int = 10) -> pd.DataFrame:
    """Decile table: how many churners each slice of the ranked list contains."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(proba)})
    df = df.sort_values("p", ascending=False).reset_index(drop=True)
    df["decile"] = pd.qcut(df.index, n_bins, labels=range(1, n_bins + 1))
    out = df.groupby("decile", observed=True).agg(customers=("y", "size"), churners=("y", "sum"))
    out["churn_rate"] = out["churners"] / out["customers"]
    out["cum_share_of_churners"] = out["churners"].cumsum() / out["churners"].sum()
    out["lift"] = out["churn_rate"] / df["y"].mean()
    return out
