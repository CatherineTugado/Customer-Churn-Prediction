from pathlib import Path

import numpy as np
import pytest

from churn.business import campaign_profit
from churn.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, prepare

DATA = Path(__file__).resolve().parents[1] / "data" / "raw" / "telco_churn.csv"


def test_prepare_has_no_missing_values():
    X, y = prepare(DATA)
    assert X[NUMERIC_FEATURES].isna().sum().sum() == 0
    assert set(X.columns) == set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    assert set(y.unique()) == {0, 1}


def test_n_services_in_range():
    X, _ = prepare(DATA)
    assert X["n_services"].between(0, 6).all()


def test_campaign_profit_simple_case():
    # One churner targeted (worth 0.3 * 12 * 100 = 360), one non-churner targeted; offer costs 50 each.
    profit = campaign_profit(
        y_true=np.array([1, 0, 1]), proba=np.array([0.9, 0.8, 0.1]),
        monthly_charges=np.array([100.0, 100.0, 100.0]), threshold=0.5,
    )
    assert profit == pytest.approx(360 - 2 * 50)
