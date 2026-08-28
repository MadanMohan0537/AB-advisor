"""Regression tests for experiment-data guardrails."""

import numpy as np
import pandas as pd
import pytest

from src.metrics import load_experiment


def test_rejects_single_arm_data():
    df = pd.DataFrame({"user_id": [1, 2], "variant": ["control", "control"], "conversion": [0, 1]})
    with pytest.raises(ValueError, match="exactly two"):
        load_experiment(df)


def test_rejects_users_assigned_to_both_arms():
    df = pd.DataFrame({"user_id": [1, 1], "variant": ["control", "treatment"], "conversion": [0, 1]})
    with pytest.raises(ValueError, match="multiple variants"):
        load_experiment(df)


def test_rejects_duplicate_wide_rows():
    df = pd.DataFrame(
        {"user_id": [1, 1, 2], "variant": ["control", "control", "treatment"], "conversion": [0, 0, 1]}
    )
    with pytest.raises(ValueError, match="Duplicate user rows"):
        load_experiment(df)


def test_split_drops_missing_metric_values_per_arm():
    df = pd.DataFrame(
        {"user_id": [1, 2, 3, 4], "variant": ["control", "control", "treatment", "treatment"], "score": [1, np.nan, 2, 3]}
    )
    control, treatment = load_experiment(df).split("score")
    assert control.tolist() == [1.0]
    assert treatment.tolist() == [2.0, 3.0]


@pytest.mark.parametrize("expected_split", [0, 1, -0.1, 1.1])
def test_rejects_invalid_expected_split(expected_split):
    df = pd.DataFrame({"user_id": [1, 2], "variant": ["control", "treatment"], "conversion": [0, 1]})
    with pytest.raises(ValueError, match="expected_split"):
        load_experiment(df, expected_split=expected_split)
