"""CSV loading, type detection, and SRM."""

from __future__ import annotations

import pandas as pd

from src.bayesian import MetricType
from src.data_gen import generate_checkout_experiment, generate_long_format
from src.metrics import detect_metric_type, load_experiment


def test_detect_binary_and_poisson():
    df = generate_checkout_experiment(n_per_arm=800, seed=1)
    assert detect_metric_type(df["converted"]) is MetricType.BINARY
    assert detect_metric_type(df["bounce"]) is MetricType.BINARY
    assert detect_metric_type(df["events_count"]) is MetricType.POISSON
    assert detect_metric_type(df["revenue"]) is MetricType.HURDLE_LOGNORMAL
    assert detect_metric_type(df["time_on_page_sec"]) is MetricType.NORMAL


def test_wide_load_and_split():
    df = generate_checkout_experiment(n_per_arm=300, seed=2)
    data = load_experiment(df)
    assert data.format == "wide"
    assert data.n_control == 300
    assert data.n_treatment == 300
    assert not data.srm_flag
    control, treatment = data.split("converted")
    assert len(control) == 300
    assert set(data.metric_cols) >= {"converted", "revenue", "events_count"}


def test_long_format_pivots():
    wide = generate_checkout_experiment(n_per_arm=120, seed=3)
    long = generate_long_format(wide)
    data = load_experiment(long)
    assert data.format == "long"
    assert "converted" in data.metric_cols
    assert data.n_control == 120


def test_srm_flag_on_unequal_assignment():
    df = pd.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(1000)],
            "variant": ["control"] * 200 + ["treatment"] * 800,
            "converted": [0, 1] * 500,
        }
    )
    data = load_experiment(df)
    assert data.srm_flag
    assert data.srm_pvalue < 0.001


def test_variant_aliases():
    df = pd.DataFrame(
        {"uid": [1, 2, 3, 4], "group": ["A", "B", "a", "b"], "converted": [0, 1, 0, 1]}
    )
    data = load_experiment(df)
    assert data.control_label == "control"
    assert data.treatment_label == "treatment"
    assert data.n_control == 2
    assert data.n_treatment == 2
