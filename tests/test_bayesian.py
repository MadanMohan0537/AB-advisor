"""Tests for conjugate Bayesian posteriors and HDI."""

from __future__ import annotations

import numpy as np
import pytest

from src.bayesian import (
    AnalysisConfig,
    MetricType,
    analyze_metric,
    highest_density_interval,
)


def test_hdi_covers_standard_normal_mass():
    rng = np.random.default_rng(0)
    samples = rng.normal(0, 1, size=50_000)
    lo, hi = highest_density_interval(samples, 0.9)
    assert lo < 0 < hi
    assert 3.0 < hi - lo < 3.6  # N(0,1) 90% HDI width ≈ 3.29


def test_binary_clear_winner():
    rng = np.random.default_rng(1)
    control = rng.binomial(1, 0.10, size=8000).astype(float)
    treatment = rng.binomial(1, 0.13, size=8000).astype(float)
    result = analyze_metric(control, treatment, "converted", MetricType.BINARY)
    assert result.prob_improvement > 0.99
    assert result.expected_relative_lift > 0.1
    assert result.relative_lift_hdi[0] > 0
    assert result.frequentist_pvalue is not None
    assert result.frequentist_pvalue < 0.01


def test_binary_no_effect_near_coin_flip():
    rng = np.random.default_rng(2)
    control = rng.binomial(1, 0.2, size=3000).astype(float)
    treatment = rng.binomial(1, 0.2, size=3000).astype(float)
    result = analyze_metric(control, treatment, "converted", MetricType.BINARY)
    assert 0.2 < result.prob_improvement < 0.8
    lo, hi = result.relative_lift_hdi
    assert lo < 0 < hi


def test_poisson_detects_rate_increase():
    rng = np.random.default_rng(3)
    control = rng.poisson(2.0, size=4000).astype(float)
    treatment = rng.poisson(2.4, size=4000).astype(float)
    result = analyze_metric(control, treatment, "events", MetricType.POISSON)
    assert result.prob_improvement > 0.99
    assert result.expected_absolute_lift > 0.2


def test_normal_mean_shift():
    rng = np.random.default_rng(4)
    control = rng.normal(50, 10, size=2000)
    treatment = rng.normal(55, 10, size=2000)
    result = analyze_metric(control, treatment, "revenue", MetricType.NORMAL)
    assert result.prob_improvement > 0.99
    assert 3 < result.expected_absolute_lift < 7


def test_hurdle_lognormal_revenue():
    rng = np.random.default_rng(5)
    n = 4000
    c_buy = rng.binomial(1, 0.1, size=n)
    t_buy = rng.binomial(1, 0.14, size=n)
    control = np.zeros(n)
    treatment = np.zeros(n)
    control[c_buy == 1] = rng.lognormal(3.9, 0.4, size=int(c_buy.sum()))
    treatment[t_buy == 1] = rng.lognormal(4.0, 0.4, size=int(t_buy.sum()))
    result = analyze_metric(control, treatment, "revenue", MetricType.HURDLE_LOGNORMAL)
    assert result.prob_improvement > 0.95
    assert result.extras["treatment"]["n_converters"] > result.extras["control"]["n_converters"]


def test_empty_group_raises():
    with pytest.raises(ValueError):
        analyze_metric(np.array([]), np.array([1.0]), "x", MetricType.NORMAL)


def test_reproducible_with_seed():
    rng = np.random.default_rng(9)
    c = rng.normal(0, 1, 200)
    t = rng.normal(0.2, 1, 200)
    cfg = AnalysisConfig(seed=11, n_samples=3000)
    a = analyze_metric(c, t, "x", MetricType.NORMAL, cfg)
    b = analyze_metric(c, t, "x", MetricType.NORMAL, cfg)
    np.testing.assert_allclose(a.control_samples, b.control_samples)
    np.testing.assert_allclose(a.prob_improvement, b.prob_improvement)
