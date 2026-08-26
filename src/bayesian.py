"""Conjugate Bayesian A/B testing engine.

Closed-form posteriors (Beta-Binomial, Student-t, LogNormal, Gamma-Poisson,
hurdle LogNormal) plus Monte Carlo summaries: P(improvement), expected lift,
HDI, expected loss, and ROPE probability. No MCMC required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from scipy import stats as spstats


class MetricType(str, Enum):
    BINARY = "binary"
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    POISSON = "poisson"
    HURDLE_LOGNORMAL = "hurdle_lognormal"


@dataclass
class AnalysisConfig:
    n_samples: int = 20_000
    cred_mass: float = 0.90
    mde: float = 0.02  # minimum detectable / useful relative lift
    rope: float = 0.01  # region of practical equivalence (± relative)
    seed: int | None = 42
    # Beta prior for conversion / hurdle conversion
    beta_alpha: float = 1.0
    beta_beta: float = 1.0
    # Gamma prior (shape, rate) for Poisson rates
    gamma_shape: float = 1.0
    gamma_rate: float = 1.0


@dataclass
class PosteriorResult:
    metric_name: str
    metric_type: MetricType
    n_control: int
    n_treatment: int
    observed_control: float
    observed_treatment: float
    observed_lift: float
    control_posterior_mean: float
    treatment_posterior_mean: float
    prob_improvement: float
    prob_degradation: float
    expected_relative_lift: float
    expected_absolute_lift: float
    relative_lift_hdi: tuple[float, float]
    absolute_diff_hdi: tuple[float, float]
    expected_loss_ship: float
    expected_loss_hold: float
    rope_probability: float
    prob_beat_mde: float
    prob_harm_beyond_mde: float
    frequentist_pvalue: float | None
    control_samples: np.ndarray = field(repr=False)
    treatment_samples: np.ndarray = field(repr=False)
    relative_lift_samples: np.ndarray = field(repr=False)
    absolute_diff_samples: np.ndarray = field(repr=False)
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_samples: bool = False) -> dict[str, Any]:
        payload = {
            "metric_name": self.metric_name,
            "metric_type": self.metric_type.value,
            "n_control": self.n_control,
            "n_treatment": self.n_treatment,
            "observed_control": self.observed_control,
            "observed_treatment": self.observed_treatment,
            "observed_lift": self.observed_lift,
            "control_posterior_mean": self.control_posterior_mean,
            "treatment_posterior_mean": self.treatment_posterior_mean,
            "prob_improvement": self.prob_improvement,
            "prob_degradation": self.prob_degradation,
            "expected_relative_lift": self.expected_relative_lift,
            "expected_absolute_lift": self.expected_absolute_lift,
            "relative_lift_hdi_low": self.relative_lift_hdi[0],
            "relative_lift_hdi_high": self.relative_lift_hdi[1],
            "absolute_diff_hdi_low": self.absolute_diff_hdi[0],
            "absolute_diff_hdi_high": self.absolute_diff_hdi[1],
            "expected_loss_ship": self.expected_loss_ship,
            "expected_loss_hold": self.expected_loss_hold,
            "rope_probability": self.rope_probability,
            "prob_beat_mde": self.prob_beat_mde,
            "prob_harm_beyond_mde": self.prob_harm_beyond_mde,
            "frequentist_pvalue": self.frequentist_pvalue,
            "extras": self.extras,
        }
        if include_samples:
            payload["control_samples"] = self.control_samples
            payload["treatment_samples"] = self.treatment_samples
            payload["relative_lift_samples"] = self.relative_lift_samples
            payload["absolute_diff_samples"] = self.absolute_diff_samples
        return payload


def highest_density_interval(samples: np.ndarray, cred_mass: float = 0.9) -> tuple[float, float]:
    """Smallest interval containing `cred_mass` of the posterior samples."""
    x = np.sort(np.asarray(samples, dtype=float).ravel())
    n = x.size
    if n == 0:
        return (float("nan"), float("nan"))
    cred_mass = float(np.clip(cred_mass, 0.0, 1.0))
    if cred_mass >= 1.0 or n == 1:
        return (float(x[0]), float(x[-1]))
    interval_len = int(np.floor(cred_mass * n))
    interval_len = max(1, min(interval_len, n - 1))
    n_intervals = n - interval_len
    widths = x[interval_len:][:n_intervals] - x[:n_intervals]
    i = int(np.argmin(widths))
    return (float(x[i]), float(x[i + interval_len]))


def _relative_lift(treatment: np.ndarray, control: np.ndarray) -> np.ndarray:
    eps = 1e-12
    denom = np.where(np.abs(control) < eps, np.sign(control) * eps + eps, control)
    return (treatment - control) / denom


def _sample_beta(successes: float, n: int, cfg: AnalysisConfig, rng: np.random.Generator) -> np.ndarray:
    a = cfg.beta_alpha + successes
    b = cfg.beta_beta + max(n - successes, 0)
    a = max(a, 1e-6)
    b = max(b, 1e-6)
    return rng.beta(a, b, size=cfg.n_samples)


def _sample_normal_mean(values: np.ndarray, cfg: AnalysisConfig, rng: np.random.Generator) -> np.ndarray:
    n = values.size
    mean = float(values.mean()) if n else 0.0
    if n < 2:
        return np.full(cfg.n_samples, mean)
    s = float(values.std(ddof=1))
    if s <= 0:
        return np.full(cfg.n_samples, mean)
    z = rng.standard_t(n - 1, size=cfg.n_samples)
    return mean + z * (s / np.sqrt(n))


def _sample_lognormal_mean(values: np.ndarray, cfg: AnalysisConfig, rng: np.random.Generator) -> np.ndarray:
    positive = values[values > 0]
    n = positive.size
    if n == 0:
        return np.zeros(cfg.n_samples)
    y = np.log(positive)
    ybar = float(y.mean())
    if n < 2:
        return np.full(cfg.n_samples, float(np.exp(ybar)))
    s2 = float(y.var(ddof=1))
    if s2 <= 0:
        return np.full(cfg.n_samples, float(np.exp(ybar)))
    chi2 = rng.chisquare(n - 1, size=cfg.n_samples)
    sigma2 = (n - 1) * s2 / np.maximum(chi2, 1e-12)
    mu = rng.normal(ybar, np.sqrt(sigma2 / n))
    return np.exp(mu + 0.5 * sigma2)


def _sample_poisson_rate(values: np.ndarray, cfg: AnalysisConfig, rng: np.random.Generator) -> np.ndarray:
    shape = cfg.gamma_shape + float(values.sum())
    rate = cfg.gamma_rate + values.size
    return rng.gamma(shape, 1.0 / rate, size=cfg.n_samples)


def _sample_hurdle_mean(values: np.ndarray, cfg: AnalysisConfig, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    n = values.size
    successes = float(np.sum(values > 0))
    p = _sample_beta(successes, n, cfg, rng)
    positive = values[values > 0]
    cond = _sample_lognormal_mean(positive, cfg, rng) if positive.size else np.zeros(cfg.n_samples)
    extras = {
        "conversion_posterior_mean": float(p.mean()),
        "conditional_revenue_mean": float(cond.mean()),
        "n_converters": int(successes),
    }
    return p * cond, extras


def sample_posterior(
    values: np.ndarray,
    metric_type: MetricType,
    cfg: AnalysisConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    extras: dict[str, Any] = {}
    if metric_type is MetricType.BINARY:
        samples = _sample_beta(float(np.sum(values)), values.size, cfg, rng)
    elif metric_type is MetricType.NORMAL:
        samples = _sample_normal_mean(values, cfg, rng)
    elif metric_type is MetricType.LOGNORMAL:
        samples = _sample_lognormal_mean(values, cfg, rng)
    elif metric_type is MetricType.POISSON:
        samples = _sample_poisson_rate(values, cfg, rng)
    elif metric_type is MetricType.HURDLE_LOGNORMAL:
        samples, extras = _sample_hurdle_mean(values, cfg, rng)
    else:
        raise ValueError(f"Unsupported metric type: {metric_type}")
    return samples, extras


def _frequentist_pvalue(control: np.ndarray, treatment: np.ndarray, metric_type: MetricType) -> float | None:
    if control.size < 2 or treatment.size < 2:
        return None
    try:
        if metric_type is MetricType.BINARY:
            s1, n1 = float(control.sum()), control.size
            s2, n2 = float(treatment.sum()), treatment.size
            p = (s1 + s2) / (n1 + n2)
            if p <= 0 or p >= 1:
                return 1.0
            se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
            z = (s2 / n2 - s1 / n1) / se
            return float(2 * spstats.norm.sf(abs(z)))
        if metric_type is MetricType.POISSON:
            # Rate comparison via two-sample t on raw counts as a rough analogue
            return float(spstats.ttest_ind(treatment, control, equal_var=False).pvalue)
        return float(spstats.ttest_ind(treatment, control, equal_var=False).pvalue)
    except Exception:
        return None


def analyze_metric(
    control: np.ndarray,
    treatment: np.ndarray,
    metric_name: str,
    metric_type: MetricType | str,
    config: AnalysisConfig | None = None,
) -> PosteriorResult:
    """Fit posteriors and return decision-ready summaries for one metric."""
    cfg = config or AnalysisConfig()
    metric_type = MetricType(metric_type)
    control = np.asarray(control, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    control = control[np.isfinite(control)]
    treatment = treatment[np.isfinite(treatment)]
    if control.size == 0 or treatment.size == 0:
        raise ValueError(f"Metric '{metric_name}' has an empty control or treatment group.")

    rng = np.random.default_rng(cfg.seed)
    c_samples, c_extra = sample_posterior(control, metric_type, cfg, rng)
    t_samples, t_extra = sample_posterior(treatment, metric_type, cfg, rng)

    abs_diff = t_samples - c_samples
    rel_lift = _relative_lift(t_samples, c_samples)

    obs_c = float(control.mean())
    obs_t = float(treatment.mean())
    obs_lift = (obs_t - obs_c) / obs_c if abs(obs_c) > 1e-12 else float("nan")

    extras = {"control": c_extra, "treatment": t_extra}
    return PosteriorResult(
        metric_name=metric_name,
        metric_type=metric_type,
        n_control=int(control.size),
        n_treatment=int(treatment.size),
        observed_control=obs_c,
        observed_treatment=obs_t,
        observed_lift=obs_lift,
        control_posterior_mean=float(c_samples.mean()),
        treatment_posterior_mean=float(t_samples.mean()),
        prob_improvement=float(np.mean(t_samples > c_samples)),
        prob_degradation=float(np.mean(t_samples < c_samples)),
        expected_relative_lift=float(np.mean(rel_lift)),
        expected_absolute_lift=float(np.mean(abs_diff)),
        relative_lift_hdi=highest_density_interval(rel_lift, cfg.cred_mass),
        absolute_diff_hdi=highest_density_interval(abs_diff, cfg.cred_mass),
        expected_loss_ship=float(np.mean(np.maximum(c_samples - t_samples, 0))),
        expected_loss_hold=float(np.mean(np.maximum(t_samples - c_samples, 0))),
        rope_probability=float(np.mean(np.abs(rel_lift) <= cfg.rope)),
        prob_beat_mde=float(np.mean(rel_lift > cfg.mde)),
        prob_harm_beyond_mde=float(np.mean(rel_lift < -cfg.mde)),
        frequentist_pvalue=_frequentist_pvalue(control, treatment, metric_type),
        control_samples=c_samples,
        treatment_samples=t_samples,
        relative_lift_samples=rel_lift,
        absolute_diff_samples=abs_diff,
        extras=extras,
    )


def analyze_many(
    groups: dict[str, tuple[np.ndarray, np.ndarray, MetricType | str]],
    config: AnalysisConfig | None = None,
) -> list[PosteriorResult]:
    """Analyze several metrics. `groups` maps name -> (control, treatment, type)."""
    return [
        analyze_metric(control, treatment, name, metric_type, config)
        for name, (control, treatment, metric_type) in groups.items()
    ]
