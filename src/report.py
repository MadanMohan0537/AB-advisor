"""Markdown / HTML experiment-review documents from analysis output."""

from __future__ import annotations

from datetime import datetime, timezone

from src.bayesian import AnalysisConfig, PosteriorResult
from src.decisions import ExperimentDecision
from src.metrics import ExperimentData
from src.summarize import experiment_template


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _signed(x: float) -> str:
    sign = "+" if x >= 0 else ""
    return f"{sign}{100 * x:.1f}%"


def markdown_report(
    experiment_name: str,
    data: ExperimentData,
    results: list[PosteriorResult],
    decision: ExperimentDecision,
    config: AnalysisConfig,
    insights: str,
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    rows = []
    for r in results:
        rows.append(
            f"| {r.metric_name} | {r.metric_type.value} | {r.n_control:,} | {r.n_treatment:,} | "
            f"{r.observed_control:.4g} | {r.observed_treatment:.4g} | {_signed(r.observed_lift)} | "
            f"{_pct(r.prob_improvement)} | {_signed(r.expected_relative_lift)} | "
            f"[{_signed(r.relative_lift_hdi[0])}, {_signed(r.relative_lift_hdi[1])}] | "
            f"{_pct(r.prob_beat_mde)} | {_pct(r.prob_harm_beyond_mde)} |"
        )

    table = "\n".join(
        [
            "| Metric | Type | n_c | n_t | Control | Treatment | Observed lift | P(better) | E[lift] | "
            f"{int(config.cred_mass * 100)}% HDI | P(> MDE) | P(harm) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|",
            *rows,
        ]
    )

    warnings = "\n".join(f"- {w}" for w in decision.warnings) or "- None"
    metric_actions = "\n".join(
        f"- **{d.metric_name}** ({d.role}): `{d.action}` — {d.rationale}"
        for d in decision.metric_decisions
    )

    return f"""# Experiment Review: {experiment_name}

Generated {now} by AB Advisor (Bayesian A/B test analyzer).

## Decision

**{decision.headline}**

{decision.rationale}

### Warnings
{warnings}

## Design snapshot

- Control label: `{data.control_label}` (n={data.n_control:,})
- Treatment label: `{data.treatment_label}` (n={data.n_treatment:,})
- Sample ratio mismatch p-value: {data.srm_pvalue:.4g} ({'FLAG' if data.srm_flag else 'ok'})
- Credible interval mass: {int(config.cred_mass * 100)}% HDI
- Minimum useful lift (MDE): {config.mde:.2%}
- ROPE: ±{config.rope:.2%}
- Conversion prior: Beta({config.beta_alpha}, {config.beta_beta})
- Poisson prior: Gamma(shape={config.gamma_shape}, rate={config.gamma_rate})
- Posterior draws: {config.n_samples:,}

## Results

{table}

## Per-metric actions

{metric_actions}

## Narrative

{insights}

## How to read this for executives

- **P(better)** is the probability the treatment mean exceeds the control mean, given the data and prior. It is not a p-value.
- **Expected lift** is the average of the posterior on (treatment − control) / control.
- **HDI** is the highest-density credible interval: the shortest range containing {int(config.cred_mass * 100)}% of posterior lift mass.
- **Expected loss** is the average amount you give up on the raw metric if you pick the worse variant. Ship when that cost is below a business threshold.
- **ROPE** (region of practical equivalence) is the lift band you consider "too small to care." High ROPE probability supports stopping for futility.

## Limitations

- Users are modeled as independent (SUTVA). Network, marketplace, or interference effects are not in the likelihood.
- Peeking is valid for these Bayesian expected-loss / probability rules, but not if you also hunt for a frequentist p-value.
- Priors are weakly informative by default. If you have a strong historical conversion rate, encode it as Beta(α, β) with a modest effective sample size (50–200).
- Multiple metrics inflate the chance that *something* looks good. Pre-register a primary metric and treat the rest as guardrails or diagnostics.
- The hurdle-lognormal model for revenue assumes zeros are non-converters and positive spend is log-normal.

---
*This document is a decision aid, not a substitute for instrumentation review or qualitative product judgment.*
"""


def html_report(markdown_text: str, experiment_name: str) -> str:
    try:
        import markdown as md

        body = md.markdown(markdown_text, extensions=["tables", "fenced_code"])
    except Exception:
        body = "<pre>" + (
            markdown_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        ) + "</pre>"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Experiment Review — {experiment_name}</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, sans-serif; max-width: 920px; margin: 40px auto; color: #0f172a; line-height: 1.55; padding: 0 20px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #e2e8f0; padding: 6px 8px; text-align: left; }}
    th {{ background: #f1f5f9; }}
    h1, h2, h3 {{ color: #0f172a; }}
    code {{ background: #f1f5f9; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
{body}
</html>
"""


def sample_size_binary(
    p_control: float,
    relative_mde: float,
    target_prob: float = 0.95,
    power: float = 0.8,
    n_grid: tuple[int, ...] | None = None,
    n_sims: int = 400,
    seed: int = 7,
) -> dict:
    """Find n per arm such that P(P(better) > target | true lift = MDE) ≈ power."""
    import numpy as np
    from src.bayesian import AnalysisConfig, MetricType, analyze_metric

    p_t = min(0.999, p_control * (1 + relative_mde))
    grid = n_grid or (200, 500, 1000, 2000, 4000, 8000, 15000, 25000)
    rng = np.random.default_rng(seed)
    cfg = AnalysisConfig(n_samples=4000, seed=None)

    best_n = None
    curve = []
    for n in grid:
        wins = 0
        for _ in range(n_sims):
            c = rng.binomial(1, p_control, size=n).astype(float)
            t = rng.binomial(1, p_t, size=n).astype(float)
            result = analyze_metric(c, t, "conversion", MetricType.BINARY, cfg)
            if result.prob_improvement >= target_prob:
                wins += 1
        achieved = wins / n_sims
        curve.append({"n_per_arm": n, "power": achieved})
        if best_n is None and achieved >= power:
            best_n = n
    return {
        "p_control": p_control,
        "p_treatment": p_t,
        "relative_mde": relative_mde,
        "target_prob": target_prob,
        "desired_power": power,
        "recommended_n_per_arm": best_n,
        "curve": curve,
    }
