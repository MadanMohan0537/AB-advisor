"""Rule-based insights plus an optional LLM rewrite that cannot invent numbers."""

from __future__ import annotations

import json
import os
from typing import Any

from src.bayesian import PosteriorResult
from src.decisions import ExperimentDecision


def _pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"


def _signed_pct(x: float, digits: int = 1) -> str:
    sign = "+" if x >= 0 else ""
    return f"{sign}{100 * x:.{digits}f}%"


def metric_template(result: PosteriorResult, cred_mass: float = 0.90) -> str:
    lift = _signed_pct(result.expected_relative_lift)
    hdi = result.relative_lift_hdi
    direction = "increases" if result.expected_relative_lift >= 0 else "decreases"
    strength = (
        "Strong evidence"
        if result.prob_improvement >= 0.95
        else "Moderate evidence"
        if result.prob_improvement >= 0.85
        else "Weak / mixed evidence"
        if result.prob_improvement >= 0.15
        else "Strong evidence against an improvement"
    )
    return (
        f"{strength} that treatment {direction} **{result.metric_name}**. "
        f"P(treatment > control) = {_pct(result.prob_improvement)}, "
        f"expected lift = {lift}, "
        f"{int(cred_mass * 100)}% HDI = [{_signed_pct(hdi[0])}, {_signed_pct(hdi[1])}]. "
        f"P(lift > MDE) = {_pct(result.prob_beat_mde)}; "
        f"P(harm beyond MDE) = {_pct(result.prob_harm_beyond_mde)}. "
        f"Expected loss if you ship = {result.expected_loss_ship:.4g} "
        f"(hold = {result.expected_loss_hold:.4g}) on the raw metric scale. "
        f"Observed means: control {result.observed_control:.4g} (n={result.n_control:,}) vs "
        f"treatment {result.observed_treatment:.4g} (n={result.n_treatment:,})."
    )


def experiment_template(
    results: list[PosteriorResult],
    decision: ExperimentDecision,
    cred_mass: float = 0.90,
) -> str:
    lines = [
        f"## Recommendation: {decision.headline}",
        "",
        decision.rationale,
        "",
    ]
    if decision.warnings:
        lines.append("### Warnings")
        for warning in decision.warnings:
            lines.append(f"- {warning}")
        lines.append("")

    lines.append("### Metric-by-metric read")
    for result in results:
        lines.append(f"- {metric_template(result, cred_mass)}")
    lines.append("")
    lines.append("### Suggested next steps")
    if decision.action == "ship":
        lines.append(
            "- Roll out to 100% and keep a holdout if the change is expensive to reverse."
        )
        lines.append("- Watch guardrails for 24–72 hours after ramp for novelty or latency effects.")
        lines.append("- Document the prior, MDE, and decision threshold you actually used.")
    elif decision.action == "hold":
        lines.append("- Do not ramp. Dig into segments (device, geo, new vs returning) for harm.")
        lines.append("- If a guardrail moved, confirm instrumentation before iterating on the UX.")
        lines.append("- Consider a follow-up experiment that protects the failing metric.")
    elif decision.action == "investigate":
        lines.append("- Pause decision-making until SRM is explained (redirects, bots, filters).")
        lines.append("- Re-run analysis only on a clean assignment window.")
    else:
        lines.append("- Keep the pre-registered stopping rule; do not raise the bar after seeing the data.")
        lines.append("- Recheck sample size vs MDE — you may be underpowered for a small effect.")
        lines.append("- Bayesian peeking is valid if you stop on expected loss / P(better), not on p-values.")
    lines.append("")
    lines.append(
        "_Caveats: users are assumed independent (SUTVA), assignment is random, "
        "and the model family matches the metric type. These are not p-values._"
    )
    return "\n".join(lines)


def _stats_payload(
    results: list[PosteriorResult],
    decision: ExperimentDecision,
    cred_mass: float,
) -> dict[str, Any]:
    return {
        "recommendation": {
            "action": decision.action,
            "headline": decision.headline,
            "rationale": decision.rationale,
            "warnings": decision.warnings,
        },
        "credible_interval_mass": cred_mass,
        "metrics": [
            {
                "name": r.metric_name,
                "type": r.metric_type.value,
                "n_control": r.n_control,
                "n_treatment": r.n_treatment,
                "observed_control": round(r.observed_control, 6),
                "observed_treatment": round(r.observed_treatment, 6),
                "observed_relative_lift": round(r.observed_lift, 6) if r.observed_lift == r.observed_lift else None,
                "prob_treatment_better": round(r.prob_improvement, 4),
                "expected_relative_lift": round(r.expected_relative_lift, 6),
                "expected_absolute_lift": round(r.expected_absolute_lift, 6),
                "relative_lift_hdi": [round(r.relative_lift_hdi[0], 6), round(r.relative_lift_hdi[1], 6)],
                "prob_beat_mde": round(r.prob_beat_mde, 4),
                "prob_harm_beyond_mde": round(r.prob_harm_beyond_mde, 4),
                "rope_probability": round(r.rope_probability, 4),
                "expected_loss_if_ship": round(r.expected_loss_ship, 8),
                "expected_loss_if_hold": round(r.expected_loss_hold, 8),
                "frequentist_pvalue": None if r.frequentist_pvalue is None else round(r.frequentist_pvalue, 6),
            }
            for r in results
        ],
        "per_metric_actions": [
            {"metric": d.metric_name, "role": d.role, "action": d.action, "rationale": d.rationale}
            for d in decision.metric_decisions
        ],
    }


SYSTEM_PROMPT = """You are an experiment-review assistant for product managers.
You receive JSON with Bayesian A/B test results. Write a concise executive brief.

Hard rules:
- Use ONLY the numbers in the JSON. Never invent, round into a different conclusion, or add statistics.
- Speak in probability language ("there is an 93% chance treatment beats control"), not p-values, unless you quote the provided frequentist p-value as a footnote.
- Lead with the decision (ship / hold / keep running / investigate).
- Call out guardrail risk and SRM warnings if present.
- Mention the credible interval and expected loss in plain English.
- Include 3 concrete next steps.
- Add a short "what this is not" caveat (not a p-value; assumes independent users).
- Do not claim causality beyond the randomized assignment implied by the experiment.
"""


def llm_enhance(
    results: list[PosteriorResult],
    decision: ExperimentDecision,
    cred_mass: float = 0.90,
    model: str | None = None,
    api_key: str | None = None,
) -> str | None:
    """Optional OpenAI rewrite. Returns None if no key or the call fails."""
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        return None
    try:
        from openai import OpenAI
    except Exception:
        return None

    model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    payload = _stats_payload(results, decision, cred_mass)
    try:
        client = OpenAI(api_key=key)
        response = client.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Write the experiment review from this JSON. "
                        "If any number is missing, say so rather than guessing.\n\n"
                        + json.dumps(payload, indent=2)
                    ),
                },
            ],
        )
        text = response.choices[0].message.content
        return text.strip() if text else None
    except Exception:
        return None


def generate_insights(
    results: list[PosteriorResult],
    decision: ExperimentDecision,
    cred_mass: float = 0.90,
    use_llm: bool = True,
    model: str | None = None,
    api_key: str | None = None,
) -> tuple[str, str]:
    """Return (display_markdown, source) where source is 'llm' or 'template'."""
    template = experiment_template(results, decision, cred_mass)
    if use_llm:
        rewritten = llm_enhance(results, decision, cred_mass, model=model, api_key=api_key)
        if rewritten:
            return rewritten, "llm"
    return template, "template"
