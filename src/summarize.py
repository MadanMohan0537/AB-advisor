"""Rule-based insights plus an optional LLM rewrite that cannot invent numbers."""

from __future__ import annotations

from dataclasses import dataclass
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

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_DEFAULT_MODEL = "deepseek-v4-flash"
OPENAI_DEFAULT_MODEL = "gpt-4o-mini"


@dataclass
class LLMSettings:
    provider: str
    api_key: str
    model: str
    base_url: str | None = None
    extra_body: dict[str, Any] | None = None


def resolve_llm_settings(
    api_key: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> LLMSettings | None:
    """Prefer DeepSeek. Fall back to OpenAI only if that is the key that is set."""
    provider = (provider or os.environ.get("LLM_PROVIDER") or "").strip().lower()
    env_deepseek = os.environ.get("DEEPSEEK_API_KEY") or ""
    env_openai = os.environ.get("OPENAI_API_KEY") or ""
    key = (api_key or env_deepseek or env_openai or "").strip()
    if not key:
        return None
    if not provider:
        if api_key:
            provider = "deepseek"
        elif env_deepseek:
            provider = "deepseek"
        else:
            provider = "openai"

    if provider in {"deepseek", "ds"}:
        chosen_model = model or os.environ.get("DEEPSEEK_MODEL") or os.environ.get("LLM_MODEL") or DEEPSEEK_DEFAULT_MODEL
        extra = {"thinking": {"type": "disabled"}} if "v4" in chosen_model else None
        return LLMSettings(
            provider="deepseek",
            api_key=key,
            model=chosen_model,
            base_url=base_url or os.environ.get("DEEPSEEK_BASE_URL") or DEEPSEEK_BASE_URL,
            extra_body=extra,
        )
    chosen_model = model or os.environ.get("OPENAI_MODEL") or os.environ.get("LLM_MODEL") or OPENAI_DEFAULT_MODEL
    return LLMSettings(
        provider="openai",
        api_key=key,
        model=chosen_model,
        base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
        extra_body=None,
    )


def _message_text(message: Any) -> str | None:
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("text"):
                parts.append(str(block["text"]))
            else:
                text = getattr(block, "text", None)
                if text:
                    parts.append(str(text))
        joined = "".join(parts).strip()
        return joined or None
    return None


def llm_enhance(
    results: list[PosteriorResult],
    decision: ExperimentDecision,
    cred_mass: float = 0.90,
    model: str | None = None,
    api_key: str | None = None,
    provider: str | None = None,
    base_url: str | None = None,
) -> tuple[str | None, str | None, LLMSettings | None]:
    """Optional DeepSeek (or OpenAI-compatible) rewrite. Returns (text, error, settings)."""
    settings = resolve_llm_settings(api_key=api_key, provider=provider, model=model, base_url=base_url)
    if settings is None:
        return None, None, None
    try:
        from openai import OpenAI
    except Exception:
        return None, "The openai package is not installed.", settings

    payload = _stats_payload(results, decision, cred_mass)
    kwargs: dict[str, Any] = {
        "model": settings.model,
        "temperature": 0.2,
        "messages": [
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
    }
    if settings.extra_body:
        kwargs["extra_body"] = settings.extra_body
    try:
        client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)
        response = client.chat.completions.create(**kwargs)
        text = _message_text(response.choices[0].message)
        if not text:
            return None, "The model returned an empty message.", settings
        return text, None, settings
    except Exception as exc:
        return None, str(exc), settings


def generate_insights(
    results: list[PosteriorResult],
    decision: ExperimentDecision,
    cred_mass: float = 0.90,
    use_llm: bool = True,
    model: str | None = None,
    api_key: str | None = None,
    provider: str | None = None,
    base_url: str | None = None,
) -> tuple[str, str, str | None]:
    """Return (display_markdown, source, error). source is llm / template / template_fallback."""
    template = experiment_template(results, decision, cred_mass)
    if not use_llm:
        return template, "template", None
    rewritten, error, settings = llm_enhance(
        results,
        decision,
        cred_mass,
        model=model,
        api_key=api_key,
        provider=provider,
        base_url=base_url,
    )
    if rewritten:
        label = f"llm:{settings.provider}" if settings else "llm"
        return rewritten, label, None
    if error:
        return template, "template_fallback", error
    return template, "template", None
