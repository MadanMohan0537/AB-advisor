"""Ship / hold / keep-running rules for Bayesian experiment results."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.bayesian import PosteriorResult


@dataclass
class DecisionThresholds:
    ship_prob: float = 0.95
    ship_mde_prob: float = 0.80
    harm_prob: float = 0.05
    futility_rope_prob: float = 0.80
    loss_ratio_ship: float = 0.25  # ship if E[loss ship] < ratio * E[loss hold]


@dataclass
class MetricDecision:
    metric_name: str
    role: str  # primary | guardrail | secondary
    action: str
    rationale: str
    confidence: str


@dataclass
class ExperimentDecision:
    action: str
    headline: str
    rationale: str
    metric_decisions: list[MetricDecision] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


LOWER_IS_BETTER_TOKENS = (
    "bounce",
    "error",
    "latency",
    "refund",
    "churn",
    "complaint",
    "crash",
    "cancel",
    "uninstall",
)


def default_higher_is_better(name: str) -> bool:
    lowered = name.lower()
    return not any(tok in lowered for tok in LOWER_IS_BETTER_TOKENS)


def classify_metric_role(name: str, explicit: dict[str, str] | None = None) -> str:
    if explicit and name in explicit:
        return explicit[name]
    lowered = name.lower()
    guardrail_tokens = (
        "error",
        "latency",
        "crash",
        "refund",
        "churn",
        "complaint",
        "support",
        "bounce",
        "uninstall",
        "cancel",
    )
    primary_tokens = ("conversion", "converted", "revenue", "purchase", "checkout", "signup")
    if any(tok in lowered for tok in guardrail_tokens):
        return "guardrail"
    if any(tok in lowered for tok in primary_tokens):
        return "primary"
    return "secondary"


def decide_metric(
    result: PosteriorResult,
    role: str,
    thresholds: DecisionThresholds | None = None,
    higher_is_better: bool = True,
) -> MetricDecision:
    t = thresholds or DecisionThresholds()
    p_win = result.prob_improvement if higher_is_better else result.prob_degradation
    p_harm = result.prob_harm_beyond_mde if higher_is_better else result.prob_beat_mde
    p_mde = result.prob_beat_mde if higher_is_better else result.prob_harm_beyond_mde
    loss_ship = result.expected_loss_ship if higher_is_better else result.expected_loss_hold
    loss_hold = result.expected_loss_hold if higher_is_better else result.expected_loss_ship

    if role == "guardrail" and p_harm >= t.harm_prob and p_win < 0.5:
        return MetricDecision(
            metric_name=result.metric_name,
            role=role,
            action="block",
            rationale=(
                f"Guardrail risk: {p_harm:.1%} chance of a harmful move beyond the MDE "
                f"({t.harm_prob:.0%} threshold)."
            ),
            confidence="high" if p_harm >= 0.2 else "medium",
        )

    if p_win >= t.ship_prob and p_mde >= t.ship_mde_prob:
        return MetricDecision(
            metric_name=result.metric_name,
            role=role,
            action="ship",
            rationale=(
                f"{p_win:.1%} probability of improvement and {p_mde:.1%} chance of beating "
                f"the {result.expected_relative_lift:.1%} expected lift's MDE."
            ),
            confidence="high",
        )

    if (
        p_win >= t.ship_prob
        and loss_hold > 0
        and loss_ship <= t.loss_ratio_ship * loss_hold
    ):
        return MetricDecision(
            metric_name=result.metric_name,
            role=role,
            action="ship",
            rationale=(
                f"{p_win:.1%} probability of improvement with low expected loss if you ship "
                f"({loss_ship:.4g} vs {loss_hold:.4g} if you hold)."
            ),
            confidence="high",
        )

    if result.rope_probability >= t.futility_rope_prob and p_mde < 0.2:
        return MetricDecision(
            metric_name=result.metric_name,
            role=role,
            action="stop_futility",
            rationale=(
                f"{result.rope_probability:.1%} of the posterior lift sits inside the practical "
                "equivalence window. Further traffic is unlikely to reveal a useful effect."
            ),
            confidence="medium",
        )

    if p_harm >= max(t.harm_prob, 0.5):
        return MetricDecision(
            metric_name=result.metric_name,
            role=role,
            action="hold",
            rationale=f"{p_harm:.1%} probability of a harmful effect beyond the MDE.",
            confidence="high" if p_harm >= 0.8 else "medium",
        )

    return MetricDecision(
        metric_name=result.metric_name,
        role=role,
        action="keep_running",
        rationale=(
            f"Evidence is mixed: P(better)={p_win:.1%}, P(lift > MDE)={p_mde:.1%}, "
            f"{result.rope_probability:.1%} of mass in the ROPE. Collect more data."
        ),
        confidence="low",
    )


def decide_experiment(
    results: list[PosteriorResult],
    roles: dict[str, str] | None = None,
    thresholds: DecisionThresholds | None = None,
    srm_flag: bool = False,
    srm_pvalue: float | None = None,
    higher_is_better: dict[str, bool] | None = None,
) -> ExperimentDecision:
    thresholds = thresholds or DecisionThresholds()
    higher_is_better = higher_is_better or {}
    warnings: list[str] = []
    if srm_flag:
        ptxt = f" (p={srm_pvalue:.4g})" if srm_pvalue is not None else ""
        warnings.append(
            "Sample ratio mismatch detected"
            + ptxt
            + ". Assignment or logging may be broken — do not ship from these numbers until SRM is explained."
        )

    metric_decisions = []
    for result in results:
        role = classify_metric_role(result.metric_name, roles)
        hib = higher_is_better.get(result.metric_name, default_higher_is_better(result.metric_name))
        metric_decisions.append(decide_metric(result, role, thresholds, hib))

    blockers = [
        d
        for d in metric_decisions
        if d.action == "block" or (d.role == "guardrail" and d.action == "hold")
    ]
    primary = [d for d in metric_decisions if d.role == "primary"]
    primary_ship = [d for d in primary if d.action == "ship"]
    primary_hold = [d for d in primary if d.action in {"hold", "stop_futility"}]
    primary_keep = [d for d in primary if d.action == "keep_running"]

    if srm_flag:
        action = "investigate"
        headline = "Do not ship — sample ratio mismatch"
        rationale = (
            "Traffic is not splitting as designed. Fix assignment, filtering, or logging "
            "before treating any posterior as a product decision."
        )
    elif blockers:
        action = "hold"
        names = ", ".join(d.metric_name for d in blockers)
        headline = f"Hold — guardrail risk on {names}"
        rationale = " ".join(d.rationale for d in blockers)
    elif primary and len(primary_ship) == len(primary):
        action = "ship"
        headline = "Ship the treatment"
        rationale = (
            "Primary metrics show a high probability of a useful improvement, and guardrails "
            "are not signaling material harm. Secondary metrics are reported for learning "
            "and do not veto this call."
        )
    elif primary_hold and not primary_ship and not primary_keep:
        action = "hold"
        headline = "Do not ship — primary metrics do not clear the bar"
        rationale = " ".join(d.rationale for d in primary_hold)
    else:
        action = "keep_running"
        headline = "Keep the experiment running"
        rationale = (
            "At least one primary metric is still in the uncertain region. "
            "Bayesian updating stays valid if you peek, but wait until expected loss "
            "or P(better) crosses your pre-registered threshold."
        )

    return ExperimentDecision(
        action=action,
        headline=headline,
        rationale=rationale,
        metric_decisions=metric_decisions,
        warnings=warnings,
    )
