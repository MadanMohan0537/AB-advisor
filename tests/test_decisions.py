"""Decision rules and template summaries stay faithful to the numbers."""

from __future__ import annotations

import numpy as np

from src.bayesian import AnalysisConfig, MetricType, analyze_metric
from src.decisions import classify_metric_role, decide_experiment
from src.summarize import experiment_template, generate_insights, metric_template


def _binary_result(p_c: float, p_t: float, n: int = 6000, seed: int = 0):
    rng = np.random.default_rng(seed)
    return analyze_metric(
        rng.binomial(1, p_c, n).astype(float),
        rng.binomial(1, p_t, n).astype(float),
        "converted",
        MetricType.BINARY,
        AnalysisConfig(seed=seed),
    )


def test_role_heuristics():
    assert classify_metric_role("converted") == "primary"
    assert classify_metric_role("revenue") == "primary"
    assert classify_metric_role("bounce") == "guardrail"
    assert classify_metric_role("time_on_page_sec") == "secondary"


def test_ship_when_primary_wins():
    result = _binary_result(0.10, 0.14)
    decision = decide_experiment([result], roles={"converted": "primary"})
    assert decision.action == "ship"
    assert "Ship" in decision.headline


def test_srm_blocks_ship():
    result = _binary_result(0.10, 0.14)
    decision = decide_experiment([result], roles={"converted": "primary"}, srm_flag=True, srm_pvalue=1e-8)
    assert decision.action == "investigate"
    assert decision.warnings


def test_template_contains_actual_probability():
    result = _binary_result(0.10, 0.13, seed=4)
    text = metric_template(result)
    assert result.metric_name in text
    assert f"{result.prob_improvement:.1%}".replace("%", "") in text.replace("%", "")
    full = experiment_template([result], decide_experiment([result]))
    assert "Recommendation" in full
    md, source, err = generate_insights([result], decide_experiment([result]), use_llm=False)
    assert source == "template"
    assert err is None
    assert "converted" in md
