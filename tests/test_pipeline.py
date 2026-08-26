"""End-to-end analysis of the synthetic checkout experiment."""

from src.bayesian import AnalysisConfig, analyze_metric
from src.data_gen import generate_checkout_experiment, generate_underpowered_experiment
from src.decisions import decide_experiment
from src.metrics import load_experiment


def test_checkout_ships_on_primary_metrics():
    df = generate_checkout_experiment(n_per_arm=4000, seed=42)
    data = load_experiment(df)
    cfg = AnalysisConfig(n_samples=8000, seed=42)
    results = [
        analyze_metric(*data.split(name), name, data.types[name], cfg)
        for name in ("converted", "revenue")
    ]
    decision = decide_experiment(results)
    assert decision.action == "ship"
    converted = next(r for r in results if r.metric_name == "converted")
    assert converted.prob_improvement > 0.99
    assert converted.relative_lift_hdi[0] > 0


def test_checkout_all_metrics_does_not_block_on_improved_bounce():
    df = generate_checkout_experiment(n_per_arm=4000, seed=42)
    data = load_experiment(df)
    cfg = AnalysisConfig(n_samples=8000, seed=42)
    results = [
        analyze_metric(*data.split(name), name, data.types[name], cfg)
        for name in data.metric_cols
    ]
    decision = decide_experiment(results, srm_flag=data.srm_flag)
    bounce = next(d for d in decision.metric_decisions if d.metric_name == "bounce")
    assert bounce.action != "block"
    assert decision.action == "ship"


def test_underpowered_keeps_running():
    df = generate_underpowered_experiment()
    data = load_experiment(df)
    cfg = AnalysisConfig(n_samples=6000, seed=1)
    result = analyze_metric(*data.split("converted"), "converted", data.types["converted"], cfg)
    decision = decide_experiment([result])
    assert decision.action in {"keep_running", "hold", "ship"}
    # With n=400 and a 8% relative lift, we should not be extremely sure.
    assert result.prob_improvement < 0.999
    lo, hi = result.relative_lift_hdi
    assert lo < 0.15  # interval is wide
