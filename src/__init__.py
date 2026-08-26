"""AB Advisor: Bayesian A/B test analysis for product decisions."""

from src.bayesian import AnalysisConfig, PosteriorResult, analyze_metric
from src.decisions import decide_experiment
from src.metrics import ExperimentData, load_experiment

__version__ = "1.0.0"
__all__ = [
    "AnalysisConfig",
    "PosteriorResult",
    "analyze_metric",
    "ExperimentData",
    "load_experiment",
    "decide_experiment",
]
