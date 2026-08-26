"""CLI for batch analysis without Streamlit."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.bayesian import AnalysisConfig, analyze_metric
from src.decisions import decide_experiment
from src.metrics import load_experiment
from src.report import markdown_report
from src.summarize import generate_insights


def main() -> None:
    parser = argparse.ArgumentParser(description="Bayesian A/B analysis → markdown report")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--name", default="Experiment")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--mde", type=float, default=0.02)
    parser.add_argument("--cred", type=float, default=0.90)
    args = parser.parse_args()

    data = load_experiment(str(args.csv))
    config = AnalysisConfig(mde=args.mde, cred_mass=args.cred)
    results = [
        analyze_metric(*data.split(name), name, data.types[name], config)
        for name in data.metric_cols
    ]
    decision = decide_experiment(results, srm_flag=data.srm_flag, srm_pvalue=data.srm_pvalue)
    insights, _ = generate_insights(results, decision, cred_mass=config.cred_mass, use_llm=False)
    report = markdown_report(args.name, data, results, decision, config, insights)
    if args.out:
        args.out.write_text(report, encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(report)


if __name__ == "__main__":
    main()
