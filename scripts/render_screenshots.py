#!/usr/bin/env python3
"""Render static PNG screenshots from the sample experiment (no browser required)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bayesian import AnalysisConfig, analyze_metric  # noqa: E402
from src.data_gen import write_samples  # noqa: E402
from src.decisions import decide_experiment  # noqa: E402
from src.metrics import load_experiment  # noqa: E402
from src.report import markdown_report  # noqa: E402
from src.summarize import generate_insights  # noqa: E402


def _kde(samples: np.ndarray, n: int = 240):
    samples = np.asarray(samples, dtype=float)
    lo, hi = np.quantile(samples, [0.002, 0.998])
    pad = 0.06 * (hi - lo + 1e-9)
    grid = np.linspace(lo - pad, hi + pad, n)
    std = samples.std()
    bw = max(1.06 * std * samples.size ** -0.2, 1e-9)
    z = (grid[:, None] - samples[None, :]) / bw
    dens = np.exp(-0.5 * z**2).mean(axis=1) / (bw * np.sqrt(2 * np.pi))
    return grid, dens


def main() -> None:
    out = ROOT / "screenshots"
    out.mkdir(parents=True, exist_ok=True)
    write_samples(ROOT / "data")
    data = load_experiment(str(ROOT / "data" / "sample_experiment.csv"))
    cfg = AnalysisConfig(n_samples=12_000, seed=42)
    results = [
        analyze_metric(*data.split(name), name, data.types[name], cfg)
        for name in data.metric_cols
    ]
    decision = decide_experiment(
        results, srm_flag=data.srm_flag, srm_pvalue=data.srm_pvalue, higher_is_better={"bounce": False}
    )
    insights, _, _ = generate_insights(results, decision, use_llm=False)
    report = markdown_report("Checkout redesign", data, results, decision, cfg, insights)
    (ROOT / "docs" / "generated_experiment_review.md").write_text(report, encoding="utf-8")

    conv = next(r for r in results if r.metric_name == "converted")
    fig = plt.figure(figsize=(13.2, 8.2), facecolor="#F8FAFC")
    gs = GridSpec(2, 3, figure=fig, height_ratios=[0.22, 0.78], hspace=0.38, wspace=0.32)

    banner = fig.add_subplot(gs[0, :])
    banner.set_axis_off()
    banner.set_xlim(0, 1)
    banner.set_ylim(0, 1)
    banner.add_patch(plt.Rectangle((0.01, 0.08), 0.98, 0.84, color="#ECFDF3", transform=banner.transAxes, zorder=0))
    banner.text(0.03, 0.62, "AB Advisor  ·  Checkout redesign", fontsize=16, fontweight="bold", color="#0F172A", va="center")
    banner.text(0.03, 0.28, decision.headline, fontsize=12, color="#166534", va="center")

    ax1 = fig.add_subplot(gs[1, 0])
    xc, yc = _kde(conv.control_samples)
    xt, yt = _kde(conv.treatment_samples)
    ax1.fill_between(xc, yc, color="#64748B", alpha=0.35, label="Control")
    ax1.plot(xc, yc, color="#64748B")
    ax1.fill_between(xt, yt, color="#2563EB", alpha=0.35, label="Treatment")
    ax1.plot(xt, yt, color="#2563EB")
    ax1.set_title("Posterior conversion rate")
    ax1.set_xlabel("Conversion")
    ax1.legend(frameon=False, fontsize=8)
    ax1.spines[["top", "right"]].set_visible(False)

    ax2 = fig.add_subplot(gs[1, 1])
    xl, yl = _kde(conv.relative_lift_samples)
    lo, hi = conv.relative_lift_hdi
    ax2.fill_between(xl, yl, color="#0F766E", alpha=0.2)
    mask = (xl >= lo) & (xl <= hi)
    ax2.fill_between(xl[mask], yl[mask], color="#0F766E", alpha=0.45, label="90% HDI")
    ax2.plot(xl, yl, color="#0F766E")
    ax2.axvline(0, color="#DC2626", lw=1.2)
    ax2.set_title("Posterior relative lift")
    ax2.set_xlabel("Lift")
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    ax2.legend(frameon=False, fontsize=8)
    ax2.spines[["top", "right"]].set_visible(False)

    ax3 = fig.add_subplot(gs[1, 2])
    names = [r.metric_name for r in results][::-1]
    means = [r.expected_relative_lift for r in results][::-1]
    lows = [r.relative_lift_hdi[0] for r in results][::-1]
    highs = [r.relative_lift_hdi[1] for r in results][::-1]
    y = np.arange(len(names))
    ax3.axvline(0, color="#94A3B8", ls="--", lw=1)
    ax3.hlines(y, lows, highs, color="#94A3B8", lw=2)
    colors = ["#16A34A" if m >= 0 else "#DC2626" for m in means]
    ax3.scatter(means, y, c=colors, s=40, zorder=3)
    ax3.set_yticks(y)
    ax3.set_yticklabels(names)
    ax3.set_title("Expected lift (90% HDI)")
    ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    ax3.spines[["top", "right"]].set_visible(False)

    fig.savefig(out / "dashboard.png", dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(out / "posterior_conversion.png", dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    fig2, ax = plt.subplots(figsize=(7.2, 4.2), facecolor="white")
    ax.fill_between(xl[mask], yl[mask], color="#0F766E", alpha=0.4)
    ax.plot(xl, yl, color="#0F766E")
    ax.axvline(0, color="#DC2626", label="No lift")
    ax.axvline(conv.expected_relative_lift, color="#0F766E", ls="--", label="Expected lift")
    ax.set_title("Conversion lift posterior")
    ax.set_xlabel("Relative lift")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig2.savefig(out / "lift_conversion.png", dpi=130, bbox_inches="tight")
    plt.close(fig2)
    print("wrote", out)


if __name__ == "__main__":
    main()
