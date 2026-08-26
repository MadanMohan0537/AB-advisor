"""AB Advisor — Streamlit dashboard for Bayesian A/B test analysis."""

from __future__ import annotations

from pathlib import Path
import os

import pandas as pd
import streamlit as st

from src.bayesian import AnalysisConfig, MetricType, analyze_metric
from src.decisions import DecisionThresholds, classify_metric_role, decide_experiment, default_higher_is_better
from src.metrics import load_experiment
from src.report import html_report, markdown_report, sample_size_binary
from src.summarize import generate_insights
from src.visualize import (
    expected_loss_bars,
    forest_plot,
    format_results_table,
    lift_distribution,
    overview_figure,
    posterior_overlay,
    probability_gauge,
    results_table,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SAMPLES = {
    "Checkout redesign (clear win)": DATA / "sample_experiment.csv",
    "Underpowered copy test (keep running)": DATA / "sample_underpowered.csv",
    "Checkout redesign — long format": DATA / "sample_experiment_long.csv",
}

TYPE_HELP = {
    MetricType.BINARY: "Beta–Binomial (conversion / 0-1)",
    MetricType.NORMAL: "Student-t posterior on the mean",
    MetricType.LOGNORMAL: "Log-normal mean (positive spend / time)",
    MetricType.POISSON: "Gamma–Poisson (counts)",
    MetricType.HURDLE_LOGNORMAL: "Conversion × log-normal spend (ARPU with zeros)",
}


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.4rem; max-width: 1200px; }
        .ab-hero { font-size: 0.95rem; color: #475569; margin-bottom: 1.2rem; }
        .decision-card {
            padding: 1rem 1.2rem; border-radius: 12px; margin: 0.6rem 0 1.2rem 0;
            border: 1px solid transparent;
        }
        .decision-ship { background: #ecfdf3; border-color: #86efac; }
        .decision-hold, .decision-investigate { background: #fef2f2; border-color: #fca5a5; }
        .decision-keep_running { background: #fffbeb; border-color: #fcd34d; }
        .decision-card h3 { margin: 0 0 0.35rem 0; }
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _decision_banner(action: str, headline: str, rationale: str, warnings: list[str]) -> None:
    st.markdown(
        f"""
        <div class="decision-card decision-{action}">
          <h3>{headline}</h3>
          <div>{rationale}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    for warning in warnings:
        st.warning(warning)


def _load_frame(choice: str, uploaded) -> pd.DataFrame:
    if uploaded is not None:
        return pd.read_csv(uploaded)
    path = SAMPLES[choice]
    if not path.exists():
        from src.data_gen import write_samples

        write_samples(DATA)
    return pd.read_csv(path)


def main() -> None:
    st.set_page_config(
        page_title="AB Advisor",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_css()
    st.title("AB Advisor")
    st.markdown(
        '<div class="ab-hero">Bayesian A/B test analyzer for product decisions. '
        "Upload experiment data, get P(treatment is better), expected lift, risk, "
        "and a plain-English recommendation — without waiting on a p-value.</div>",
        unsafe_allow_html=True,
    )

    tab_analyze, tab_size, tab_guide = st.tabs(
        ["Analyze experiment", "Sample size planner", "How to read results"]
    )

    with tab_analyze:
        _analyze_tab()
    with tab_size:
        _sample_size_tab()
    with tab_guide:
        _guide_tab()


def _analyze_tab() -> None:
    with st.sidebar:
        st.header("Data")
        sample_choice = st.selectbox("Sample dataset", list(SAMPLES))
        uploaded = st.file_uploader("Or upload a CSV", type=["csv"])
        experiment_name = st.text_input("Experiment name", "Checkout redesign")

        st.header("Decision bar")
        cred_mass = st.slider("Credible interval mass (HDI)", 0.80, 0.99, 0.90, 0.01)
        mde = st.slider("Minimum useful relative lift (MDE)", 0.0, 0.20, 0.02, 0.005)
        rope = st.slider("ROPE (± relative lift)", 0.0, 0.10, 0.01, 0.005)
        ship_prob = st.slider("Ship if P(better) ≥", 0.80, 0.99, 0.95, 0.01)
        n_samples = st.select_slider("Posterior draws", options=[5000, 10000, 20000, 40000], value=20000)

        st.header("Priors")
        st.caption("Weakly informative defaults. Historical conversion: set mean and strength.")
        prior_mean = st.number_input("Historical conversion (optional)", 0.0, 1.0, 0.0, 0.01)
        prior_n = st.number_input("Prior strength (effective users)", 0, 500, 2, 1)
        if prior_mean > 0 and prior_n >= 2:
            beta_alpha = prior_mean * prior_n
            beta_beta = (1 - prior_mean) * prior_n
        else:
            beta_alpha, beta_beta = 1.0, 1.0
        st.caption(f"Beta({beta_alpha:.2f}, {beta_beta:.2f})")

        st.header("Narrative")
        provider = st.selectbox(
            "LLM provider",
            ["deepseek", "openai", "off"],
            index=0,
            help="DeepSeek is the default. Choose off to keep the deterministic template.",
        )
        default_model = "deepseek-v4-flash" if provider != "openai" else "gpt-4o-mini"
        llm_model = st.text_input(
            "Model",
            value=os.environ.get("DEEPSEEK_MODEL") or os.environ.get("OPENAI_MODEL") or default_model,
        )
        api_key = st.text_input(
            "API key",
            type="password",
            help="Paste a DeepSeek key (platform.deepseek.com). Leave blank to use DEEPSEEK_API_KEY from the environment.",
        )
        use_llm = provider != "off"
        if use_llm and not (api_key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")):
            st.caption("No key yet — the template will be used until DEEPSEEK_API_KEY is set.")

    try:
        raw = _load_frame(sample_choice, uploaded)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        return

    st.subheader("Preview")
    st.dataframe(raw.head(12), width="stretch")

    guessed = None
    try:
        guessed = load_experiment(raw)
        st.caption(
            f"Detected **{guessed.format}** format · control n={guessed.n_control:,} · "
            f"treatment n={guessed.n_treatment:,} · SRM p={guessed.srm_pvalue:.3g}"
        )
        if guessed.srm_flag:
            st.error("Sample ratio mismatch — assignment or logging may be broken.")
    except Exception as exc:
        st.warning(f"Auto-detect needs a tweak: {exc}")

    cols = list(raw.columns)
    default_user = guessed.user_col if guessed else cols[0]
    default_variant = guessed.variant_col if guessed else cols[1]
    default_metrics = guessed.metric_cols if guessed else [
        c for c in cols if c not in {default_user, default_variant}
    ]

    c1, c2, c3 = st.columns(3)
    user_col = c1.selectbox("User column", cols, index=cols.index(default_user) if default_user in cols else 0)
    variant_col = c2.selectbox(
        "Variant column", cols, index=cols.index(default_variant) if default_variant in cols else 1
    )
    metric_cols = c3.multiselect("Metrics to analyze", [c for c in cols if c not in {user_col, variant_col}], default=default_metrics)

    if not metric_cols:
        st.info("Select at least one metric column.")
        return

    try:
        data = load_experiment(raw, user_col=user_col, variant_col=variant_col, metric_cols=metric_cols)
    except Exception as exc:
        st.error(str(exc))
        return

    st.markdown("##### Metric types and roles")
    type_overrides: dict[str, MetricType] = {}
    roles: dict[str, str] = {}
    hib: dict[str, bool] = {}
    header = st.columns([2, 2, 2, 2])
    header[0].caption("Metric")
    header[1].caption("Model")
    header[2].caption("Role")
    header[3].caption("Direction")
    for name in metric_cols:
        row = st.columns([2, 2, 2, 2])
        row[0].markdown(f"**{name}**")
        detected = data.types.get(name, MetricType.NORMAL)
        options = list(MetricType)
        type_overrides[name] = row[1].selectbox(
            f"type_{name}",
            options,
            index=options.index(detected),
            format_func=lambda t: TYPE_HELP[t],
            label_visibility="collapsed",
        )
        role_default = classify_metric_role(name)
        roles[name] = row[2].selectbox(
            f"role_{name}",
            ["primary", "guardrail", "secondary"],
            index=["primary", "guardrail", "secondary"].index(role_default),
            label_visibility="collapsed",
        )
        hib[name] = row[3].selectbox(
            f"dir_{name}",
            [True, False],
            index=0 if default_higher_is_better(name) else 1,
            format_func=lambda x: "Higher is better" if x else "Lower is better",
            label_visibility="collapsed",
        )

    run = st.button("Run Bayesian analysis", type="primary")
    if run:
        config = AnalysisConfig(
            n_samples=int(n_samples),
            cred_mass=float(cred_mass),
            mde=float(mde),
            rope=float(rope),
            beta_alpha=float(beta_alpha),
            beta_beta=float(beta_beta),
        )
        results = []
        progress = st.progress(0.0, text="Sampling posteriors…")
        for i, name in enumerate(metric_cols):
            control, treatment = data.split(name)
            results.append(
                analyze_metric(control, treatment, name, type_overrides[name], config)
            )
            progress.progress((i + 1) / len(metric_cols), text=f"Analyzed {name}")
        progress.empty()
        decision = decide_experiment(
            results,
            roles=roles,
            thresholds=DecisionThresholds(ship_prob=float(ship_prob)),
            srm_flag=data.srm_flag,
            srm_pvalue=data.srm_pvalue,
            higher_is_better=hib,
        )
        insights, source, llm_error = generate_insights(
            results,
            decision,
            cred_mass=config.cred_mass,
            use_llm=use_llm,
            api_key=api_key or None,
            provider=None if provider == "off" else provider,
            model=llm_model or None,
        )
        st.session_state["analysis"] = {
            "data": data,
            "results": results,
            "decision": decision,
            "insights": insights,
            "source": source,
            "llm_error": llm_error,
            "config": config,
            "experiment_name": experiment_name,
            "hib": hib,
            "roles": roles,
        }

    payload = st.session_state.get("analysis")
    if not payload:
        st.info("Configure metrics, then run the analysis.")
        return

    results = payload["results"]
    decision = payload["decision"]
    config: AnalysisConfig = payload["config"]
    roles = payload.get("roles") or {}
    _decision_banner(decision.action, decision.headline, decision.rationale, decision.warnings)

    k1, k2, k3, k4 = st.columns(4)
    primary = next((r for r in results if roles.get(r.metric_name) == "primary"), results[0])
    k1.metric("P(treatment > control)", f"{primary.prob_improvement:.1%}", primary.metric_name)
    k2.metric("Expected lift", f"{primary.expected_relative_lift:+.1%}")
    hdi = primary.relative_lift_hdi
    k3.metric(f"{int(config.cred_mass * 100)}% HDI", f"{hdi[0]:+.1%} to {hdi[1]:+.1%}")
    k4.metric("E[loss] if you ship", f"{primary.expected_loss_ship:.4g}")

    st.plotly_chart(overview_figure(results), width="stretch", key="overview_charts")
    st.plotly_chart(forest_plot(results, config.cred_mass), width="stretch", key="forest_plot")

    table = results_table(results, config.cred_mass)
    st.subheader("Metrics table")
    st.dataframe(format_results_table(table, config.cred_mass), width="stretch")
    st.caption(
        "P(better) is P(treatment mean > control mean), not a p-value. "
        "The frequentist p-value is shown only as a translation aid."
    )

    st.subheader("Narrative")
    src = payload["source"]
    if src.startswith("llm"):
        st.caption(f"DeepSeek / LLM rewrite ({src})")
    elif src == "template_fallback":
        st.caption("Template summary — LLM rewrite failed")
        err = payload.get("llm_error")
        if err:
            st.warning(f"Could not reach the LLM: {err}")
    else:
        st.caption("Template summary (deterministic, no API key required)")
    st.markdown(payload["insights"])

    st.subheader("Per-metric posteriors")
    tabs = st.tabs([r.metric_name for r in results])
    for tab, result in zip(tabs, results):
        with tab:
            left, right = st.columns(2)
            slug = result.metric_name.replace(" ", "_")
            left.plotly_chart(posterior_overlay(result), width="stretch", key=f"posterior_{slug}")
            right.plotly_chart(lift_distribution(result, config.cred_mass), width="stretch", key=f"lift_{slug}")
            g1, g2 = st.columns(2)
            g1.plotly_chart(probability_gauge(result), width="stretch", key=f"gauge_{slug}")
            g2.plotly_chart(expected_loss_bars(result), width="stretch", key=f"loss_{slug}")
            if result.metric_type is MetricType.HURDLE_LOGNORMAL:
                st.caption(
                    "Hurdle model: overall mean = P(convert) × E[spend | convert]. "
                    f"Control converters: {result.extras.get('control', {}).get('n_converters')}, "
                    f"treatment: {result.extras.get('treatment', {}).get('n_converters')}."
                )

    md_text = markdown_report(
        payload["experiment_name"],
        payload["data"],
        results,
        decision,
        config,
        payload["insights"],
    )
    html_text = html_report(md_text, payload["experiment_name"])
    d1, d2 = st.columns(2)
    d1.download_button(
        "Download markdown report",
        md_text.encode("utf-8"),
        file_name="experiment_review.md",
        mime="text/markdown",
    )
    d2.download_button(
        "Download HTML report (print to PDF)",
        html_text.encode("utf-8"),
        file_name="experiment_review.html",
        mime="text/html",
    )


def _sample_size_tab() -> None:
    st.subheader("Binary conversion sample size")
    st.write(
        "Simulate experiments at a hypothesized lift and find how many users per arm "
        "you need before P(treatment > control) usually clears your ship bar. "
        "This is the Bayesian analogue of power analysis."
    )
    c1, c2, c3, c4 = st.columns(4)
    p_c = c1.number_input("Baseline conversion", 0.01, 0.9, 0.10, 0.01)
    mde = c2.number_input("Relative MDE", 0.01, 0.5, 0.20, 0.01)
    target = c3.slider("Ship threshold P(better)", 0.8, 0.99, 0.95, 0.01)
    power = c4.slider("Desired chance of crossing it", 0.5, 0.95, 0.80, 0.05)
    if st.button("Estimate sample size"):
        with st.spinner("Simulating experiments…"):
            out = sample_size_binary(
                p_control=float(p_c),
                relative_mde=float(mde),
                target_prob=float(target),
                power=float(power),
                n_sims=200,
            )
        rec = out["recommended_n_per_arm"]
        if rec:
            st.success(f"About **{rec:,} users per arm** (treatment conversion {out['p_treatment']:.1%}).")
        else:
            st.warning("Even the largest grid point did not reach the desired power. Raise MDE or lower the bar.")
        st.dataframe(pd.DataFrame(out["curve"]).style.format({"power": "{:.0%}"}), width="stretch")


def _guide_tab() -> None:
    st.markdown(
        """
### Why Bayesian for PMs

Frequentist A/B tests answer: *If the treatment did nothing, how surprising is this data?*
That is a p-value. It is not the probability the new checkout is better.

Bayesian analysis answers the question you actually bring to a launch review:

- There is a **93% probability** the treatment increases conversion.
- The **expected lift** is +8.4%.
- The **90% highest-density interval** is [+3.1%, +13.9%].
- If we ship and we are wrong, the **expected loss** is 0.12 conversion points.

Those statements come from the posterior — the distribution of plausible metric means after seeing the data.

### Metric roles

| Role | What it is | How we treat it |
|---|---|---|
| **Primary / north star** | The reason you ran the test (conversion, revenue) | Must clear P(better) and MDE |
| **Guardrail** | Things that must not break (errors, bounce, latency) | Can **block** a ship even if the primary wins |
| **Secondary** | Learning metrics (time on page, events) | Reported, not used as a veto by default |

### Priors

The default Beta(1, 1) prior is uniform on conversion rates — it barely speaks.
If last quarter's checkout converted at 10% and you trust ~100 users of history, set
prior mean 0.10 and strength 100 → Beta(10, 90). The prior fades as experiment traffic grows.

### Peeking

p-values are invalid if you peek and stop when they look good. These Bayesian rules
(expected loss, P(better) vs a pre-registered threshold) **remain valid under sequential monitoring**.
Do not mix the two: if you peeked, do not also quote a frequentist p-value as if the sample size were fixed.

### Limitations

Independent users (SUTVA), correct assignment, a model that matches the metric family,
and a pre-registered primary metric. SRM (sample ratio mismatch) means you should not ship from the numbers at all.
        """
    )


if __name__ == "__main__":
    main()
