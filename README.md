# AB Advisor

**Bayesian A/B test analyzer for product decisions.**

Upload experiment data (user IDs, variant assignment, metrics). The tool fits conjugate Bayesian models, then says things a launch review can actually use:

> There is a **99.9%** probability the new checkout increases conversion, with an expected lift of **+24%**. The 90% highest-density interval is **[+17%, +32%]** — it does not include zero. Expected loss if you ship is tiny compared with holding.

That is not a p-value. A p-value answers *“how surprising is this data if nothing changed?”* You do not walk into an exec review to discuss surprisingness. You walk in to decide whether to ship.

![Dashboard](screenshots/dashboard.png)

## Why Bayesian (and why it is harder)

| Frequentist t-test / z-test | This tool |
|---|---|
| p = 0.04 | P(treatment > control) = 97% |
| “Significant” vs “not” | Expected lift, 90% HDI, chance of beating the MDE |
| Invalid if you peek | Probability and expected-loss rules stay valid under sequential monitoring |
| Binary conversion only in the simple case | Conversion, revenue (including zeros), counts, continuous |

The cost is real: you must pick a **prior**, a **minimum useful lift (MDE)**, and a **loss threshold**. Those are product judgments, not statistical afterthoughts. AB Advisor makes them explicit in the sidebar instead of hiding them in a stats-package default.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.data_gen
streamlit run app.py
```

Open the app, leave **Checkout redesign (clear win)** selected, click **Run Bayesian analysis**.

Headless / CI:

```bash
python -m src.cli data/sample_experiment.csv --name "Checkout redesign" --out docs/experiment_review.md
pytest -q
```

Optional LLM rewrite of the narrative uses **DeepSeek** (OpenAI-compatible). Set `DEEPSEEK_API_KEY` or paste the key in the sidebar. The model only sees the computed JSON. If the key is missing, a deterministic template is used — numbers never depend on the LLM.

```bash
export DEEPSEEK_API_KEY=sk-...
streamlit run app.py
# or
python -m src.cli data/sample_experiment.csv --name "Checkout redesign" --llm
```

Defaults: `https://api.deepseek.com` and `deepseek-v4-flash` (thinking disabled so the brief stays tight). Override with `DEEPSEEK_MODEL` or the Model field in the sidebar. OpenAI remains available if you switch the provider dropdown.

## Input format

**Wide** (one row per user):

```text
user_id,variant,converted,revenue,time_on_page_sec,events_count,bounce
c-00001,control,0,0.00,162.4,2,1
t-00018,treatment,1,54.20,171.1,4,0
```

**Long** (one row per user × metric):

```text
user_id,variant,metric_name,metric_value
c-00001,control,converted,0
c-00001,control,revenue,0.00
```

Variant aliases `A`/`B`, `0`/`1`, `control`/`treatment` are normalized automatically. Multiple metrics run in one pass.

## What the engine actually does

No PyMC sampling loop. Each metric family has a **conjugate posterior**, so a dashboard refresh is milliseconds, not minutes. Monte Carlo is used only to turn two posteriors into P(better), lift, and expected loss (~20k draws).

| Metric pattern | Model | Prior (default) |
|---|---|---|
| 0/1 conversion, bounce | Beta–Binomial | Beta(1, 1) uniform |
| Continuous (time on page) | Unknown mean & variance → Student-t on the mean | Weakly informative |
| Strictly positive skewed | Log-normal mean | Inverse-χ² on log-variance |
| Counts (events) | Gamma–Poisson | Gamma(1, 1) |
| Revenue with many zeros (ARPU) | Hurdle: Beta conversion × log-normal spend \| convert | Same as above |

For each metric the report includes:

1. **P(treatment > control)** — probability of improvement (direction can be flipped for bounce / latency).
2. **Expected relative lift** — mean of (treatment − control) / control.
3. **90% HDI** — shortest interval containing 90% of posterior lift mass.
4. **P(lift > MDE)** and **P(harm beyond MDE)**.
5. **ROPE probability** — share of lift inside ± a “too small to care” band.
6. **Expected loss** of shipping vs holding, on the raw metric scale.
7. A frequentist p-value **as a footnote only**, so you can translate for teams that still ask.

### Expected loss (the decision metric)

If you ship treatment, you lose when treatment is actually worse:

\[
L_{\text{ship}} = \mathbb{E}[\max(\theta_C - \theta_T, 0)]
\]

Hold when \(L_{\text{ship}}\) is below a business threshold (for conversion, often a small fraction of baseline). This is how GrowthBook-style Bayesian stacks decide, and it is more honest than “p < 0.05”.

## Dashboard

1. Upload CSV or pick a sample.
2. Confirm user / variant / metric columns (auto-detected).
3. Override metric **type**, **role** (primary / guardrail / secondary), and **direction**.
4. Set HDI mass, MDE, ROPE, ship bar, and optional historical conversion prior.
5. Run analysis: KPI cards, forest plot, density overlays, lift with shaded HDI, expected-loss bars, narrative, downloadable markdown + HTML (print to PDF).

There is also a **sample size planner**: simulate binary experiments at a hypothesized lift and report users per arm until P(better) usually clears your ship threshold.

## Repository layout

```text
app.py                 Streamlit UI
src/bayesian.py        Conjugate posteriors + Monte Carlo summaries
src/metrics.py         Wide/long loaders, type detection, SRM check
src/decisions.py       Ship / hold / keep-running / investigate
src/summarize.py       Templates + optional DeepSeek rewrite
src/visualize.py       Plotly posteriors, lift, forest, gauges
src/report.py          Markdown/HTML experiment review + sample size
src/data_gen.py        Synthetic checkout + underpowered datasets
src/cli.py             Batch report from a CSV
data/                  Sample CSVs
notebooks/             Math walkthrough
docs/                  Example experiment-review memo
tests/                 Engine, loaders, decision rules
```

## For product managers

### Choosing metrics

- **North star / primary** — the reason the experiment exists (conversion, revenue). Pre-register one. AB Advisor will not let a pretty secondary metric outvote a weak primary.
- **Guardrails** — bounce, errors, refunds, latency. A guardrail with material P(harm) **blocks ship** even if conversion won.
- **Secondary** — time on page, event counts. Learning only. Inconclusive secondaries do not stall a clean primary win.

### Setting priors from history

If last quarter’s checkout converted at 10% and you would trust that as ~100 users of information (not 100,000 — you do not want the past to dominate):

- Prior mean = 0.10, strength = 100 → Beta(10, 90).

The prior’s influence shrinks as experiment `n` grows. Strength 2 (the uniform default) is the right starting point when you are unsure.

### Talking to executives

Do not say “we got significance.” Say:

1. **Probability the change is positive.**
2. **How big we think it is** (expected lift + interval).
3. **What we risk if we are wrong** (expected loss; guardrails).
4. **What we will do next** (ramp, hold, or wait for a pre-registered bar).

The sample memo in [`docs/sample_experiment_review.md`](docs/sample_experiment_review.md) is the tone to copy.

### Peeking and sequential tests

If someone refreshed this dashboard daily and stopped when P(better) crossed 95%, that is a **valid Bayesian sequential rule** — provided the threshold was set in advance. It is **not** valid to peek, then also quote a frequentist p-value as if `n` were fixed. The UI shows p-values only as a translation aid; do not mix the two stories in a review.

## Sample data (known effects)

`data/sample_experiment.csv` is a checkout redesign, 5,000 users per arm (realized values from seed 42):

| Metric | Control | Treatment | What you should see |
|---|---|---|---|
| converted | 9.9% | 12.9% | Clear ship (P(better) ≈ 100%, expected lift ≈ +31%) |
| revenue (ARPU, zeros included) | ~$5.39 | ~$7.56 | Clear ship via the hurdle model |
| time_on_page_sec | ~180s | ~180s | Null (HDI includes 0) |
| events_count | ~3.03 | ~3.08 | Mild / inconclusive |
| bounce | 41.6% | 40.3% | Mild improvement (lower is better; does not block) |

`data/sample_underpowered.csv` is the same story with n = 400 and a tiny lift — the tool should say **keep running**, not “not significant, kill it.”

## Limitations (read before you ship)

- **Independence / SUTVA.** Users do not interfere. Marketplace, social, or cache effects need a different design.
- **Assignment is random and logged correctly.** Sample ratio mismatch (SRM) is flagged with a chi-square test; if it fires, the recommendation is *investigate*, not ship.
- **Model family matches the metric.** A skewed revenue column forced through a Normal model will lie. Auto-detect is a starting point — override in the UI.
- **Multiple metrics.** Peeking at twelve KPIs inflates the chance something looks good. Pre-register the primary.
- **No CUPED / variance reduction, no switchback, no clustered users.** Those are extensions, not in v1.
- **The LLM cannot invent numbers**, but it can still over-smooth caveats. Prefer the template when the decision is contentious.

## Tests

```bash
pytest -q
```

Coverage includes a known conversion lift (P(better) → 1), a true null (HDI contains 0), Poisson rates, hurdle revenue, long-format CSV, SRM, and the ship/investigate decision rules.

## License

Use and modify freely for internal experiment reviews.
