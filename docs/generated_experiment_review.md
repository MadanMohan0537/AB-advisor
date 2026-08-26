# Experiment Review: Checkout redesign

Generated 2026-08-26 21:27 UTC by AB Advisor (Bayesian A/B test analyzer).

## Decision

**Ship the treatment**

Primary metrics show a high probability of a useful improvement, and guardrails are not signaling material harm. Secondary metrics are reported for learning and do not veto this call.

### Warnings
- None

## Design snapshot

- Control label: `control` (n=5,000)
- Treatment label: `treatment` (n=5,000)
- Sample ratio mismatch p-value: 1 (ok)
- Credible interval mass: 90% HDI
- Minimum useful lift (MDE): 2.00%
- ROPE: ±1.00%
- Conversion prior: Beta(1.0, 1.0)
- Poisson prior: Gamma(shape=1.0, rate=1.0)
- Posterior draws: 12,000

## Results

| Metric | Type | n_c | n_t | Control | Treatment | Observed lift | P(better) | E[lift] | 90% HDI | P(> MDE) | P(harm) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| converted | binary | 5,000 | 5,000 | 0.0986 | 0.1294 | +31.2% | 100.0% | +31.4% | [+19.7%, +43.8%] | 100.0% | 0.0% |
| revenue | hurdle_lognormal | 5,000 | 5,000 | 5.389 | 7.564 | +40.4% | 100.0% | +40.8% | [+26.3%, +55.6%] | 100.0% | 0.0% |
| time_on_page_sec | normal | 5,000 | 5,000 | 179.7 | 179.9 | +0.1% | 58.0% | +0.1% | [-0.6%, +0.8%] | 0.0% | 0.0% |
| events_count | poisson | 5,000 | 5,000 | 3.031 | 3.083 | +1.7% | 93.0% | +1.7% | [-0.2%, +3.6%] | 39.5% | 0.1% |
| bounce | binary | 5,000 | 5,000 | 0.4164 | 0.4034 | -3.1% | 9.7% | -3.1% | [-7.1%, +0.6%] | 1.6% | 67.6% |

## Per-metric actions

- **converted** (primary): `ship` — 100.0% probability of improvement and 100.0% chance of beating the 31.4% expected lift's MDE.
- **revenue** (primary): `ship` — 100.0% probability of improvement and 100.0% chance of beating the 40.8% expected lift's MDE.
- **time_on_page_sec** (secondary): `stop_futility` — 97.2% of the posterior lift sits inside the practical equivalence window. Further traffic is unlikely to reveal a useful effect.
- **events_count** (secondary): `keep_running` — Evidence is mixed: P(better)=93.0%, P(lift > MDE)=39.5%, 26.1% of mass in the ROPE. Collect more data.
- **bounce** (guardrail): `keep_running` — Evidence is mixed: P(better)=90.3%, P(lift > MDE)=67.6%, 14.4% of mass in the ROPE. Collect more data.

## Narrative

## Recommendation: Ship the treatment

Primary metrics show a high probability of a useful improvement, and guardrails are not signaling material harm. Secondary metrics are reported for learning and do not veto this call.

### Metric-by-metric read
- Strong evidence that treatment increases **converted**. P(treatment > control) = 100.0%, expected lift = +31.4%, 90% HDI = [+19.7%, +43.8%]. P(lift > MDE) = 100.0%; P(harm beyond MDE) = 0.0%. Expected loss if you ship = 0 (hold = 0.03074) on the raw metric scale. Observed means: control 0.0986 (n=5,000) vs treatment 0.1294 (n=5,000).
- Strong evidence that treatment increases **revenue**. P(treatment > control) = 100.0%, expected lift = +40.8%, 90% HDI = [+26.3%, +55.6%]. P(lift > MDE) = 100.0%; P(harm beyond MDE) = 0.0%. Expected loss if you ship = 0 (hold = 2.186) on the raw metric scale. Observed means: control 5.389 (n=5,000) vs treatment 7.564 (n=5,000).
- Weak / mixed evidence that treatment increases **time_on_page_sec**. P(treatment > control) = 58.0%, expected lift = +0.1%, 90% HDI = [-0.6%, +0.8%]. P(lift > MDE) = 0.0%; P(harm beyond MDE) = 0.0%. Expected loss if you ship = 0.2462 (hold = 0.4084) on the raw metric scale. Observed means: control 179.7 (n=5,000) vs treatment 179.9 (n=5,000).
- Moderate evidence that treatment increases **events_count**. P(treatment > control) = 93.0%, expected lift = +1.7%, 90% HDI = [-0.2%, +3.6%]. P(lift > MDE) = 39.5%; P(harm beyond MDE) = 0.1%. Expected loss if you ship = 0.001129 (hold = 0.05255) on the raw metric scale. Observed means: control 3.031 (n=5,000) vs treatment 3.083 (n=5,000).
- Strong evidence against an improvement that treatment decreases **bounce**. P(treatment > control) = 9.7%, expected lift = -3.1%, 90% HDI = [-7.1%, +0.6%]. P(lift > MDE) = 1.6%; P(harm beyond MDE) = 67.6%. Expected loss if you ship = 0.0133 (hold = 0.0004256) on the raw metric scale. Observed means: control 0.4164 (n=5,000) vs treatment 0.4034 (n=5,000).

### Suggested next steps
- Roll out to 100% and keep a holdout if the change is expensive to reverse.
- Watch guardrails for 24–72 hours after ramp for novelty or latency effects.
- Document the prior, MDE, and decision threshold you actually used.

_Caveats: users are assumed independent (SUTVA), assignment is random, and the model family matches the metric type. These are not p-values._

## How to read this for executives

- **P(better)** is the probability the treatment mean exceeds the control mean, given the data and prior. It is not a p-value.
- **Expected lift** is the average of the posterior on (treatment − control) / control.
- **HDI** is the highest-density credible interval: the shortest range containing 90% of posterior lift mass.
- **Expected loss** is the average amount you give up on the raw metric if you pick the worse variant. Ship when that cost is below a business threshold.
- **ROPE** (region of practical equivalence) is the lift band you consider "too small to care." High ROPE probability supports stopping for futility.

## Limitations

- Users are modeled as independent (SUTVA). Network, marketplace, or interference effects are not in the likelihood.
- Peeking is valid for these Bayesian expected-loss / probability rules, but not if you also hunt for a frequentist p-value.
- Priors are weakly informative by default. If you have a strong historical conversion rate, encode it as Beta(α, β) with a modest effective sample size (50–200).
- Multiple metrics inflate the chance that *something* looks good. Pre-register a primary metric and treat the rest as guardrails or diagnostics.
- The hurdle-lognormal model for revenue assumes zeros are non-converters and positive spend is log-normal.

---
*This document is a decision aid, not a substitute for instrumentation review or qualitative product judgment.*
