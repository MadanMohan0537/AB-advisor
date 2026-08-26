"""Synthetic A/B test datasets with known effects for demos and tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_checkout_experiment(
    n_per_arm: int = 5000,
    seed: int = 42,
    conversion_c: float = 0.10,
    conversion_t: float = 0.125,
    log_revenue_c: tuple[float, float] = (3.89, 0.45),
    log_revenue_t: tuple[float, float] = (3.98, 0.45),
    time_c: tuple[float, float] = (180.0, 40.0),
    time_t: tuple[float, float] = (180.0, 40.0),
    events_c: float = 3.0,
    events_t: float = 3.05,
    bounce_c: float = 0.42,
    bounce_t: float = 0.405,
) -> pd.DataFrame:
    """Checkout redesign: clear conversion/revenue win, quiet guardrails."""
    rng = np.random.default_rng(seed)
    rows = []
    for variant, n, p_conv, log_rev, time_p, lam, p_bounce in (
        ("control", n_per_arm, conversion_c, log_revenue_c, time_c, events_c, bounce_c),
        ("treatment", n_per_arm, conversion_t, log_revenue_t, time_t, events_t, bounce_t),
    ):
        converted = rng.binomial(1, p_conv, size=n)
        revenue = np.zeros(n, dtype=float)
        buyers = converted == 1
        revenue[buyers] = rng.lognormal(log_rev[0], log_rev[1], size=int(buyers.sum()))
        time_on_page = np.clip(rng.normal(time_p[0], time_p[1], size=n), 5, None)
        events_count = rng.poisson(lam, size=n)
        bounce = rng.binomial(1, p_bounce, size=n)
        for i in range(n):
            rows.append(
                {
                    "user_id": f"{variant[0]}-{i:05d}",
                    "variant": variant,
                    "converted": int(converted[i]),
                    "revenue": round(float(revenue[i]), 2),
                    "time_on_page_sec": round(float(time_on_page[i]), 1),
                    "events_count": int(events_count[i]),
                    "bounce": int(bounce[i]),
                }
            )
    return pd.DataFrame(rows)


def generate_underpowered_experiment(n_per_arm: int = 400, seed: int = 7) -> pd.DataFrame:
    """Small n, tiny lift — should usually recommend keep running."""
    return generate_checkout_experiment(
        n_per_arm=n_per_arm,
        seed=seed,
        conversion_c=0.10,
        conversion_t=0.108,
        log_revenue_c=(3.9, 0.5),
        log_revenue_t=(3.91, 0.5),
        time_c=(120.0, 30.0),
        time_t=(121.0, 30.0),
        events_c=2.0,
        events_t=2.02,
        bounce_c=0.40,
        bounce_t=0.398,
    )


def generate_long_format(wide: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in wide.columns if c not in {"user_id", "variant"}]
    return wide.melt(
        id_vars=["user_id", "variant"],
        value_vars=metric_cols,
        var_name="metric_name",
        value_name="metric_value",
    )


def write_samples(data_dir: str | Path | None = None) -> dict[str, Path]:
    root = Path(data_dir) if data_dir else Path(__file__).resolve().parents[1] / "data"
    root.mkdir(parents=True, exist_ok=True)
    checkout = generate_checkout_experiment()
    underpowered = generate_underpowered_experiment()
    paths = {
        "checkout": root / "sample_experiment.csv",
        "checkout_long": root / "sample_experiment_long.csv",
        "underpowered": root / "sample_underpowered.csv",
    }
    checkout.to_csv(paths["checkout"], index=False)
    generate_long_format(checkout).to_csv(paths["checkout_long"], index=False)
    underpowered.to_csv(paths["underpowered"], index=False)
    return paths


if __name__ == "__main__":
    written = write_samples()
    for name, path in written.items():
        print(f"{name}: {path} ({path.stat().st_size} bytes)")
