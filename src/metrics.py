"""Load A/B test CSVs (wide or long) and infer metric families."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as spstats

from src.bayesian import MetricType

USER_ALIASES = ("user_id", "userid", "uid", "user", "id", "visitor_id", "anonymous_id")
VARIANT_ALIASES = ("variant", "variation", "group", "arm", "bucket", "treatment", "ab_group", "exp_group")
CONTROL_ALIASES = {"control", "a", "0", "baseline", "ctrl", "reference"}
TREATMENT_ALIASES = {"treatment", "b", "1", "variant", "test", "treat", "experimental"}
LONG_METRIC_NAME = ("metric_name", "metric", "kpi", "measure")
LONG_METRIC_VALUE = ("metric_value", "value", "metric_val")


@dataclass
class ExperimentData:
    frame: pd.DataFrame
    user_col: str
    variant_col: str
    metric_cols: list[str]
    format: str  # "wide" | "long"
    control_label: str
    treatment_label: str
    n_control: int
    n_treatment: int
    srm_pvalue: float
    srm_flag: bool
    types: dict[str, MetricType] = field(default_factory=dict)

    def split(self, metric: str) -> tuple[np.ndarray, np.ndarray]:
        df = self.frame
        control = df.loc[df[self.variant_col] == self.control_label, metric].to_numpy(dtype=float)
        treatment = df.loc[df[self.variant_col] == self.treatment_label, metric].to_numpy(dtype=float)
        return control, treatment


def _pick_column(columns: list[str], aliases: tuple[str, ...]) -> str | None:
    lowered = {c.lower().strip(): c for c in columns}
    for alias in aliases:
        if alias in lowered:
            return lowered[alias]
    return None


def _normalize_variant(value: object) -> str:
    text = str(value).strip().lower()
    if text in CONTROL_ALIASES:
        return "control"
    if text in TREATMENT_ALIASES:
        return "treatment"
    return text


def detect_metric_type(series: pd.Series) -> MetricType:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return MetricType.NORMAL
    unique = set(s.unique().tolist())
    if unique <= {0, 1} or unique <= {0.0, 1.0}:
        return MetricType.BINARY

    is_integer = np.allclose(s.to_numpy(), np.round(s.to_numpy()))
    non_negative = bool((s >= 0).all())
    if is_integer and non_negative:
        max_val = float(s.max())
        n_unique = int(s.nunique())
        mean = float(s.mean())
        var = float(s.var(ddof=1)) if s.size > 1 else 0.0
        if max_val <= 80 and n_unique <= 60 and mean > 0 and (var / mean if mean else 0) < 8:
            if unique - {0.0, 1.0} and n_unique > 2:
                return MetricType.POISSON

    frac_zero = float((s == 0).mean())
    positive = s[s > 0]
    if frac_zero >= 0.15 and positive.size >= 20 and non_negative:
        skew = float(positive.skew())
        if skew > 0.8:
            return MetricType.HURDLE_LOGNORMAL

    if non_negative and (s > 0).all() and s.size >= 20:
        log_skew = float(np.log(s).skew())
        raw_skew = float(s.skew())
        if raw_skew > 1.2 and abs(log_skew) < abs(raw_skew):
            return MetricType.LOGNORMAL
    return MetricType.NORMAL


def _srm_pvalue(n_control: int, n_treatment: int, expected_share: float = 0.5) -> float:
    total = n_control + n_treatment
    if total == 0:
        return 1.0
    expected = np.array([expected_share, 1 - expected_share]) * total
    observed = np.array([n_control, n_treatment], dtype=float)
    chi2, p = spstats.chisquare(observed, expected)
    return float(p)


def _wide_from_long(df: pd.DataFrame, user_col: str, variant_col: str, name_col: str, value_col: str) -> pd.DataFrame:
    pivoted = df.pivot_table(
        index=[user_col, variant_col],
        columns=name_col,
        values=value_col,
        aggfunc="first",
    )
    pivoted.columns = [str(c) for c in pivoted.columns]
    return pivoted.reset_index()


def load_experiment(
    source: str | pd.DataFrame,
    user_col: str | None = None,
    variant_col: str | None = None,
    metric_cols: list[str] | None = None,
    control_label: str | None = None,
    treatment_label: str | None = None,
    expected_split: float = 0.5,
    srm_threshold: float = 0.001,
) -> ExperimentData:
    df = pd.read_csv(source) if isinstance(source, str) else source.copy()
    df.columns = [str(c).strip() for c in df.columns]
    columns = list(df.columns)

    name_col = _pick_column(columns, LONG_METRIC_NAME)
    value_col = _pick_column(columns, LONG_METRIC_VALUE)
    user_col = user_col or _pick_column(columns, USER_ALIASES)
    variant_col = variant_col or _pick_column(columns, VARIANT_ALIASES)
    if user_col is None or variant_col is None:
        raise ValueError(
            "Could not detect user_id and variant columns. "
            "Rename them or pass user_col / variant_col explicitly."
        )

    fmt = "long" if name_col and value_col else "wide"
    if fmt == "long":
        df = _wide_from_long(df, user_col, variant_col, name_col, value_col)
        columns = list(df.columns)

    df[variant_col] = df[variant_col].map(_normalize_variant)
    labels = [str(v) for v in df[variant_col].dropna().unique()]
    if control_label:
        control_label = _normalize_variant(control_label)
    if treatment_label:
        treatment_label = _normalize_variant(treatment_label)
    if control_label is None:
        control_label = "control" if "control" in labels else labels[0]
    if treatment_label is None:
        treatment_label = "treatment" if "treatment" in labels else next(
            (lab for lab in labels if lab != control_label), labels[-1]
        )

    reserved = {user_col, variant_col}
    if metric_cols is None:
        metric_cols = [
            c
            for c in df.columns
            if c not in reserved and pd.api.types.is_numeric_dtype(df[c])
        ]
    if not metric_cols:
        raise ValueError("No numeric metric columns found.")

    for col in metric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    n_control = int((df[variant_col] == control_label).sum())
    n_treatment = int((df[variant_col] == treatment_label).sum())
    srm_p = _srm_pvalue(n_control, n_treatment, expected_split)
    types = {col: detect_metric_type(df[col]) for col in metric_cols}

    return ExperimentData(
        frame=df,
        user_col=user_col,
        variant_col=variant_col,
        metric_cols=metric_cols,
        format=fmt,
        control_label=control_label,
        treatment_label=treatment_label,
        n_control=n_control,
        n_treatment=n_treatment,
        srm_pvalue=srm_p,
        srm_flag=srm_p < srm_threshold,
        types=types,
    )
