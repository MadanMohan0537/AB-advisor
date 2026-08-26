"""LLM provider resolution — DeepSeek first, no live API calls."""

from __future__ import annotations

from src.summarize import (
    DEEPSEEK_BASE_URL,
    DEEPSEEK_DEFAULT_MODEL,
    generate_insights,
    resolve_llm_settings,
)


def test_no_key_returns_none(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert resolve_llm_settings() is None


def test_deepseek_env_is_default(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    settings = resolve_llm_settings()
    assert settings is not None
    assert settings.provider == "deepseek"
    assert settings.base_url == DEEPSEEK_BASE_URL
    assert settings.model == DEEPSEEK_DEFAULT_MODEL
    assert settings.extra_body == {"thinking": {"type": "disabled"}}


def test_sidebar_key_assumes_deepseek(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = resolve_llm_settings(api_key="sk-from-ui")
    assert settings is not None
    assert settings.provider == "deepseek"
    assert settings.api_key == "sk-from-ui"


def test_openai_only_when_that_key_is_set(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    settings = resolve_llm_settings()
    assert settings is not None
    assert settings.provider == "openai"
    assert settings.base_url is None


def test_generate_insights_without_key_uses_template(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from src.bayesian import AnalysisConfig, MetricType, analyze_metric
    from src.decisions import decide_experiment
    import numpy as np

    rng = np.random.default_rng(0)
    result = analyze_metric(
        rng.binomial(1, 0.1, 200).astype(float),
        rng.binomial(1, 0.12, 200).astype(float),
        "converted",
        MetricType.BINARY,
        AnalysisConfig(n_samples=2000, seed=0),
    )
    md, source, err = generate_insights([result], decide_experiment([result]), use_llm=True)
    assert source == "template"
    assert err is None
    assert "converted" in md
