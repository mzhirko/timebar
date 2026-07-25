"""Model configuration guards: no silent default model, empty extraction
fails loudly (issues found in external testing)."""

import pytest

from tdg_chrono.cli import main


def test_no_default_model_clear_error(monkeypatch):
    pytest.importorskip("openai")
    monkeypatch.delenv("TDG_LLM_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    from tdg_extractors.llm_pipeline import LLMPipeline
    with pytest.raises(ValueError, match="No LLM model configured"):
        LLMPipeline()


def test_model_env_fallback(monkeypatch):
    pytest.importorskip("openai")
    monkeypatch.setenv("TDG_LLM_MODEL", "gemma3:4b")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    from tdg_extractors.llm_pipeline import LLMPipeline
    pipe = LLMPipeline()          # no args: resolved from env, no network call
    assert pipe.model == "gemma3:4b"
    assert pipe.base_url == "http://localhost:11434/v1"


def test_unknown_kwargs_rejected_readably(tmp_path):
    (tmp_path / "doc.txt").write_text("Some text without config.")
    with pytest.raises(SystemExit):
        main(["build", str(tmp_path), "-o", str(tmp_path / "out"),
              "--extractor", "heideltime", "--model", "x"])
