"""Three model roles that must not tread on each other.

The tool talks to a model for three unrelated jobs, and there is no reason
they should be the same model or provider. Running a small local model for
extraction, a hosted one for answering and a dedicated embedding model was
not expressible: every role fell back to one OPENAI_API_KEY, and the
embedding client hardcoded the literal string "ollama" as its key, so a
hosted embedding service could never authenticate.
"""

from __future__ import annotations

import pytest

from tdg_chrono.models import ModelConfig, client_for, resolve, summarise

ROLES = ("extract", "answer", "embed")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "TDG_LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    for role in ("EXTRACT", "ANSWER", "EMBED"):
        for suffix in ("MODEL", "BASE_URL", "API_KEY"):
            monkeypatch.delenv(f"TDG_{role}_{suffix}", raising=False)


def test_a_flag_beats_the_environment(monkeypatch):
    monkeypatch.setenv("TDG_ANSWER_MODEL", "from-env")
    assert resolve("answer", model="from-flag").model == "from-flag"


def test_a_role_specific_variable_beats_the_shared_one(monkeypatch):
    monkeypatch.setenv("TDG_LLM_MODEL", "shared")
    monkeypatch.setenv("TDG_ANSWER_MODEL", "specific")
    assert resolve("answer").model == "specific"
    assert resolve("extract").model == "shared", "the fallback still applies"


def test_three_providers_at_once_do_not_collide(monkeypatch):
    """The setup that was previously impossible."""
    monkeypatch.setenv("TDG_EXTRACT_MODEL", "llama3")
    monkeypatch.setenv("TDG_EXTRACT_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("TDG_ANSWER_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("TDG_ANSWER_API_KEY", "sk-hosted")
    monkeypatch.setenv("TDG_EMBED_MODEL", "embed-v3")
    monkeypatch.setenv("TDG_EMBED_BASE_URL", "https://embeddings.example/v1")
    monkeypatch.setenv("TDG_EMBED_API_KEY", "sk-embed")

    extract, answer, embed = (resolve(r) for r in ROLES)
    assert (extract.model, extract.is_local) == ("llama3", True)
    assert (answer.model, answer.api_key) == ("gpt-4o-mini", "sk-hosted")
    assert (embed.model, embed.api_key) == ("embed-v3", "sk-embed")
    assert answer.api_key != embed.api_key, "keys must stay independent"


def test_a_hosted_embedding_service_gets_a_real_key(monkeypatch):
    """The embedding client used to hardcode "ollama" and could not
    authenticate anywhere else."""
    monkeypatch.setenv("TDG_EMBED_API_KEY", "sk-real")
    assert resolve("embed", model="e").effective_key == "sk-real"


def test_a_local_endpoint_needs_no_credential():
    cfg = resolve("embed", model="e", base_url="http://localhost:11434/v1")
    assert cfg.is_local
    assert cfg.effective_key, "the client insists on some string"


def test_an_unconfigured_role_is_simply_off():
    assert not resolve("embed").configured


def test_asking_for_a_client_without_a_model_says_which_variable():
    with pytest.raises(SystemExit) as e:
        client_for(resolve("answer"))
    assert "TDG_ANSWER_MODEL" in str(e.value)


def test_a_hosted_role_without_a_key_says_both_ways_out():
    with pytest.raises(SystemExit) as e:
        client_for(resolve("answer", model="gpt-4o-mini"))
    message = str(e.value)
    assert "TDG_ANSWER_API_KEY" in message
    assert "--base-url" in message, "a local model is the other option"


def test_an_unknown_role_is_refused():
    with pytest.raises(ValueError):
        resolve("translate")


def test_the_summary_names_every_live_role(monkeypatch):
    monkeypatch.setenv("TDG_EXTRACT_MODEL", "llama3")
    monkeypatch.setenv("TDG_ANSWER_MODEL", "gpt-4o-mini")
    text = summarise([resolve(r) for r in ROLES])
    assert "llama3" in text and "gpt-4o-mini" in text
    assert "embed" not in text, "an unconfigured role should not be announced"
