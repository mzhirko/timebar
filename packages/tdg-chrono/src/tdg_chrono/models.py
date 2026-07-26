"""Which model does what, and where each one lives.

The tool talks to a language model for three unrelated jobs, and there is no
reason they should be the same model or even the same provider:

  extract   reads documents and proposes dated facts
  answer    writes the prose in `tdg-chrono ask`
  embed     compares entity names, and ranks passages for retrieval

Running a small local model for extraction, a larger hosted one for
answering, and a dedicated embedding model for similarity is an ordinary
setup. It used not to be expressible: every role fell back to one
`OPENAI_API_KEY`, and the embedding client hardcoded the string "ollama" as
its key, so a hosted embedding service could not authenticate at all.

Each role now resolves independently, in this order:

  1. the command-line flag
  2. that role's own environment variable
  3. the shared fallback, for the common case of one provider for everything

Nothing is guessed. A role with no model configured is simply off, except
extraction, which has no default and says so.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

# Per-role environment variables, then the shared fallback each defers to.
_ENV = {
    "extract": {"model": ("TDG_EXTRACT_MODEL", "TDG_LLM_MODEL"),
                "base_url": ("TDG_EXTRACT_BASE_URL", "OPENAI_BASE_URL"),
                "api_key": ("TDG_EXTRACT_API_KEY", "OPENAI_API_KEY")},
    "answer": {"model": ("TDG_ANSWER_MODEL", "TDG_LLM_MODEL"),
               "base_url": ("TDG_ANSWER_BASE_URL", "OPENAI_BASE_URL"),
               "api_key": ("TDG_ANSWER_API_KEY", "OPENAI_API_KEY")},
    "embed": {"model": ("TDG_EMBED_MODEL",),
              "base_url": ("TDG_EMBED_BASE_URL", "OPENAI_BASE_URL"),
              "api_key": ("TDG_EMBED_API_KEY", "OPENAI_API_KEY")},
}


@dataclass
class ModelConfig:
    """One role's model, endpoint and key."""

    role: str
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None

    @property
    def configured(self) -> bool:
        return bool(self.model)

    @property
    def is_local(self) -> bool:
        """A base_url pointing at this machine needs no real credential."""
        url = self.base_url or ""
        return any(h in url for h in ("localhost", "127.0.0.1", "0.0.0.0",
                                      "::1", "host.docker.internal"))

    @property
    def effective_key(self) -> str:
        """The key to hand the client.

        A local server ignores it but the client insists on one, so a
        placeholder is supplied rather than failing on a credential that
        was never needed.
        """
        return self.api_key or ("local" if self.base_url else "")

    def describe(self) -> str:
        where = self.base_url or "the default OpenAI endpoint"
        return f"{self.role}: {self.model} via {where}"


def _first(names: tuple) -> Optional[str]:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def resolve(role: str, *, model: Optional[str] = None,
            base_url: Optional[str] = None,
            api_key: Optional[str] = None) -> ModelConfig:
    """Settle one role's configuration: flag, then its env, then the shared one."""
    if role not in _ENV:
        raise ValueError(f"unknown model role {role!r}; expected one of "
                         f"{sorted(_ENV)}")
    env = _ENV[role]
    return ModelConfig(
        role=role,
        model=model or _first(env["model"]),
        base_url=base_url or _first(env["base_url"]),
        api_key=api_key or _first(env["api_key"]),
    )


def client_for(config: ModelConfig):
    """An OpenAI-compatible client for this role, or a clear refusal."""
    if not config.configured:
        raise SystemExit(
            f"error: no model configured for {config.role}.\n"
            f"       Pass --model, or set "
            f"{_ENV[config.role]['model'][0]}.")
    if not config.base_url and not config.api_key:
        raise SystemExit(
            f"error: no API key for {config.role}, and no --base-url to say "
            "where the model is.\n"
            f"       Either set {_ENV[config.role]['api_key'][0]} (or "
            "OPENAI_API_KEY) for a hosted provider,\n"
            "       or pass --base-url http://localhost:11434/v1 for a local "
            "one.")
    from openai import OpenAI

    kwargs = {"api_key": config.effective_key}
    if config.base_url:
        kwargs["base_url"] = config.base_url
    return OpenAI(**kwargs)


def summarise(configs: list) -> str:
    """One line naming what will be used, so a mixed setup is visible."""
    live = [c for c in configs if c.configured]
    if not live:
        return "no models configured"
    return "; ".join(c.describe() for c in live)
