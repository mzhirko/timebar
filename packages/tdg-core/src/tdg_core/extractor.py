"""The one interface every extractor implements (Phase 0.4).

An extractor turns document text into a TDG. Anything satisfying this
protocol is interchangeable: HeidelTime+spaCy, an LLM, a human-filled
form, or a third-party system that never imports this codebase (it can
just emit schema-valid JSON instead).

Third-party extractors register through the ``tdg.extractors`` entry
point group:

    [project.entry-points."tdg.extractors"]
    my-extractor = "my_pkg.extract:MyExtractor"

and become available to ``tdg-chrono build --extractor my-extractor``
and to ``tdg-bench`` with no code changes here.
"""

from __future__ import annotations

from datetime import date
from importlib.metadata import entry_points
from typing import Optional, Protocol, runtime_checkable

from tdg_core.tdg import TemporalDependencyGraph


@runtime_checkable
class Extractor(Protocol):
    """Turns document text into a TemporalDependencyGraph."""

    name: str

    def extract(
        self,
        text: str,
        *,
        document_id: str,
        document_type: str = "unknown",
        reference_date: Optional[date] = None,
    ) -> TemporalDependencyGraph: ...


def available_extractors() -> dict[str, "type"]:
    """Discover installed extractors via the ``tdg.extractors`` entry point group.

    Returns {name: class}. Loading is deferred to the caller so that an
    extractor with heavy dependencies (spacy, openai) costs nothing
    until actually selected.
    """
    found: dict[str, type] = {}
    for ep in entry_points(group="tdg.extractors"):
        found[ep.name] = ep  # EntryPoint; call .load() when selected
    return found


def load_extractor(name: str, **kwargs) -> Extractor:
    """Load and instantiate a registered extractor by name.

    Raises a readable error naming the missing extra when the
    extractor's dependencies are not installed.
    """
    eps = available_extractors()
    if name not in eps:
        raise KeyError(
            f"No extractor named {name!r}. Installed: {sorted(eps) or 'none'}. "
            "Extractors ship in tdg-chrono extras: pip install 'tdg-chrono[llm]' "
            "or 'tdg-chrono[nlp]'."
        )
    try:
        cls = eps[name].load()
        return cls(**kwargs)
    except ImportError as e:
        raise ImportError(
            f"Extractor {name!r} is registered but its dependencies are not "
            f"installed ({e}). Install the matching extra, e.g. "
            f"pip install 'tdg-chrono[llm]' or 'tdg-chrono[nlp]'."
        ) from e
