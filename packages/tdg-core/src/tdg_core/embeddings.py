"""
Embedding-based similarity for entity matching.

Provides semantic similarity between entity names using nomic-embed-text
via Ollama's OpenAI-compatible /v1/embeddings endpoint.

Used by:
  - cross_doc.py: entity resolution across documents
  - align.py: TDG-Catala variable alignment

Falls back to token-based Jaccard similarity when embeddings are unavailable
(Ollama not running, model not loaded, etc).

Usage:
    from tdg_core.embeddings import EmbeddingSimilarity

    sim = EmbeddingSimilarity(base_url="http://localhost:11434/v1")
    score = sim.similarity("termination", "denunciation")
    score = sim.similarity("agreement", "cat food")
"""

from __future__ import annotations

import math
import re
from typing import Optional


# ─── Cosine similarity (no dependencies) ─────────────────────────────────

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ─── Article prefix stripping (shared with graph_builder) ────────────────

_ARTICLE_PREFIX = re.compile(
    r"^(Art\.?\s*\d+[\.\d]*\s*|"
    r"[Ss]ection\s*\d+[\(\)\.\d\w]*\s*|"
    r"§\s*\d+[\(\)\.\d\w]*\s*|"
    r"Article\s*\d+[\.\d]*\s*)",
    re.IGNORECASE,
)

# Jurisdiction boilerplate to strip from entity names, e.g. the statute title
# trailing a concept ("effective date of termination of the Employment Rights
# Act 1996"). Empty by default: the engine ships with no jurisdiction
# vocabulary, matching the rule stated in entailment.py — legal vocabulary is
# loaded from data, never written here. Rule packs declare their own via
# "entity_boilerplate" in aliases.json.
_ENTITY_BOILERPLATE: list[re.Pattern] = []
_BOILERPLATE_SOURCES: list[str] = []


def register_entity_boilerplate(patterns, *, source: str = "<dict>") -> None:
    """Add boilerplate patterns stripped from entity names before comparison.

    Patterns are regular expressions, matched case-insensitively. They are
    additive so several packs can contribute, and every registration records
    its source so a surprising normalisation can be traced back to the file
    that asked for it.
    """
    for p in patterns:
        try:
            _ENTITY_BOILERPLATE.append(re.compile(p, re.IGNORECASE))
        except re.error as e:
            raise ValueError(
                f"invalid entity_boilerplate pattern {p!r} from {source}: {e}"
            ) from e
    if patterns:
        _BOILERPLATE_SOURCES.append(source)


def clear_entity_boilerplate() -> None:
    """Drop all registered boilerplate (used when a pack replaces defaults)."""
    _ENTITY_BOILERPLATE.clear()
    _BOILERPLATE_SOURCES.clear()


def boilerplate_sources() -> list[str]:
    """Where the active boilerplate patterns came from, for the audit trail."""
    return list(_BOILERPLATE_SOURCES)


def normalise_entity(entity: str) -> str:
    """Strip article prefix and registered boilerplate, lowercase for comparison."""
    stripped = _ARTICLE_PREFIX.sub("", entity).strip()
    for pattern in _ENTITY_BOILERPLATE:
        stripped = pattern.sub(" ", stripped).strip()
    stripped = re.sub(r"\s{2,}", " ", stripped)
    return stripped.lower() if stripped else entity.lower()


# ─── Token Jaccard fallback ──────────────────────────────────────────────

_STOP = {"the", "of", "a", "an", "in", "on", "to", "for", "and", "or"}


def _token_jaccard(a: str, b: str) -> float:
    """Token-based Jaccard similarity (fallback when embeddings unavailable)."""
    tokens_a = set(a.lower().split()) - _STOP
    tokens_b = set(b.lower().split()) - _STOP
    if not tokens_a or not tokens_b:
        return 0.0
    overlap = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return overlap / union if union else 0.0


# ─── Main class ──────────────────────────────────────────────────────────

class EmbeddingSimilarity:
    """Compute semantic similarity between entity names using embeddings.

    Caches embeddings per normalised string to avoid redundant API calls.
    Falls back to token Jaccard when the embedding service is unavailable.

    Args:
        base_url: Ollama OpenAI-compatible base URL (e.g. "http://localhost:11434/v1")
        model: embedding model name (default: "nomic-embed-text")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "nomic-embed-text",
    ):
        self.model = model
        self.base_url = base_url
        self._cache: dict[str, list[float]] = {}
        self._client = None
        self._available: Optional[bool] = None  # None = not tested yet

    def _get_client(self):
        """Lazy-init the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key="ollama",
                    base_url=self.base_url,
                )
            except ImportError:
                self._available = False
                return None
        return self._client

    def _embed(self, text: str) -> Optional[list[float]]:
        """Get embedding for a single text string. Returns None on failure."""
        normalised = normalise_entity(text)
        if normalised in self._cache:
            return self._cache[normalised]

        client = self._get_client()
        if client is None:
            return None

        try:
            response = client.embeddings.create(
                model=self.model,
                input=normalised,
            )
            vec = response.data[0].embedding
            self._cache[normalised] = vec
            self._available = True
            return vec
        except Exception:
            # Connection refused, model not loaded, timeout, etc.
            if self._available is None:
                self._available = False
            return None

    def _embed_batch(self, texts: list[str]) -> dict[str, Optional[list[float]]]:
        """Embed multiple texts, using cache where possible.

        Returns dict mapping normalised text → embedding vector (or None).
        """
        results = {}
        uncached = []
        for text in texts:
            normalised = normalise_entity(text)
            if normalised in self._cache:
                results[normalised] = self._cache[normalised]
            else:
                uncached.append(normalised)

        if not uncached:
            return results

        client = self._get_client()
        if client is None:
            for n in uncached:
                results[n] = None
            return results

        try:
            response = client.embeddings.create(
                model=self.model,
                input=uncached,
            )
            for item in response.data:
                normalised = uncached[item.index]
                self._cache[normalised] = item.embedding
                results[normalised] = item.embedding
            self._available = True
        except Exception:
            if self._available is None:
                self._available = False
            for n in uncached:
                results[n] = None

        return results

    def similarity(self, a: str, b: str) -> float:
        """Compute similarity between two entity names.

        Uses embedding cosine similarity when available, falls back to
        token Jaccard when embeddings are unavailable.

        Returns 0.0-1.0.
        """
        na = normalise_entity(a)
        nb = normalise_entity(b)

        # Exact match after normalisation
        if na == nb:
            return 1.0

        # Try embeddings
        vec_a = self._embed(a)
        vec_b = self._embed(b)

        if vec_a is not None and vec_b is not None:
            return _cosine_similarity(vec_a, vec_b)

        # Fallback: token Jaccard
        return _token_jaccard(na, nb)

    @property
    def is_available(self) -> Optional[bool]:
        """Whether the embedding service is available.

        None = not tested yet, True = working, False = unavailable.
        """
        return self._available

    def cache_size(self) -> int:
        """Number of cached embeddings."""
        return len(self._cache)

    def preload(self, entities: list[str]) -> int:
        """Pre-cache embeddings for a list of entity names.

        Returns number of successfully embedded entities.
        """
        results = self._embed_batch(entities)
        return sum(1 for v in results.values() if v is not None)