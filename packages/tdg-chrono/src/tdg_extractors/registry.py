"""Entry-point targets wrapping the pipelines in the Extractor protocol.

Import of openai/spacy happens only inside these constructors, so
`pip install tdg-chrono` (no extras) never touches them.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from tdg_core.tdg import TemporalDependencyGraph


class LLMExtractor:
    """LLM one-shot extractor (OpenAI-compatible endpoint; Ollama works)."""

    name = "llm"

    def __init__(self, **kwargs):
        from tdg_extractors.llm_pipeline import LLMPipeline  # imports openai
        self._pipe = LLMPipeline(**kwargs)

    def extract(self, text: str, *, document_id: str,
                document_type: str = "unknown",
                reference_date: Optional[date] = None) -> TemporalDependencyGraph:
        return self._pipe.process(text, document_id=document_id,
                                  document_type=document_type)


class HeidelTimeExtractor:
    """HeidelTime + spaCy extractor. Offline, no API key, no GPU."""

    name = "heideltime"

    def __init__(self, **kwargs):
        from tdg_extractors.pipeline import TDGPipeline  # imports spacy
        self._pipe = TDGPipeline(**kwargs)

    def extract(self, text: str, *, document_id: str,
                document_type: str = "unknown",
                reference_date: Optional[date] = None) -> TemporalDependencyGraph:
        return self._pipe.process(text, document_id=document_id,
                                  document_type=document_type)
