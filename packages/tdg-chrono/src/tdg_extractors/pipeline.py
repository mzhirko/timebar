"""
Main TDG extraction pipeline.

Orchestrates: HeidelTime extraction → spaCy role classification
              → spaCy entity linking → graph building → scenarios

Usage:
    from tdg_extractors.pipeline import TDGPipeline

    pipe = TDGPipeline()
    tdg = pipe.process(
        text="The war began on September 1, 1939 and ended on May 8, 1945.",
        document_id="ww2",
        document_type="historical",
    )
    print(tdg.summary())

For fallback (no Java/spaCy), see tdg_extractors/fallbacks.py.
"""

from __future__ import annotations

import spacy

from datetime import date
from typing import Optional

from tdg_core.tdg import TemporalDependencyGraph, TemporalFact
from tdg_extractors.timex_extractor import HeidelTimeExtractor
from tdg_extractors.role_classifier import SpacyRoleClassifier
from tdg_extractors.entity_linker import SpacyEntityLinker
from tdg_core.graph_builder import GraphBuilder
from tdg_extractors.scenario_generator import generate_edit_scenarios


# Document type → HeidelTime document_type
_DOCTYPE_TO_HEIDEL = {
    "historical":   "narrative",
    "biographical": "narrative",
    "legal":        "news",
    "corporate":    "news",
    "medical":      "scientific",
    "unknown":      "news",
}


class TDGPipeline:
    """
    Full text → TDG pipeline using HeidelTime + spaCy.

    The spaCy model is loaded once and shared between the role classifier
    and entity linker to avoid loading it twice.
    """

    def __init__(self, heideltime_path: Optional[str] = None):
        """
        Args:
            heideltime_path: Optional path to HeidelTime installation.
                             py-heideltime auto-detects if None.
        """
        self.heideltime_path = heideltime_path
        self.nlp = spacy.load("en_core_web_trf")
        self.role_classifier = SpacyRoleClassifier(nlp=self.nlp)
        self.entity_linker = SpacyEntityLinker(nlp=self.nlp)
        self.graph_builder = GraphBuilder()

    def process(
        self,
        text: str,
        document_id: str = "doc_001",
        document_type: str = "unknown",
        document_entity: Optional[str] = None,
        reference_date: Optional[date] = None,
        generate_scenarios: bool = True,
    ) -> TemporalDependencyGraph:
        """
        Full pipeline: text → TemporalDependencyGraph.

        Args:
            text: Source text to process
            document_id: Identifier for the document
            document_type: "historical", "legal", "medical", "corporate",
                           "biographical", or "unknown"
            document_entity: If set, all timex are linked to this entity.
                             Useful for single-entity documents.
            reference_date: Reference date for relative expressions.
                            Defaults to today in HeidelTime.
            generate_scenarios: Whether to auto-generate edit scenarios.
        """
        extractor = HeidelTimeExtractor(
            document_type=_DOCTYPE_TO_HEIDEL.get(document_type, "news"),
            heideltime_path=self.heideltime_path,
        )

        # Override entity linker if document_entity is provided
        linker = self.entity_linker
        if document_entity:
            linker = SpacyEntityLinker(
                document_entity=document_entity,
                nlp=self.nlp,
            )

        # --- Step 1: Extract temporal expressions ---
        timex_spans = extractor.extract(text, reference_date)
        if not timex_spans:
            return TemporalDependencyGraph(
                document_id=document_id,
                document_type=document_type,
                source_text=text,
            )

        # --- Step 2: Classify role + link entity for each timex ---
        facts: list[TemporalFact] = []
        seen: set[tuple[str, str, str]] = set()

        # Pre-parse the full text once so sentence boundaries are available
        # for both role classification and sentence extraction
        doc = self.nlp(text)
        # Cache the doc in classifier and linker so they don't re-parse
        self.role_classifier._doc_cache[text] = doc
        self.entity_linker._doc_cache[text] = doc

        for span in timex_spans:
            signal = self.role_classifier.classify(text, span.start_char, span.end_char)

            # DURATION type always wins over verb-based classification
            if span.timex_type == "DURATION":
                signal.role = "DURATION"
                signal.confidence = max(signal.confidence, 0.85)
                signal.reason = "timex_type=DURATION"

            entity_name = linker.link(text, span.start_char, span.end_char)

            # Deduplicate: same entity + role + value = same fact
            val = span.value or span.text
            dedup_key = (entity_name, signal.role, val)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            facts.append(TemporalFact(
                id=f"f{len(facts) + 1}",
                entity=entity_name,
                role=signal.role,
                timex=span,
                sentence=_get_sentence(text, span.start_char, doc),
                confidence=signal.confidence,
                signal_verb=signal.verb,
                signal_prep=signal.prep,
            ))

        # --- Step 3: Build dependency graph ---
        tdg = self.graph_builder.build(
            facts=facts,
            document_id=document_id,
            document_type=document_type,
            source_text=text,
        )

        # --- Step 4: Generate edit scenarios ---
        if generate_scenarios:
            tdg.edit_scenarios = generate_edit_scenarios(tdg)

        return tdg

    def process_batch(
        self,
        texts: list[str],
        document_ids: Optional[list[str]] = None,
        document_types: Optional[list[str]] = None,
    ) -> list[TemporalDependencyGraph]:
        """
        Process multiple texts. document_types defaults to "unknown" for all.
        """
        if document_ids is None:
            document_ids = [f"doc_{i:04d}" for i in range(len(texts))]
        if document_types is None:
            document_types = ["unknown"] * len(texts)

        return [
            self.process(text, doc_id, doc_type)
            for text, doc_id, doc_type in zip(texts, document_ids, document_types)
        ]


def _get_sentence(text: str, char_offset: int, doc=None) -> str:
    """
    Extract the sentence containing the given character offset.
    Uses spaCy sentence segmentation if a doc is provided,
    otherwise falls back to punctuation heuristic.
    """
    if doc is not None:
        for sent in doc.sents:
            if sent.start_char <= char_offset < sent.end_char:
                return sent.text.strip()

    # Fallback: naive punctuation heuristic
    start = text.rfind(".", 0, char_offset)
    start = start + 1 if start >= 0 else 0
    end = text.find(".", char_offset)
    end = end + 1 if end >= 0 else len(text)
    return text[start:end].strip()
