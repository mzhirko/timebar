"""
Fallback pipeline components — no Java or spaCy required.

Use these when HeidelTime or spaCy are unavailable, or for quick tests.

Usage:
    from tdg_extractors.fallbacks import FallbackPipeline

    pipe = FallbackPipeline()
    tdg = pipe.process(text, document_id="test", document_type="legal")
    print(tdg.summary())

Components:
    RegexExtractor        — regex + dateutil date extraction
    PatternRoleClassifier — verb proximity role classification
    HeuristicEntityLinker — capitalization-based entity linking
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from tdg_core.tdg import TemporalDependencyGraph, TemporalFact
from tdg_extractors.timex_extractor import RegexExtractor
from tdg_extractors.role_classifier import PatternRoleClassifier
from tdg_extractors.entity_linker import HeuristicEntityLinker
from tdg_core.graph_builder import GraphBuilder
from tdg_extractors.scenario_generator import generate_edit_scenarios
from tdg_extractors.pipeline import _get_sentence


class FallbackPipeline:
    """
    Pure Python pipeline — no Java or spaCy required.
    Useful for quick tests and CI environments without NLP dependencies.
    Lower accuracy than the main TDGPipeline.
    """

    def __init__(self):
        self.extractor = RegexExtractor()
        self.role_classifier = PatternRoleClassifier()
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
        linker = HeuristicEntityLinker(document_entity=document_entity)

        timex_spans = self.extractor.extract(text, reference_date)
        if not timex_spans:
            return TemporalDependencyGraph(
                document_id=document_id,
                document_type=document_type,
                source_text=text,
            )

        facts: list[TemporalFact] = []
        seen: set[tuple[str, str, str]] = set()

        for span in timex_spans:
            signal = self.role_classifier.classify(text, span.start_char, span.end_char)

            if span.timex_type == "DURATION":
                signal.role = "DURATION"
                signal.confidence = max(signal.confidence, 0.85)

            entity_name = linker.link(text, span.start_char, span.end_char)

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
                sentence=_get_sentence(text, span.start_char),
                confidence=signal.confidence,
                signal_verb=signal.verb,
                signal_prep=signal.prep,
            ))

        tdg = self.graph_builder.build(
            facts=facts,
            document_id=document_id,
            document_type=document_type,
            source_text=text,
        )

        if generate_scenarios:
            tdg.edit_scenarios = generate_edit_scenarios(tdg)

        return tdg
