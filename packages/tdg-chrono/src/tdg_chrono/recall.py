"""Deterministic recall audit over a probabilistic extractor.

The LLM extractor is the only non-deterministic component in the pipeline,
and its failure mode is silence: a date sits in the document, no fact is
emitted for it, and the chronology looks complete. Nothing downstream can
notice, because everything downstream only ever sees the facts.

This module closes that hole with the same trick the project uses
elsewhere — quarantine the model, then check it with code. A pure-regex
date finder sweeps the *raw* document text and every date it surfaces is
matched against the extracted facts. Whatever is left over is reported as
an unextracted date, quoted in context.

The sweep runs on raw text rather than cleaned text on purpose, so it also
catches dates destroyed in preprocessing rather than missed by the model.
It covers the whole pipeline, not just its last stage.

This is a recall floor, not a recall measurement: the regex finds explicit
date expressions, so relative references ("three weeks later") are outside
its reach and absence of warnings is not proof of completeness.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from tdg_core.tdg import TemporalDependencyGraph


@dataclass
class MissedDate:
    """A date present in the document that no extracted fact accounts for."""

    text: str                    # surface form, e.g. "12 July 2025"
    value: Optional[str]         # normalized ISO value where resolvable
    sentence: str                # containing sentence, for the audit trail

    def __str__(self) -> str:
        shown = f"{self.text} ({self.value})" if self.value else self.text
        return f"{shown} — {self.sentence}"


def _sentence_around(text: str, start: int, end: int) -> str:
    """Return the sentence containing the span, for quoting in the report."""
    left = max(text.rfind(". ", 0, start), text.rfind("\n", 0, start))
    right_candidates = [i for i in (text.find(". ", end), text.find("\n", end))
                        if i != -1]
    right = min(right_candidates) + 1 if right_candidates else len(text)
    return re.sub(r"\s+", " ", text[left + 1:right]).strip()


def _fact_values(tdg: TemporalDependencyGraph) -> set[str]:
    """Every temporal value the extraction accounted for, normalized."""
    values: set[str] = set()
    for fact in tdg.facts:
        timex = fact.timex
        if timex.value:
            values.add(str(timex.value).strip().lower())
        if timex.date_parsed:
            values.add(timex.date_parsed.isoformat())
        if timex.text:
            values.add(timex.text.strip().lower())
    return values


def audit_recall(raw_text: str, tdg: TemporalDependencyGraph,
                 *, include_durations: bool = False) -> list[MissedDate]:
    """Return dates found in raw_text that no fact in tdg accounts for.

    Durations ("within 28 days") are excluded by default: they are relative
    periods that the graph legitimately represents as an edge rather than a
    dated fact, so reporting them would be noise.
    """
    from tdg_extractors.timex_extractor import RegexExtractor

    accounted = _fact_values(tdg)
    missed: list[MissedDate] = []
    seen: set[str] = set()

    for span in RegexExtractor().extract(raw_text, None):
        if span.timex_type == "DURATION" and not include_durations:
            continue
        # A bare year is usually part of a fuller date already captured.
        if span.date_parsed is None and span.value is None:
            continue

        candidates = {str(span.value).strip().lower() if span.value else "",
                      span.text.strip().lower()}
        if span.date_parsed:
            candidates.add(span.date_parsed.isoformat())
        if candidates & accounted:
            continue

        key = span.value or span.text
        if key in seen:
            continue
        seen.add(key)

        missed.append(MissedDate(
            text=span.text,
            value=span.value,
            sentence=_sentence_around(raw_text, span.start_char, span.end_char),
        ))

    return missed


def format_report(per_document: dict[str, list[MissedDate]]) -> list[str]:
    """Render audit results as printable lines. Empty when nothing was missed."""
    lines: list[str] = []
    total = sum(len(v) for v in per_document.values())
    if not total:
        return lines
    lines.append(
        f"\nrecall audit: {total} date(s) present in the source text but not "
        f"represented by any extracted fact.")
    lines.append(
        "  These are gaps in extraction, not errors in the timeline. Check "
        "them against the source before relying on the chronology.")
    for doc_id, misses in sorted(per_document.items()):
        if not misses:
            continue
        lines.append(f"  {doc_id}:")
        for m in misses:
            lines.append(f"    - {m}")
    return lines
