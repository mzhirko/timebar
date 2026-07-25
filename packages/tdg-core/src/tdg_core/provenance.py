"""Can a fact's quote actually be found in the document it claims to be from?

The tool's central promise is that every date on the timeline links to the
sentence it came from, so a reader can check it. That promise is only worth
anything if somebody checks that the sentence is really there.

Nothing did. A shipped example carried a document with no text at all, its
character offsets set to zero, and a quote that existed only as a string in
a JSON file. It was counted as independent corroboration and raised another
document's date to "agreed" — the label that tells a reader to stop looking.

This module answers the question for each fact. It does not decide what to
do about the answer; the chronology decides that, and says so in the output.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tdg_core.tdg import TemporalDependencyGraph, TemporalFact

# Shorter quotes match too loosely to be evidence of anything.
_MIN_QUOTE_CHARS = 12

# A body shorter than the shortest usable quote cannot corroborate anything,
# which is how a placeholder like "…" is recognised without guessing at a
# length that would also reject a genuinely short document.
_MIN_SOURCE_CHARS = _MIN_QUOTE_CHARS


def _flat(text: str) -> str:
    """Whitespace-insensitive form, so a hard-wrapped quote still matches."""
    return " ".join((text or "").split())


def quote_candidates(fact: TemporalFact) -> list[str]:
    """Every text on a fact that might be its quote.

    Extractors disagree about which field holds the source sentence, so both
    are considered rather than trusting a field name.
    """
    return [t for t in (fact.sentence, fact.timex.text) if t and t.strip()]


def fact_is_supported(fact: TemporalFact, source_text: str) -> bool:
    """True when one of the fact's quotes appears in the document text.

    No length heuristic on the document: whether the quote is there is the
    question, and answering it directly avoids rejecting a short document
    that genuinely contains what it claims.
    """
    haystack = _flat(source_text)
    for candidate in quote_candidates(fact):
        needle = _flat(candidate)
        if len(needle) >= _MIN_QUOTE_CHARS and needle in haystack:
            return True
    return False


@dataclass
class DocumentProvenance:
    """What could and could not be checked in one document."""

    doc_id: str
    has_source_text: bool
    total: int
    supported: set = field(default_factory=set)
    unsupported: set = field(default_factory=set)

    @property
    def fully_supported(self) -> bool:
        return self.total > 0 and not self.unsupported

    @property
    def summary(self) -> str:
        if not self.has_source_text:
            return (f"{self.doc_id}: no source text shipped, so none of its "
                    f"{self.total} fact(s) can be checked")
        if self.unsupported:
            return (f"{self.doc_id}: {len(self.unsupported)} of {self.total} "
                    "quote(s) not found in its own source text")
        return f"{self.doc_id}: all {self.total} quote(s) check out"


def check_document(tdg: TemporalDependencyGraph) -> DocumentProvenance:
    """Verify every fact in one document against that document's own text."""
    source = tdg.source_text or ""
    report = DocumentProvenance(
        doc_id=tdg.document_id,
        has_source_text=len(_flat(source)) >= _MIN_SOURCE_CHARS,
        total=len(tdg.facts))
    for fact in tdg.facts:
        if fact_is_supported(fact, source):
            report.supported.add(fact.id)
        else:
            report.unsupported.add(fact.id)
    return report


def check_bundle(tdgs: dict) -> dict:
    """Verify a whole bundle. Returns {document_id: DocumentProvenance}."""
    return {doc_id: check_document(tdg) for doc_id, tdg in tdgs.items()}


def unverified_keys(reports: dict) -> set:
    """(document_id, fact_id) pairs whose quote could not be confirmed."""
    return {(doc_id, fact_id)
            for doc_id, rep in reports.items()
            for fact_id in rep.unsupported}


def format_report(reports: dict) -> list[str]:
    """Printable lines. Empty when every quote in the bundle checks out."""
    problems = [r for r in reports.values() if r.unsupported]
    if not problems:
        return []
    total = sum(len(r.unsupported) for r in problems)
    lines = [f"\nprovenance: {total} fact(s) could not be checked against the "
             "text of the document they came from."]
    lines.append("  A quote that cannot be located is still shown, but it does "
                 "not count as corroboration for any other document.")
    for rep in sorted(problems, key=lambda r: r.doc_id):
        lines.append(f"  - {rep.summary}")
    return lines
