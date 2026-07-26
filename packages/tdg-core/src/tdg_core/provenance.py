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


# ─── Relations: is the period the edge claims actually in the document? ──
#
# An extractor can invent a dependency. Two of the three relations in a test
# bundle were the model computing the gap between two dates it had seen and
# writing it down as though the document stated a rule:
#
#   hearing -> employment termination, "+45 days"   (the gap; 45 appears nowhere)
#   hearing -> employment termination, "+53 days"   (likewise)
#
# A third named a real period from the text but hung it off the wrong anchor:
# "within 28 days of service" attached to the start of employment, six years
# earlier. Nothing detected any of them.
#
# Two deterministic questions catch all three. Is the period stated in the
# document at all? And is the anchor named in the sentence that states it?

import re as _re

_OFFSET_RE = _re.compile(
    r"(\d+)\s*(day|days|week|weeks|month|months|year|years|d|w|m|y)\b",
    _re.IGNORECASE)

_UNIT_FAMILY = {"d": "day", "day": "day", "days": "day",
                "w": "week", "week": "week", "weeks": "week",
                "m": "month", "month": "month", "months": "month",
                "y": "year", "year": "year", "years": "year"}


def _number_words() -> dict:
    """English numerals, from data. A statute spells out what an edge digitises."""
    import json
    from importlib import resources
    try:
        with resources.files("tdg_core.data").joinpath(
                "number_words.en.json").open() as fh:
            return json.load(fh)
    except (FileNotFoundError, ModuleNotFoundError):  # pragma: no cover
        return {"numbers": {}, "units": {}}


_WORDS = _number_words()


@dataclass
class RelationCheck:
    """Whether one dependency's arithmetic is actually in the document."""

    doc_id: str
    from_id: str
    to_id: str
    constraint_type: str
    anchor: str
    target: str
    offset: str
    verdict: str          # "supported" | "offset-not-in-source" | "anchor-not-in-clause"
    clause: str = ""

    @property
    def supported(self) -> bool:
        return self.verdict == "supported"

    @property
    def summary(self) -> str:
        if self.verdict == "offset-not-in-source":
            return (f"{self.doc_id}: '{self.target}' is {self.offset} after "
                    f"'{self.anchor}' — the document never says so.")
        if self.verdict == "anchor-not-in-clause":
            return (f"{self.doc_id}: '{self.target}' is {self.offset} after "
                    f"'{self.anchor}' — but the document counts {self.offset} "
                    f"from something else: \"{self.clause[:70]}\"")
        return f"{self.doc_id}: '{self.anchor}' -> '{self.target}' ({self.offset}) checks out"


def _stated_offsets(text: str) -> set:
    """Every (n, unit) period the document states, digits or words alike."""
    found = set()
    flat = _flat(text).lower()
    for n, unit in _OFFSET_RE.findall(flat):
        found.add((int(n), _UNIT_FAMILY[unit.lower()]))
    for word, value in _WORDS.get("numbers", {}).items():
        for family, spellings in _WORDS.get("units", {}).items():
            for spelling in spellings:
                if f"{word} {spelling}" in flat:
                    found.add((value, family))
    return found


def _clause_stating(text: str, n: int, unit: str) -> str:
    """The sentence in which the document states this period."""
    words = {w for w, v in _WORDS.get("numbers", {}).items() if v == n}
    spellings = _WORDS.get("units", {}).get(unit, [unit])
    for sentence in _re.split(r"(?<=[.;:])\s+", _flat(text)):
        low = sentence.lower()
        if any(f"{form} {sp}" in low
               for sp in spellings for form in ({str(n)} | words)):
            return sentence
    return ""


def check_relation(tdg: TemporalDependencyGraph, dep) -> RelationCheck | None:
    """Verify one dependency's period against the document's own words.

    Returns None when the edge asserts no period, which is the normal case
    for a pure ordering constraint and nothing to check.
    """
    from tdg_core.linking import tokenise

    facts = {f.id: f for f in tdg.facts}
    anchor, target = facts.get(dep.from_id), facts.get(dep.to_id)
    if anchor is None or target is None:
        return None

    text = dep.constraint_expr or ""
    match = _OFFSET_RE.search(text)
    if match:
        n, unit = int(match.group(1)), _UNIT_FAMILY[match.group(2).lower()]
    elif dep.delta_days:
        n, unit = int(dep.delta_days), "day"
    else:
        return None

    common = dict(doc_id=tdg.document_id, from_id=dep.from_id, to_id=dep.to_id,
                  constraint_type=dep.constraint_type,
                  anchor=anchor.entity, target=target.entity,
                  offset=f"{n} {unit}{'s' if n != 1 else ''}")

    if (n, unit) not in _stated_offsets(tdg.source_text or ""):
        return RelationCheck(verdict="offset-not-in-source", **common)

    clause = _clause_stating(tdg.source_text or "", n, unit)
    anchor_words = {t for t in tokenise(anchor.entity) if len(t) > 3}
    if anchor_words and not (anchor_words & set(tokenise(clause))):
        return RelationCheck(verdict="anchor-not-in-clause", clause=clause, **common)
    return RelationCheck(verdict="supported", clause=clause, **common)


def check_relations(tdg: TemporalDependencyGraph) -> list:
    """Every checkable dependency in one document."""
    out = []
    for dep in tdg.dependencies:
        result = check_relation(tdg, dep)
        if result is not None:
            out.append(result)
    return out


def check_bundle_relations(tdgs: dict) -> dict:
    """Verify every document's dependencies. {document_id: [RelationCheck]}."""
    return {doc_id: check_relations(tdg) for doc_id, tdg in tdgs.items()}


def unsupported_relations(checks: dict) -> list:
    """Every dependency whose arithmetic the document does not support."""
    return [c for results in checks.values() for c in results if not c.supported]


def format_relation_report(checks: dict) -> list:
    """Printable lines. Empty when every stated period checks out."""
    bad = unsupported_relations(checks)
    if not bad:
        return []
    lines = [f"\nrelation audit: {len(bad)} dependenc(ies) assert a period the "
             "document does not support."]
    lines.append("  These are not used for arithmetic. A date calculated from "
                 "an invented period would look exactly like a real one.")
    for c in sorted(bad, key=lambda c: (c.doc_id, c.from_id)):
        lines.append(f"  - {c.summary}")
    return lines


# ─── The hash the schema promises ────────────────────────────────────────

def source_hash(text: str) -> str:
    """SHA-256 of a document's text, as the schema defines it."""
    import hashlib
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def check_source_hash(tdg: TemporalDependencyGraph) -> tuple[str, str]:
    """Compare a declared source hash with the text, if both are present.

    The schema tells producers that a bundle with sensitive content may ship
    offsets and a hash instead of the text. Nothing read that hash, so the
    substitute it offered was never actually a substitute: such a bundle had
    no provenance path at all. Returns (verdict, detail).

    Verdicts: "match", "mismatch", "text-only", "hash-only", "neither".
    """
    declared = (tdg.source_text_sha256 or "").strip().lower()
    has_text = len(_flat(tdg.source_text or "")) >= _MIN_SOURCE_CHARS

    if declared and has_text:
        actual = source_hash(tdg.source_text)
        if actual == declared:
            return "match", "the shipped text matches its declared hash"
        return "mismatch", (f"declared {declared[:12]}... but the shipped text "
                            f"hashes to {actual[:12]}...; one of them is stale")
    if declared:
        return "hash-only", ("text externalised; supply it to check the quotes "
                             f"against hash {declared[:12]}...")
    if has_text:
        return "text-only", "text shipped, no hash declared to pin it to"
    return "neither", ("no source text and no hash, so nothing in this "
                       "document can be verified at all")


def trusted_dependencies(tdg: TemporalDependencyGraph) -> list:
    """The dependencies whose arithmetic the document actually supports.

    Every consumer that walks additive edges must filter through this, not
    just the chronology. Guarding one path and leaving the rest open was a
    real hole: a fabricated "45 days" edge was correctly rejected by the
    timeline while What-if still used it to move a termination date by a
    month, `stale` still reported a fact as invalidated by it, and the
    viewer still advertised the dependency as real.

    A rejected edge is dropped rather than downgraded. An edge asserting no
    period at all is kept, since there is no arithmetic to be wrong about.
    """
    keep = []
    for dep in tdg.dependencies:
        result = check_relation(tdg, dep)
        if result is None or result.supported:
            keep.append(dep)
    return keep
