"""
Temporal Dependency Graph — data structures.

Matches the LegalEditBench / TempChain JSON schema from the research framing doc.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from typing import Literal, Optional

import networkx as nx

SCHEMA_VERSION = "1.0"

TemporalRole = Literal["START", "END", "DURATION", "CONTAINS", "UNKNOWN"]
ConstraintType = Literal["additive", "ordering", "interval", "periodic"]


@dataclass
class TimexSpan:
    """A temporal expression found in text (TIMEX3-like)."""
    text: str                       # raw surface text
    timex_type: str                 # DATE, TIME, DURATION, SET
    value: Optional[str]            # normalized ISO value or duration string
    start_char: int                 # character offset in source text
    end_char: int
    date_parsed: Optional[date] = None
    duration_days: Optional[int] = None  # for DURATION type

    def to_dict(self) -> dict:
        d = asdict(self)
        d["date_parsed"] = self.date_parsed.isoformat() if self.date_parsed else None
        return d


@dataclass
class TemporalFact:
    """A node in the TDG: a temporal expression linked to an entity with a role."""
    id: str                         # e.g. "f1"
    entity: str                     # the entity this fact is about
    role: TemporalRole              # START, END, DURATION, CONTAINS, UNKNOWN
    timex: TimexSpan                # the underlying temporal expression
    sentence: str                   # source sentence
    confidence: float = 1.0
    # Extracted linguistic signals that determined the role
    signal_verb: Optional[str] = None
    signal_prep: Optional[str] = None
    # Coreference: if set, this fact is a duplicate of the named fact id.
    # It is preserved in the output for auditability but excluded from
    # graph construction (edges are built from the canonical fact only).
    is_duplicate_of: Optional[str] = None
    # False when the fact has no resolved value AND no temporal signal in its
    # text (a clause fragment the LLM mis-tagged as temporal). Kept in the graph
    # for auditability; downstream consumers may skip weak facts. NOT dropped —
    # flagging avoids irreversible, regex-gated data loss.
    temporal_content: bool = True

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "entity": self.entity,
            "role": self.role,
            "value": self.timex.value,
            "raw_text": self.timex.text,
            "timex_type": self.timex.timex_type,
            "sentence": self.sentence,
            "confidence": self.confidence,
            "signal_verb": self.signal_verb,
            "signal_prep": self.signal_prep,
            "temporal_content": self.temporal_content,
            "start_char": self.timex.start_char,
            "end_char": self.timex.end_char,
        }
        if self.timex.date_parsed:
            d["date_parsed"] = self.timex.date_parsed.isoformat()
        if self.timex.duration_days is not None:
            d["duration_days"] = self.timex.duration_days
        if self.is_duplicate_of is not None:
            d["is_duplicate_of"] = self.is_duplicate_of
        return d


@dataclass
class TemporalDependency:
    """A typed constraint edge between two TemporalFacts."""
    from_id: str
    to_id: str
    constraint_type: ConstraintType
    constraint_expr: str             # human-readable: "+5y8m", "start < end"
    delta_days: Optional[int] = None
    confidence: float = 1.0
    verified: bool = False           # True if arithmetic checked out
    allen_relation: Optional[str] = None
    # Number of independent passages that stated this relationship.
    # > 1 means the same temporal constraint was corroborated by multiple
    # parts of the document (e.g. both letters in an exchange-of-letters doc).
    corroboration_count: int = 1

    def to_dict(self) -> dict:
        return asdict(self)


def _format_value(value: Optional[str], raw: str, duration_days: Optional[int]) -> str:
    """Human-readable display of a temporal value. Converts ISO durations like P66Y."""
    if not value:
        return raw
    # Convert ISO 8601 duration to readable form
    if value.startswith("P") and duration_days is not None:
        years, rem = divmod(duration_days, 365)
        months, days = divmod(rem, 30)
        parts = []
        if years:
            parts.append(f"{years}y")
        if months:
            parts.append(f"{months}m")
        if days and not years:  # skip leftover days for long durations
            parts.append(f"{days}d")
        return " ".join(parts) if parts else value
    return value


@dataclass
class TemporalDependencyGraph:
    """Full TDG for a document — the core output of the pipeline."""
    document_id: str
    document_type: str               # "historical", "legal", "medical", etc.
    source_text: str
    facts: list[TemporalFact] = field(default_factory=list)
    dependencies: list[TemporalDependency] = field(default_factory=list)
    edit_scenarios: list[dict] = field(default_factory=list)
    # Which matter (case, claimant, file reference) this document belongs to.
    # Optional and never inferred: when two documents carry different matters
    # the linker refuses to connect them, and when it is absent the linker
    # links freely and says so. Inferring matter identity from text statistics
    # was measured and does not work — see PATCH-NOTES.md.
    matter: Optional[str] = None
    # Named people and organisations this document is about. Extracted, not
    # inferred: naming the parties is a reading task a model does well, while
    # deducing which documents share a case from text statistics was measured
    # and does not work. Used to tell matters apart when no matter is declared.
    parties: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {
            "schema_version": SCHEMA_VERSION,
            "document_id": self.document_id,
            "document_type": self.document_type,
            "source_text": self.source_text,
            "facts": [f.to_dict() for f in self.facts],
            "dependencies": [d.to_dict() for d in self.dependencies],
            "edit_scenarios": self.edit_scenarios,
        }
        if self.matter is not None:
            d["matter"] = self.matter
        if self.parties:
            d["parties"] = list(self.parties)
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_networkx(self) -> nx.DiGraph:
        g = nx.DiGraph()
        for f in self.facts:
            g.add_node(f.id, entity=f.entity, role=f.role,
                       value=f.timex.value, raw=f.timex.text)
        for d in self.dependencies:
            g.add_edge(d.from_id, d.to_id,
                       type=d.constraint_type,
                       expr=d.constraint_expr,
                       delta_days=d.delta_days,
                       verified=d.verified)
        return g

    def summary(self) -> str:
        lines = [
            f"TDG: {self.document_id} ({self.document_type})",
            f"  Facts: {len(self.facts)}",
        ]
        for f in self.facts:
            display_val = _format_value(f.timex.value, f.timex.text, f.timex.duration_days)
            lines.append(f"    [{f.id}] {f.entity} / {f.role:10s} = {display_val}"
                         f"  (verb={f.signal_verb}, prep={f.signal_prep})")
        lines.append(f"  Dependencies: {len(self.dependencies)}")
        for d in self.dependencies:
            v = " [verified]" if d.verified else ""
            lines.append(f"    {d.from_id} --[{d.constraint_type}: {d.constraint_expr}]--> {d.to_id}{v}")
        lines.append(f"  Edit scenarios: {len(self.edit_scenarios)}")
        return "\n".join(lines)