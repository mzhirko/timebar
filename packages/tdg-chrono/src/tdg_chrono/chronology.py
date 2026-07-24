"""Chronology core (Phase 1.1) — the only genuinely new logic.

Turns per-document TDGs plus cross-document coreference links into a
single timeline: one row per real-world event, every row carrying its
source quotes, disputes surfaced as rows rather than resolved silently.

Design rules (from RELEASE_TODO / D3, D4):
  - Nothing is dropped. Facts that cannot be placed on the timeline go
    to an ``unplaced`` bucket with a stated reason.
  - Disagreement between coreferent facts renders as ONE row with all
    values and all quotes (status="disputed"). The builder never picks
    a winner.
  - Merging is aggressive but recoverable: ``MergeOverrides`` lets a
    user split a wrong merge or force a missed one, and overrides are
    applied on top of automatic clustering on every rebuild.
  - Derived rows ("three weeks after service") are placed on the
    timeline only when the arithmetic is fully determined, and carry
    their derivation string.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from datetime import date, timedelta
from typing import Iterable, Literal, Optional

from tdg_core.tdg import TemporalDependencyGraph, TemporalFact
from tdg_core.cross_doc import CrossDocLink, CrossDocLinker
from tdg_core.entailment import _offset_from_text

Precision = Literal["day", "month", "year", "relative", "none"]
Status = Literal["agreed", "disputed", "single_source", "derived"]

FactKey = tuple[str, str]  # (document_id, fact_id)

# Confidence floor for accepting an automatic coreference merge.
DEFAULT_MERGE_THRESHOLD = 0.45


# ─── Result dataclasses ───────────────────────────────────────────────────

@dataclass
class SourceRef:
    """One document's statement of an event — the provenance unit."""
    doc_id: str
    fact_id: str
    quote: str                    # source sentence
    value: Optional[str]          # normalised value as stated by this source
    date_parsed: Optional[date]
    start_char: int
    end_char: int
    confidence: float

    def to_dict(self) -> dict:
        d = asdict(self)
        d["date_parsed"] = self.date_parsed.isoformat() if self.date_parsed else None
        return d


@dataclass
class DisputedValue:
    value: str
    date_parsed: Optional[date]
    doc_ids: list[str]

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "date_parsed": self.date_parsed.isoformat() if self.date_parsed else None,
            "doc_ids": self.doc_ids,
        }


@dataclass
class ChronologyEvent:
    event_id: str
    date: Optional[date]          # sort key; earliest disputed value when disputed
    precision: Precision
    label: str
    sources: list[SourceRef]
    status: Status
    disputed_values: list[DisputedValue] = field(default_factory=list)
    confidence: float = 1.0
    derivation: Optional[str] = None   # for status="derived"

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "date": self.date.isoformat() if self.date else None,
            "precision": self.precision,
            "label": self.label,
            "status": self.status,
            "confidence": round(self.confidence, 3),
            "sources": [s.to_dict() for s in self.sources],
            "disputed_values": [v.to_dict() for v in self.disputed_values],
            "derivation": self.derivation,
        }


@dataclass
class UnplacedItem:
    reason: str                   # why it could not be placed
    event: ChronologyEvent        # still a full event: label, sources, quotes

    def to_dict(self) -> dict:
        return {"reason": self.reason, "event": self.event.to_dict()}


@dataclass
class MergeOverrides:
    """User corrections to automatic clustering (D4: recoverable merging).

    force_merge: groups of fact keys that must share a row.
    split:       fact keys pulled out of whatever cluster they landed in,
                 each becoming its own row.
    Splits are applied after force-merges, so a split always wins for
    the specific fact it names.
    """
    force_merge: list[list[FactKey]] = field(default_factory=list)
    split: list[FactKey] = field(default_factory=list)


@dataclass
class Chronology:
    events: list[ChronologyEvent]
    unplaced: list[UnplacedItem]
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "meta": self.meta,
            "events": [e.to_dict() for e in self.events],
            "unplaced": [u.to_dict() for u in self.unplaced],
        }

    @property
    def disputed_count(self) -> int:
        return sum(1 for e in self.events if e.status == "disputed")


# ─── Union-find ───────────────────────────────────────────────────────────

class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[FactKey, FactKey] = {}

    def add(self, k: FactKey) -> None:
        self.parent.setdefault(k, k)

    def find(self, k: FactKey) -> FactKey:
        while self.parent[k] != k:
            self.parent[k] = self.parent[self.parent[k]]
            k = self.parent[k]
        return k

    def union(self, a: FactKey, b: FactKey) -> None:
        self.add(a)
        self.add(b)
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def clusters(self) -> list[list[FactKey]]:
        groups: dict[FactKey, list[FactKey]] = {}
        for k in self.parent:
            groups.setdefault(self.find(k), []).append(k)
        return list(groups.values())


# ─── Precision / labelling helpers ────────────────────────────────────────

_MONTH_RE = re.compile(r"^\d{4}-\d{2}$")
_YEAR_RE = re.compile(r"^\d{4}$")


def _fact_precision(f: TemporalFact) -> Precision:
    if f.timex.date_parsed:
        return "day"
    v = f.timex.value or ""
    if _MONTH_RE.match(v):
        return "month"
    if _YEAR_RE.match(v):
        return "year"
    if v.startswith("P") or f.timex.timex_type == "DURATION":
        return "relative"
    if v:
        return "relative"
    return "none"


_PRECISION_RANK = {"day": 0, "month": 1, "year": 2, "relative": 3, "none": 4}

_ROLE_LABEL = {"START": "start", "END": "end", "DURATION": "duration",
               "CONTAINS": "", "UNKNOWN": ""}


def _label(facts: list[TemporalFact]) -> str:
    best = max(facts, key=lambda f: (f.confidence, len(f.entity)))
    role = _ROLE_LABEL.get(best.role, "")
    ent = best.entity.strip() or best.timex.text
    return f"{ent} ({role})" if role else ent


def _partial_to_date(value: str) -> Optional[date]:
    """YYYY-MM → first of month; YYYY → Jan 1. For sorting only."""
    if _MONTH_RE.match(value):
        y, m = value.split("-")
        return date(int(y), int(m), 1)
    if _YEAR_RE.match(value):
        return date(int(value), 1, 1)
    return None


# ─── Core builder ─────────────────────────────────────────────────────────

def build_chronology(
    tdgs: dict[str, TemporalDependencyGraph],
    links: Optional[Iterable[CrossDocLink]] = None,
    *,
    merge_threshold: float = DEFAULT_MERGE_THRESHOLD,
    overrides: Optional[MergeOverrides] = None,
    derive_undated: bool = True,
    meta: Optional[dict] = None,
) -> Chronology:
    """Build a single chronology from per-document TDGs.

    Args:
        tdgs: {document_id: TDG}
        links: cross-document links. When None, a CrossDocLinker is run
            over the given TDGs (embedder-free Jaccard fallback).
        merge_threshold: minimum link confidence for an automatic merge.
        overrides: user force-merge / split corrections, re-applied on
            top of automatic clustering every run.
        derive_undated: place undated facts via additive dependencies
            from dated ones, marked status="derived".
    """
    if links is None:
        linker = CrossDocLinker()
        for t in tdgs.values():
            linker.add_tdg(t)
        links = linker.find_coreferences() + linker.find_contradictions()
    links = list(links)

    fact_by_key: dict[FactKey, TemporalFact] = {}
    for doc_id, tdg in tdgs.items():
        for f in tdg.facts:
            fact_by_key[(doc_id, f.id)] = f

    uf = _UnionFind()
    for k in fact_by_key:
        uf.add(k)

    # 1. Within-document duplicates (is_duplicate_of) always merge.
    for (doc_id, fid), f in fact_by_key.items():
        if f.is_duplicate_of and (doc_id, f.is_duplicate_of) in fact_by_key:
            uf.union((doc_id, f.is_duplicate_of), (doc_id, fid))

    # 2. Cross-document coreference links above threshold.
    #    Contradiction links also merge — a contradiction IS the same
    #    event told two ways; the disagreement is surfaced on the row.
    for link in links:
        if link.link_type not in ("coreference", "contradiction"):
            continue
        if link.confidence < merge_threshold:
            continue
        a = (link.from_doc, link.from_fact)
        b = (link.to_doc, link.to_fact)
        if a in fact_by_key and b in fact_by_key:
            uf.union(a, b)

    # 3. User overrides, applied on top (D4: recoverable).
    overrides = overrides or MergeOverrides()
    for group in overrides.force_merge:
        keys = [tuple(k) for k in group if tuple(k) in fact_by_key]
        for a, b in zip(keys, keys[1:]):
            uf.union(a, b)

    split_set = {tuple(k) for k in overrides.split}
    clusters: list[list[FactKey]] = []
    for cluster in uf.clusters():
        kept = [k for k in cluster if k not in split_set]
        if kept:
            clusters.append(kept)
        clusters.extend([[k] for k in cluster if k in split_set])

    contradiction_pairs = {
        frozenset([(l.from_doc, l.from_fact), (l.to_doc, l.to_fact)])
        for l in links if l.link_type == "contradiction"
    }

    # ── Cluster → event ──────────────────────────────────────────────
    events: list[ChronologyEvent] = []
    unplaced: list[UnplacedItem] = []
    counter = 0

    def _next_id() -> str:
        nonlocal counter
        counter += 1
        return f"ev{counter:03d}"

    for cluster in clusters:
        facts = [(k, fact_by_key[k]) for k in sorted(cluster)]
        sources = [
            SourceRef(
                doc_id=k[0], fact_id=k[1],
                quote=f.sentence or f.timex.text,
                value=f.timex.value,
                date_parsed=f.timex.date_parsed,
                start_char=f.timex.start_char,
                end_char=f.timex.end_char,
                confidence=f.confidence,
            )
            for k, f in facts
        ]
        only_facts = [f for _, f in facts]
        label = _label(only_facts)
        confidence = min(f.confidence for f in only_facts)
        event_id = _next_id()

        # Non-temporal noise: never dropped, never on the timeline.
        if all(not f.temporal_content for f in only_facts):
            unplaced.append(UnplacedItem(
                reason="no temporal content (extractor flag)",
                event=ChronologyEvent(event_id, None, "none", label,
                                      sources, "single_source",
                                      confidence=confidence),
            ))
            continue

        # Distinct resolved dates across the cluster.
        by_date: dict[date, list[str]] = {}
        for k, f in facts:
            if f.timex.date_parsed:
                by_date.setdefault(f.timex.date_parsed, []).append(k[0])

        has_contradiction = any(
            frozenset([a, b]) in contradiction_pairs
            for i, (a, _) in enumerate(facts)
            for b, _ in facts[i + 1:]
        )

        if len(by_date) >= 2 or (has_contradiction and by_date):
            disputed = [
                DisputedValue(value=d.isoformat(), date_parsed=d,
                              doc_ids=sorted(set(docs)))
                for d, docs in sorted(by_date.items())
            ]
            # also surface unresolved conflicting raw values, if any
            events.append(ChronologyEvent(
                event_id=event_id,
                date=min(by_date),           # sort by earliest claimed date
                precision="day",
                label=label,
                sources=sources,
                status="disputed",
                disputed_values=disputed,
                confidence=confidence,
            ))
            continue

        if len(by_date) == 1:
            d = next(iter(by_date))
            n_docs = len({k[0] for k, _ in [(k, f) for k, f in facts]})
            status: Status = "agreed" if len(sources) > 1 and n_docs > 1 else "single_source"
            events.append(ChronologyEvent(
                event_id=event_id, date=d, precision="day", label=label,
                sources=sources, status=status, confidence=confidence,
            ))
            continue

        # No fully-resolved date: try partial (month/year) placement.
        best = min(only_facts, key=lambda f: _PRECISION_RANK[_fact_precision(f)])
        prec = _fact_precision(best)
        if prec in ("month", "year"):
            events.append(ChronologyEvent(
                event_id=event_id,
                date=_partial_to_date(best.timex.value or ""),
                precision=prec, label=label, sources=sources,
                status="agreed" if len(sources) > 1 else "single_source",
                confidence=confidence,
            ))
            continue

        # Undated. Try derivation, else unplaced.
        derived = _try_derive(event_id, cluster, fact_by_key, tdgs,
                              label, sources, confidence) if derive_undated else None
        if derived:
            events.append(derived)
        else:
            reason = ("relative expression not yet resolved"
                      if prec == "relative" else "no date found")
            unplaced.append(UnplacedItem(
                reason=reason,
                event=ChronologyEvent(event_id, None, prec, label, sources,
                                      "single_source", confidence=confidence),
            ))

    events.sort(key=lambda e: (e.date or date.max, e.label))
    chron = Chronology(events=events, unplaced=unplaced, meta=dict(meta or {}))
    chron.meta.setdefault("documents", sorted(tdgs))
    chron.meta.setdefault("counts", {
        "events": len(events),
        "disputed": chron.disputed_count,
        "unplaced": len(unplaced),
    })
    return chron


def _try_derive(
    event_id: str,
    cluster: list[FactKey],
    fact_by_key: dict[FactKey, TemporalFact],
    tdgs: dict[str, TemporalDependencyGraph],
    label: str,
    sources: list[SourceRef],
    confidence: float,
) -> Optional[ChronologyEvent]:
    """Place an undated cluster via an additive dependency from a dated fact.

    Only when the arithmetic is fully determined: a dated from-fact and
    an offset readable from the dependency (delta_days or an offset
    parsed from constraint_expr). The derivation string carries the
    working, per D3: derivations, not verdicts.
    """
    member_ids = {k for k in cluster}
    for doc_id, tdg in tdgs.items():
        for dep in tdg.dependencies:
            if dep.constraint_type != "additive":
                continue
            to_key = (doc_id, dep.to_id)
            from_key = (doc_id, dep.from_id)
            if to_key not in member_ids or from_key not in fact_by_key:
                continue
            anchor = fact_by_key[from_key]
            if not anchor.timex.date_parsed:
                continue
            offset = _offset_from_text(dep.constraint_expr or "")
            if offset is not None:
                placed = offset.apply(anchor.timex.date_parsed)
                how = f"'{dep.constraint_expr}'"
            elif dep.delta_days is not None:
                placed = anchor.timex.date_parsed + timedelta(days=dep.delta_days)
                how = f"delta_days={dep.delta_days}"
            else:
                continue
            derivation = (
                f"{placed.isoformat()} = {anchor.timex.date_parsed.isoformat()} "
                f"({anchor.entity} [{doc_id}/{anchor.id}]) + {how}; "
                f"stated in: \"{(anchor.sentence or '')[:160]}\""
            )
            return ChronologyEvent(
                event_id=event_id, date=placed, precision="day", label=label,
                sources=sources, status="derived",
                confidence=min(confidence, dep.confidence),
                derivation=derivation,
            )
    return None
