"""Capability commands (Phase 1.7): interval, contradictions, whatif, deadline.

All four are thin surfaces over logic that already exists in tdg-core;
each answers a question a practitioner actually asks, with quotes and a
derivation attached. All operate on a folder of TDG JSON (run
``tdg-chrono build`` with an extractor first, or author TDGs directly).
"""

from __future__ import annotations

import copy
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

from tdg_core.allen_classifier import classify_allen
from tdg_core.cross_doc import CrossDocLinker
from tdg_core.entailment import _offset_from_text
from tdg_core.tdg import TemporalDependencyGraph, TemporalFact

FactKey = tuple[str, str]


def _find(tdgs: dict[str, TemporalDependencyGraph], key: FactKey) -> TemporalFact:
    doc, fid = key
    if doc not in tdgs:
        raise KeyError(f"no document '{doc}' (have: {sorted(tdgs)})")
    for f in tdgs[doc].facts:
        if f.id == fid:
            return f
    available = ", ".join(f"{f.id} ({f.entity})" for f in tdgs[doc].facts) or "none"
    raise KeyError(f"no fact '{fid}' in document '{doc}'. "
                   f"That document has: {available}")


def _interval_of(tdg_doc: TemporalDependencyGraph, entity_query: str
                 ) -> tuple[Optional[date], Optional[date], list[TemporalFact]]:
    """Resolve an entity's [start, end] from its START/END facts.

    Matches facts whose entity contains the query (case-insensitive).
    A missing END means the interval is open-ended (ongoing).
    """
    q = entity_query.lower()
    hits = [f for f in tdg_doc.facts
            if q in f.entity.lower() and f.timex.date_parsed]
    starts = [f for f in hits if f.role == "START"]
    ends = [f for f in hits if f.role == "END"]
    start = min((f.timex.date_parsed for f in starts), default=None)
    end = max((f.timex.date_parsed for f in ends), default=None)
    return start, end, starts + ends


# ─── interval ────────────────────────────────────────────────────────────

def interval_between(tdgs, key_a: FactKey, key_b: FactKey) -> dict:
    """Allen relation between two facts (each a point or an interval)."""
    fa, fb = _find(tdgs, key_a), _find(tdgs, key_b)
    da, db = fa.timex.date_parsed, fb.timex.date_parsed
    rel = classify_allen(da, da, db, db)
    return {
        "relation": rel,
        "a": {"key": list(key_a), "entity": fa.entity,
              "date": da.isoformat() if da else None, "quote": fa.sentence},
        "b": {"key": list(key_b), "entity": fb.entity,
              "date": db.isoformat() if db else None, "quote": fb.sentence},
        "derivation": (f"{fa.entity} [{da}] {rel} {fb.entity} [{db}]"
                       if da and db else
                       "UNKNOWN: one of the facts has no resolved date"),
    }


def interval_contains(tdgs, doc_id: str, entity: str, on: date) -> dict:
    """Was <entity> active/live on <date>? Derived from START/END facts."""
    if doc_id not in tdgs:
        raise KeyError(f"no document '{doc_id}' (have: {sorted(tdgs)})")
    start, end, evidence = _interval_of(tdgs[doc_id], entity)
    if start is None and end is None:
        return {"answer": "UNKNOWN", "entity": entity,
                "derivation": f"no dated START/END facts matching '{entity}' "
                              f"in {doc_id}",
                "request": f"Supply the start and/or end date of '{entity}'."}
    if start is not None and on < start:
        answer, why = "NO", f"{on} is before start {start}"
    elif end is not None and on > end:
        answer, why = "NO", f"{on} is after end {end}"
    elif start is None:
        answer, why = "YES (start unknown)", f"{on} is on/before end {end}; start date not found"
    elif end is None:
        answer, why = "YES (no end found)", f"{on} is on/after start {start}; no end fact — treated as ongoing"
    else:
        answer, why = "YES", f"{start} <= {on} <= {end}"
    return {
        "answer": answer, "entity": entity, "on": on.isoformat(),
        "start": start.isoformat() if start else None,
        "end": end.isoformat() if end else None,
        "derivation": why,
        "evidence": [{"fact_id": f.id, "role": f.role, "quote": f.sentence}
                     for f in evidence],
    }


# ─── contradictions ──────────────────────────────────────────────────────

def contradiction_report(tdgs, *, composed: bool = True) -> dict:
    """Bundle-level report of cross-document conflicts, with both quotes.

    Reports the same disagreements the chronology marks disputed. These used
    to differ: the report only counted a conflict once the two dates were
    more than the entailment tolerance apart, while the timeline flagged any
    two documents giving different dates. So a two-day discrepancy — the kind
    that decides whether a claim is in time — showed as "0 contradictions"
    beside a chronology reporting one dispute. Whether a gap is small enough
    to ignore is the reader's call, not the report's.
    """
    linker = CrossDocLinker(composed=composed)
    for t in tdgs.values():
        linker.add_tdg(t)
    facts = {(d, f.id): f for d, t in tdgs.items() for f in t.facts}
    items = []
    found = linker.find_coreferences() + linker.find_contradictions()
    conflicting = [l for l in found
                   if l.value_a and l.value_b and l.value_a != l.value_b]
    for link in conflicting:
        fa = facts.get((link.from_doc, link.from_fact))
        fb = facts.get((link.to_doc, link.to_fact))
        items.append({
            "value_a": link.value_a, "value_b": link.value_b,
            "delta_days": link.delta_days,
            "confidence": round(link.confidence, 3),
            "explanation": link.explanation,
            "a": {"doc": link.from_doc, "fact": link.from_fact,
                  "quote": fa.sentence if fa else ""},
            "b": {"doc": link.to_doc, "fact": link.to_fact,
                  "quote": fb.sentence if fb else ""},
        })
    return {"documents": sorted(tdgs), "contradictions": items,
            "count": len(items)}


# ─── whatif ──────────────────────────────────────────────────────────────

def has_dependents(tdgs, key: FactKey) -> bool:
    """Is any date defined relative to this one?

    Moving a date with no dependents changes nothing, which is correct but
    reads as a broken feature when the answer is an empty result and no
    explanation.
    """
    from tdg_core.provenance import trusted_dependencies

    doc_id, fact_id = key
    tdg = tdgs.get(doc_id)
    if tdg is None:
        return False
    return any(d.from_id == fact_id and d.constraint_type == "additive"
               for d in trusted_dependencies(tdg))


def whatif(tdgs, key: FactKey, new_date: date, max_hops: int = 10) -> dict:
    """Move one root date; recompute every downstream date along additive
    dependencies; report the diff. Sources are never mutated (copies)."""
    tdgs = {k: copy.deepcopy(v) for k, v in tdgs.items()}
    root = _find(tdgs, key)
    old_root = root.timex.date_parsed
    if old_root is None:
        raise ValueError(f"{key} has no resolved date to move")
    shift = (new_date - old_root).days

    changes = [{"key": list(key), "entity": root.entity,
                "was": old_root.isoformat(), "now": new_date.isoformat(),
                "via": "user edit (root)"}]
    root.timex.date_parsed = new_date
    root.timex.value = new_date.isoformat()

    doc_id = key[0]
    tdg_doc = tdgs[doc_id]
    fact_map = {f.id: f for f in tdg_doc.facts}

    # BFS along additive dependencies within the document. Only the ones the
    # document actually supports: an invented period must not move a date.
    from tdg_core.provenance import trusted_dependencies

    trusted = trusted_dependencies(tdg_doc)
    frontier = {key[1]}
    seen = set(frontier)
    for _ in range(max_hops):
        nxt = set()
        for dep in trusted:
            if dep.constraint_type != "additive" or dep.from_id not in frontier:
                continue
            if dep.to_id in seen:
                continue
            src, dst = fact_map.get(dep.from_id), fact_map.get(dep.to_id)
            if not src or not dst or src.timex.date_parsed is None:
                continue
            off = _offset_from_text(dep.constraint_expr or "")
            if off is not None:
                placed = off.apply(src.timex.date_parsed)
                via = f"{src.entity} + '{dep.constraint_expr}'"
            elif dep.delta_days is not None:
                placed = src.timex.date_parsed + timedelta(days=dep.delta_days)
                via = f"{src.entity} + {dep.delta_days}d"
            else:
                continue
            was = dst.timex.date_parsed
            dst.timex.date_parsed = placed
            dst.timex.value = placed.isoformat()
            changes.append({"key": [doc_id, dst.id], "entity": dst.entity,
                            "was": was.isoformat() if was else None,
                            "now": placed.isoformat(), "via": via})
            nxt.add(dep.to_id)
            seen.add(dep.to_id)
        if not nxt:
            break
        frontier = nxt

    return {"shift_days": shift, "changes": changes,
            "unchanged_note": "facts with no dependency path from the edited "
                              "date are unaffected by construction"}


# ─── deadline (the funnel) ───────────────────────────────────────────────

def merged_instance(tdgs) -> TemporalDependencyGraph:
    """A chronology-shaped instance TDG: all documents' facts in one graph
    with document-prefixed ids, ready for check_entailment."""
    facts = []
    for doc_id, tdg_doc in tdgs.items():
        for f in tdg_doc.facts:
            g = copy.deepcopy(f)
            g.id = f"{doc_id}:{f.id}"
            if g.is_duplicate_of:
                g.is_duplicate_of = f"{doc_id}:{g.is_duplicate_of}"
            facts.append(g)
    return TemporalDependencyGraph(
        document_id="+".join(sorted(tdgs)), document_type="bundle",
        source_text="", facts=facts)


# ─── stale ───────────────────────────────────────────────────────────────

def stale_report(tdgs, key: FactKey, delta_days: int = 0) -> dict:
    """Which facts stop being trustworthy when one date changes.

    Answers "if this date turns out to be wrong, what else is now wrong?"
    across the whole bundle, following both the arithmetic inside a document
    and the coreference links between documents.

    The engine has had this since the beginning, under a docstring naming
    the use case it was written for -- "which RAG chunks become stale when a
    date changes?" -- and nothing ever called it. Exposing it is what makes
    the hook reachable: a retrieval system holding an answer built on a
    document can ask which of its stored answers a correction invalidates.
    """
    linker = CrossDocLinker(composed=True)
    for t in tdgs.values():
        linker.add_tdg(t)
    root = _find(tdgs, key)
    stale = linker.propagate_staleness(key[0], key[1], delta_days=delta_days)

    facts = {(d, f.id): f for d, t in tdgs.items() for f in t.facts}
    items = []
    for s in stale:
        f = facts.get((s.doc_id, s.fact_id))
        items.append({
            "doc_id": s.doc_id, "fact_id": s.fact_id,
            "entity": f.entity if f else "",
            "value": s.old_value,
            "quote": (f.sentence if f else "") or "",
            "reason": s.reason,
            "same_document": s.doc_id == key[0],
        })
    return {
        "changed": {"key": list(key), "entity": root.entity,
                    "value": root.timex.value,
                    "quote": root.sentence or ""},
        "stale": items,
        "count": len(items),
        "documents_touched": sorted({i["doc_id"] for i in items}),
    }
