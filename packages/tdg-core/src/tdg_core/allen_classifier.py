"""
Allen Interval Algebra classifier for TDG.

Adds Allen relation labels to pairs of temporal facts that have
resolvable date intervals. This is a post-processing step that runs
after graph_builder.py — it does not modify facts or existing edges.

All 13 Allen relations are supported:
  BEFORE, AFTER, MEETS, MET_BY,
  OVERLAPS, OVERLAPPED_BY,
  STARTS, STARTED_BY,
  DURING, CONTAINS,
  FINISHES, FINISHED_BY,
  EQUALS

Usage:
    from allen_classifier import add_allen_relations

    tdg = pipeline.process(text, ...)
    new_edges = add_allen_relations(tdg)
    # tdg.dependencies now contains original edges + Allen edges

New Allen edges have constraint_type="allen" and allen_relation set to
one of the 13 relation strings. They serialise correctly via to_dict().
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from tdg_core.tdg import TemporalDependency, TemporalDependencyGraph, TemporalFact


# ---------------------------------------------------------------------------
# Core classifier
# ---------------------------------------------------------------------------

def classify_allen(
    a_start: Optional[date],
    a_end: Optional[date],
    b_start: Optional[date],
    b_end: Optional[date],
) -> str:
    """
    Classify the Allen interval relation between A=[a_start, a_end]
    and B=[b_start, b_end].

    Returns one of the 13 Allen relation strings, or "UNKNOWN" when
    any endpoint is None or when an interval is invalid (start > end).

    EQUALS is checked before MEETS so that point events (start==end)
    on the same date classify correctly as EQUALS rather than MEETS.
    """
    if None in (a_start, a_end, b_start, b_end):
        return "UNKNOWN"
    if a_start > a_end or b_start > b_end:
        return "UNKNOWN"

    if a_start == b_start and a_end == b_end:
        return "EQUALS"
    if a_end < b_start:
        return "BEFORE"
    if b_end < a_start:
        return "AFTER"
    if a_end == b_start:
        return "MEETS"
    if b_end == a_start:
        return "MET_BY"
    if a_start < b_start and a_end > b_start and a_end < b_end:
        return "OVERLAPS"
    if b_start < a_start and b_end > a_start and b_end < a_end:
        return "OVERLAPPED_BY"
    if a_start == b_start and a_end < b_end:
        return "STARTS"
    if a_start == b_start and a_end > b_end:
        return "STARTED_BY"
    if a_start > b_start and a_end == b_end:
        return "FINISHES"
    if a_start < b_start and a_end == b_end:
        return "FINISHED_BY"
    if a_start > b_start and a_end < b_end:
        return "DURING"
    if a_start < b_start and a_end > b_end:
        return "CONTAINS"
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Interval resolution
# ---------------------------------------------------------------------------

def _resolve_interval(fact: TemporalFact) -> tuple[Optional[date], Optional[date]]:
    """
    Derive a [start, end] date interval from a TemporalFact.

    Facts with a single parsed date (START, END, or a dated DURATION)
    are treated as point intervals [date, date]. This is correct for
    instantaneous legal events (signing, expiry, accession).

    Returns (None, None) when no date can be resolved.
    """
    d = fact.timex.date_parsed
    if d is not None:
        return (d, d)
    return (None, None)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def add_allen_relations(tdg: TemporalDependencyGraph) -> list[TemporalDependency]:
    """
    Compute Allen relations between all pairs of facts that have
    resolvable date intervals and add them as new edges to tdg.dependencies.

    Rules:
    - Only facts with a resolvable date interval are considered.
    - Pairs that already have an existing edge in either direction are skipped
      to avoid duplication.
    - Self-pairs are skipped.
    - Relations that resolve to UNKNOWN are not added.

    Args:
        tdg: A TemporalDependencyGraph (mutated in place).

    Returns:
        The list of new TemporalDependency edges that were added.
    """
    existing_pairs: set[tuple[str, str]] = {
        (d.from_id, d.to_id) for d in tdg.dependencies
    }
    existing_pairs_rev: set[tuple[str, str]] = {
        (d.to_id, d.from_id) for d in tdg.dependencies
    }

    # Resolve intervals once
    resolved: list[tuple[TemporalFact, date, date]] = []
    for f in tdg.facts:
        start, end = _resolve_interval(f)
        if start is not None:
            resolved.append((f, start, end))

    new_deps: list[TemporalDependency] = []

    for i, (fa, a_s, a_e) in enumerate(resolved):
        for fb, b_s, b_e in resolved[i + 1:]:
            pair = (fa.id, fb.id)
            pair_rev = (fb.id, fa.id)

            if pair in existing_pairs or pair in existing_pairs_rev:
                continue
            if pair_rev in existing_pairs or pair_rev in existing_pairs_rev:
                continue

            relation = classify_allen(a_s, a_e, b_s, b_e)
            if relation == "UNKNOWN":
                continue

            dep = TemporalDependency(
                from_id=fa.id,
                to_id=fb.id,
                constraint_type="allen",
                constraint_expr=f"{fa.id} {relation} {fb.id}",
                confidence=0.8,
                verified=True,
                allen_relation=relation,
            )
            new_deps.append(dep)
            existing_pairs.add(pair)

    tdg.dependencies.extend(new_deps)
    return new_deps
