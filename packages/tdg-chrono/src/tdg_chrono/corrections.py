"""Corrections loop (Phase 1.5).

Row-level accept / reject / edit / merge / split, stored in a
``corrections.json`` file keyed by ``(doc_id, fact_id)`` and re-applied
automatically on every rebuild.

Invariants (from RELEASE_TODO 1.5):
  - **Additive and reversible.** The source TDGs are never mutated;
    corrections apply to in-memory copies at build time. Deleting an
    entry from the file fully reverts it on the next run.
  - **Last-wins.** A later correction for the same fact and operation
    class overrides an earlier one (so "accept" cancels a prior
    "reject" without editing history).
  - **Nothing disappears silently.** Rejected facts leave the timeline
    but are listed in the chronology metadata with their quotes.
  - **Every correction is gold annotation.** Each entry records the
    original value (``was``) and a timestamp, so the file doubles as an
    annotation set harvested from normal use.

File format (version 1)::

    {"version": 1, "corrections": [
        {"op": "reject",    "doc_id": "et1", "fact_id": "f4", "note": "...", "ts": "..."},
        {"op": "accept",    "doc_id": "et1", "fact_id": "f1"},
        {"op": "edit_date", "doc_id": "et1", "fact_id": "f1",
         "new_date": "2025-07-12", "was": "2025-07-14"},
        {"op": "edit_label","doc_id": "et1", "fact_id": "f1",
         "new_label": "effective date of termination", "was": "termination"},
        {"op": "merge", "keys": [["dismissal_letter", "f1"], ["et1", "f1"]]},
        {"op": "split", "doc_id": "et1", "fact_id": "f1"}
    ]}
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

from tdg_core.tdg import TemporalDependencyGraph

from tdg_chrono.chronology import Chronology, FactKey, MergeOverrides

OPS = ("accept", "reject", "edit_date", "edit_label", "merge", "split")


@dataclass
class Correction:
    op: str
    doc_id: Optional[str] = None
    fact_id: Optional[str] = None
    new_date: Optional[str] = None
    new_label: Optional[str] = None
    keys: Optional[list[list[str]]] = None   # for merge
    note: str = ""
    was: Optional[str] = None                # original value — the gold harvest
    ts: Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v not in (None, "")}

    @property
    def key(self) -> Optional[FactKey]:
        if self.doc_id and self.fact_id:
            return (self.doc_id, self.fact_id)
        return None


@dataclass
class CorrectionOutcome:
    tdgs: dict[str, TemporalDependencyGraph]   # corrected COPIES
    overrides: MergeOverrides
    accepted: set[FactKey]
    rejected: list[dict]                       # visible, never silent
    edits: list[dict]


# ─── File I/O ────────────────────────────────────────────────────────────

def load_corrections(path: str | Path) -> list[Correction]:
    p = Path(path)
    if not p.exists():
        return []
    data = json.loads(p.read_text())
    if data.get("version", 1) != 1:
        raise ValueError(f"unsupported corrections version {data.get('version')!r}")
    out = []
    for c in data.get("corrections", []):
        if c.get("op") not in OPS:
            raise ValueError(f"unknown correction op {c.get('op')!r} (known: {OPS})")
        out.append(Correction(**c))
    return out


def save_corrections(path: str | Path, corrections: list[Correction]) -> None:
    Path(path).write_text(json.dumps(
        {"version": 1, "corrections": [c.to_dict() for c in corrections]},
        indent=2))


def append_correction(path: str | Path, correction: Correction) -> None:
    corrections = load_corrections(path)
    correction.ts = correction.ts or datetime.now(timezone.utc).isoformat(timespec="seconds")
    corrections.append(correction)
    save_corrections(path, corrections)


# ─── Application ─────────────────────────────────────────────────────────

def apply_corrections(
    tdgs: dict[str, TemporalDependencyGraph],
    corrections: list[Correction],
) -> CorrectionOutcome:
    """Apply corrections to deep copies of the TDGs. Sources untouched."""
    tdgs = {k: copy.deepcopy(v) for k, v in tdgs.items()}
    overrides = MergeOverrides()

    # last-wins per (fact, op-class): walk in order, keep the final state
    accept_state: dict[FactKey, str] = {}      # key -> "accept" | "reject"
    date_edit: dict[FactKey, Correction] = {}
    label_edit: dict[FactKey, Correction] = {}
    for c in corrections:
        if c.op in ("accept", "reject") and c.key:
            accept_state[c.key] = c.op
        elif c.op == "edit_date" and c.key:
            date_edit[c.key] = c
        elif c.op == "edit_label" and c.key:
            label_edit[c.key] = c
        elif c.op == "merge" and c.keys:
            overrides.force_merge.append([tuple(k) for k in c.keys])
        elif c.op == "split" and c.key:
            overrides.split.append(c.key)

    accepted = {k for k, v in accept_state.items() if v == "accept"}
    rejected_keys = {k for k, v in accept_state.items() if v == "reject"}

    edits: list[dict] = []
    rejected: list[dict] = []

    for doc_id, tdg in tdgs.items():
        kept = []
        for f in tdg.facts:
            k = (doc_id, f.id)
            if k in rejected_keys:
                rejected.append({
                    "doc_id": doc_id, "fact_id": f.id,
                    "entity": f.entity, "quote": f.sentence,
                    "was_value": f.timex.value,
                })
                continue  # excluded from the copy → excluded from clustering
            if k in date_edit:
                c = date_edit[k]
                c.was = c.was or (f.timex.value or "")
                f.timex.value = c.new_date
                f.timex.date_parsed = date.fromisoformat(c.new_date)
                f.confidence = 1.0            # a human set it
                edits.append({"key": list(k), "field": "date",
                              "was": c.was, "now": c.new_date})
            if k in label_edit:
                c = label_edit[k]
                c.was = c.was or f.entity
                f.entity = c.new_label
                edits.append({"key": list(k), "field": "label",
                              "was": c.was, "now": c.new_label})
            kept.append(f)
        tdg.facts = kept

    return CorrectionOutcome(tdgs=tdgs, overrides=overrides,
                             accepted=accepted, rejected=rejected, edits=edits)


def mark_confirmed(chron: Chronology, accepted: set[FactKey]) -> None:
    """Post-build: a human-accepted source confirms its whole row."""
    for e in chron.events:
        if any((s.doc_id, s.fact_id) in accepted for s in e.sources):
            e.confidence = 1.0
    chron.meta.setdefault("counts", {})["confirmed"] = sum(
        1 for e in chron.events
        if any((s.doc_id, s.fact_id) in accepted for s in e.sources))


def export_gold(corrections: list[Correction]) -> list[dict]:
    """The annotation harvest: every human decision with before/after."""
    return [c.to_dict() for c in corrections
            if c.op in ("accept", "reject", "edit_date", "edit_label")]
