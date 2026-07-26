"""Corrections loop tests (Phase 1.5).

Uses the same 3-document fixture as test_chronology. Verifies every
invariant: source TDGs never mutated, corrections reversible by
deletion, last-wins per fact, rejects visible not silent, edits
resolve disputes, and the gold-annotation harvest.
"""

from __future__ import annotations

from datetime import date

import pytest

from tdg_chrono.chronology import build_chronology
from tdg_chrono.corrections import (Correction, apply_corrections,
                                    export_gold, load_corrections,
                                    mark_confirmed, save_corrections)

from test_chronology import bundle  # noqa: F401  (reuse the fixture)


def _build(bundle, corrections):
    outcome = apply_corrections(bundle, corrections)
    chron = build_chronology(outcome.tdgs, overrides=outcome.overrides)
    mark_confirmed(chron, outcome.accepted)
    return chron, outcome


def test_sources_never_mutated(bundle):
    corrections = [Correction(op="edit_date", doc_id="et1", fact_id="f1",
                              new_date="2025-07-12"),
                   Correction(op="reject", doc_id="grounds_of_resistance",
                              fact_id="f9")]
    before = {d: t.to_json() for d, t in bundle.items()}
    _build(bundle, corrections)
    after = {d: t.to_json() for d, t in bundle.items()}
    assert before == after


def test_edit_date_resolves_dispute_and_records_was(bundle):
    corr = Correction(op="edit_date", doc_id="et1", fact_id="f1",
                      new_date="2025-07-12")
    chron, outcome = _build(bundle, [corr])
    assert all(e.status != "disputed" for e in chron.events)
    merged = [e for e in chron.events if e.date == date(2025, 7, 12)]
    assert len(merged) == 1 and merged[0].status == "agreed"
    # the gold harvest keeps the original value
    assert corr.was == "2025-07-14"
    assert outcome.edits[0]["was"] == "2025-07-14"


def test_reject_removes_row_but_stays_visible(bundle):
    corrections = [Correction(op="reject", doc_id="et1", fact_id="f3")]
    chron, outcome = _build(bundle, corrections)
    assert not any(e.date == date(2025, 10, 1) for e in chron.events)
    assert outcome.rejected[0]["fact_id"] == "f3"
    assert "presented to the tribunal" in outcome.rejected[0]["quote"]


def test_reversible_by_deleting_the_entry(bundle):
    corrections = [Correction(op="reject", doc_id="et1", fact_id="f3")]
    chron_with, _ = _build(bundle, corrections)
    chron_without, _ = _build(bundle, [])   # entry deleted → full revert
    assert not any(e.date == date(2025, 10, 1) for e in chron_with.events)
    assert any(e.date == date(2025, 10, 1) for e in chron_without.events)


def test_last_wins_accept_cancels_reject(bundle):
    corrections = [Correction(op="reject", doc_id="et1", fact_id="f3"),
                   Correction(op="accept", doc_id="et1", fact_id="f3")]
    chron, outcome = _build(bundle, corrections)
    row = [e for e in chron.events if e.date == date(2025, 10, 1)]
    assert len(row) == 1
    assert row[0].confidence == 1.0          # human-confirmed
    assert not outcome.rejected
    assert chron.meta["counts"]["confirmed"] >= 1


def test_merge_and_split_ops(bundle):
    split_only = [Correction(op="split", doc_id="et1", fact_id="f1")]
    chron, _ = _build(bundle, split_only)
    assert all(e.status != "disputed" for e in chron.events)

    remerge = split_only + [Correction(
        op="merge", keys=[["dismissal_letter", "f1"], ["et1", "f1"]])]
    # NOTE: split wins over merge for the fact it names (documented order)
    chron2, _ = _build(bundle, remerge)
    assert all(e.status != "disputed" for e in chron2.events)


def test_file_roundtrip_and_gold_export(tmp_path, bundle):
    f = tmp_path / "corrections.json"
    corrections = [
        Correction(op="edit_date", doc_id="et1", fact_id="f1",
                   new_date="2025-07-12", note="letter is controlling"),
        Correction(op="merge", keys=[["a", "f1"], ["b", "f1"]]),
    ]
    _build(bundle, corrections)              # populates `was`
    save_corrections(f, corrections)
    reloaded = load_corrections(f)
    assert reloaded[0].was == "2025-07-14"
    gold = export_gold(reloaded)
    # A merge IS an annotation, and the most informative one available: it
    # says two facts are the same event, which is the judgement the
    # cross-document linker exists to make. Excluding it discarded the only
    # labelled data the tool produces by being used.
    assert len(gold) == 2
    edits = [g for g in gold if g["op"] == "edit_date"]
    assert edits[0]["was"] == "2025-07-14" and edits[0]["new_date"] == "2025-07-12"
    merges = [g for g in gold if g["op"] == "merge"]
    assert merges[0]["keys"] == [["a", "f1"], ["b", "f1"]]


def test_unknown_op_rejected(tmp_path):
    f = tmp_path / "c.json"
    f.write_text('{"version": 1, "corrections": [{"op": "delete_everything"}]}')
    with pytest.raises(ValueError, match="unknown correction op"):
        load_corrections(f)
