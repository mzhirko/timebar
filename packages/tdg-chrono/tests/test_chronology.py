"""Golden tests for the chronology core against a hand-built 3-document
fixture (Phase 1.1): a dismissal letter, an ET1 claim form, and a
grounds-of-resistance response.

Covers: agreed event across documents, disputed event (12 vs 14 July),
single-source event, undated derived event, unplaced relative fact,
merge overrides (split and force-merge), and the never-drop invariant.
"""

from __future__ import annotations

from datetime import date

import pytest

from tdg_core.tdg import (
    TemporalDependencyGraph, TemporalFact, TemporalDependency, TimexSpan)
from tdg_chrono.chronology import (
    MergeOverrides, build_chronology, DEFAULT_MERGE_THRESHOLD)


def _fact(fid, entity, role, sentence, value=None, parsed=None,
          raw="", start=0, end=0, conf=0.9, temporal=True):
    return TemporalFact(
        id=fid, entity=entity, role=role,
        timex=TimexSpan(text=raw or (value or ""), timex_type="DATE",
                        value=value, start_char=start, end_char=end,
                        date_parsed=parsed),
        sentence=sentence, confidence=conf, temporal_content=temporal)


def _document(*sentences: str) -> str:
    """A document body containing the sentences its facts quote.

    The fixtures used to set source_text to a single ellipsis, which made
    every quote in them unverifiable, and the suite then asserted "agreed"
    on that. A test bundle should look like the thing it stands in for.
    """
    return "\n\n".join(sentences) + "\n"


@pytest.fixture()
def bundle():
    letter = TemporalDependencyGraph(
        document_id="dismissal_letter", document_type="legal",
        source_text=_document(
            "Your employment terminates with effect from 12 July 2025.",
            "You commenced employment with the company on 3 March 2019."),
        facts=[
            _fact("f1", "effective date of termination", "END",
                  "Your employment terminates with effect from 12 July 2025.",
                  value="2025-07-12", parsed=date(2025, 7, 12), start=40, end=52),
            _fact("f2", "commencement of employment", "START",
                  "You commenced employment with the company on 3 March 2019.",
                  value="2019-03-03", parsed=date(2019, 3, 3)),
        ])
    et1 = TemporalDependencyGraph(
        document_id="et1", document_type="legal",
        source_text=_document(
            "The claimant's employment terminated with effect from 14 July 2025.",
            "The claim was presented to the tribunal on 1 October 2025.",
            "The response is due within 28 days of service of the claim.",
            "The claim was served on the respondent on 6 October 2025."),
        facts=[
            _fact("f1", "effective date of termination", "END",
                  "The claimant's employment terminated with effect from 14 July 2025.",
                  value="2025-07-14", parsed=date(2025, 7, 14)),
            _fact("f3", "presentation of the claim", "START",
                  "The claim was presented to the tribunal on 1 October 2025.",
                  value="2025-10-01", parsed=date(2025, 10, 1)),
            _fact("f4", "response deadline", "END",
                  "The response is due within 28 days of service of the claim.",
                  value="P28D", raw="within 28 days"),
            _fact("f5", "service of the claim", "START",
                  "The claim was served on the respondent on 6 October 2025.",
                  value="2025-10-06", parsed=date(2025, 10, 6)),
        ],
        dependencies=[
            TemporalDependency(from_id="f5", to_id="f4",
                               constraint_type="additive",
                               constraint_expr="within 28 days of",
                               delta_days=28),
        ])
    response = TemporalDependencyGraph(
        document_id="grounds_of_resistance", document_type="legal",
        source_text=_document(
            "The claimant commenced employment with the company on 3 March 2019.",
            "The respondent denies each and every allegation."),
        facts=[
            _fact("f1", "commencement of employment", "START",
                  "The claimant commenced employment with the company on 3 March 2019.",
                  value="2019-03-03", parsed=date(2019, 3, 3)),
            _fact("f9", "some clause fragment", "UNKNOWN",
                  "The respondent denies each and every allegation.",
                  temporal=False),
        ])
    return {"dismissal_letter": letter, "et1": et1,
            "grounds_of_resistance": response}


def test_nothing_dropped(bundle):
    chron = build_chronology(bundle)
    total_source_facts = sum(len(t.facts) for t in bundle.values())
    placed = sum(len(e.sources) for e in chron.events)
    unplaced = sum(len(u.event.sources) for u in chron.unplaced)
    assert placed + unplaced == total_source_facts


def test_disputed_event_is_one_row_with_both_values(bundle):
    chron = build_chronology(bundle)
    disputed = [e for e in chron.events if e.status == "disputed"]
    assert len(disputed) == 1
    e = disputed[0]
    values = {v.value for v in e.disputed_values}
    assert values == {"2025-07-12", "2025-07-14"}
    docs = {s.doc_id for s in e.sources}
    assert docs == {"dismissal_letter", "et1"}
    # sort key: earliest claimed value, no winner picked
    assert e.date == date(2025, 7, 12)
    # both quotes carried
    assert any("12 July 2025" in s.quote for s in e.sources)
    assert any("14 July 2025" in s.quote for s in e.sources)


def test_agreed_event_merges_across_documents(bundle):
    chron = build_chronology(bundle)
    agreed = [e for e in chron.events
              if e.date == date(2019, 3, 3) and e.status == "agreed"]
    assert len(agreed) == 1
    assert {s.doc_id for s in agreed[0].sources} == {
        "dismissal_letter", "grounds_of_resistance"}


def test_single_source_event(bundle):
    chron = build_chronology(bundle)
    ev = [e for e in chron.events if e.date == date(2025, 10, 1)]
    assert len(ev) == 1 and ev[0].status == "single_source"


def test_derived_row_carries_derivation(bundle):
    chron = build_chronology(bundle)
    derived = [e for e in chron.events if e.status == "derived"]
    assert len(derived) == 1
    e = derived[0]
    assert e.date == date(2025, 11, 3)  # 6 Oct + 28 days
    assert "2025-10-06" in e.derivation
    assert "service of the claim" in e.derivation


def test_non_temporal_fact_goes_to_unplaced_not_dropped(bundle):
    chron = build_chronology(bundle)
    reasons = [u.reason for u in chron.unplaced]
    assert any("no temporal content" in r for r in reasons)


def test_events_sorted_by_date(bundle):
    chron = build_chronology(bundle)
    dated = [e.date for e in chron.events if e.date]
    assert dated == sorted(dated)


def test_split_override_unmerges(bundle):
    ov = MergeOverrides(split=[("et1", "f1")])
    chron = build_chronology(bundle, overrides=ov)
    # the dispute dissolves into two single-source rows
    assert all(e.status != "disputed" for e in chron.events)
    dates = {e.date for e in chron.events}
    assert date(2025, 7, 12) in dates and date(2025, 7, 14) in dates


def test_force_merge_override(bundle):
    # sever the automatic merge with an absurd threshold, then force it back
    ov = MergeOverrides(force_merge=[[("dismissal_letter", "f1"), ("et1", "f1")]])
    chron = build_chronology(bundle, overrides=ov, merge_threshold=1.01)
    disputed = [e for e in chron.events if e.status == "disputed"]
    assert len(disputed) == 1


def test_derivation_can_be_disabled(bundle):
    chron = build_chronology(bundle, derive_undated=False)
    assert all(e.status != "derived" for e in chron.events)
    assert any("relative" in u.reason for u in chron.unplaced)


def test_meta_counts(bundle):
    chron = build_chronology(bundle)
    c = chron.meta["counts"]
    assert c["events"] == len(chron.events)
    assert c["disputed"] == 1
    assert c["unplaced"] == len(chron.unplaced)
