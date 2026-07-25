"""The recall audit turns a silent extraction miss into a visible gap.

The failure it exists to catch: a date sits in the document, the extractor
emits no fact for it, and the chronology looks complete because everything
downstream only ever sees facts.
"""

from __future__ import annotations

from datetime import date

from tdg_core.tdg import TemporalDependencyGraph, TemporalFact, TimexSpan

from tdg_chrono.recall import audit_recall, format_report

TEXT = ("I write to confirm the hearing held on 28 May 2025. "
        "Your employment terminates with effect from 12 July 2025.")


def _tdg(*dates: str) -> TemporalDependencyGraph:
    facts = []
    for i, value in enumerate(dates, 1):
        y, m, d = (int(x) for x in value.split("-"))
        facts.append(TemporalFact(
            id=f"f{i}",
            entity="employment",
            role="END",
            timex=TimexSpan(text=value, timex_type="DATE", value=value,
                            start_char=0, end_char=0,
                            date_parsed=date(y, m, d)),
            sentence="",
        ))
    return TemporalDependencyGraph(document_id="doc", document_type="legal",
                                   source_text=TEXT, facts=facts)


def test_reports_a_date_the_extractor_missed():
    missed = audit_recall(TEXT, _tdg("2025-05-28"))
    assert [m.value for m in missed] == ["2025-07-12"]


def test_silent_when_every_date_is_accounted_for():
    assert audit_recall(TEXT, _tdg("2025-05-28", "2025-07-12")) == []


def test_total_extraction_failure_reports_every_date():
    missed = audit_recall(TEXT, _tdg())
    assert {m.value for m in missed} == {"2025-05-28", "2025-07-12"}


def test_missed_date_is_quoted_in_context():
    """A bare date is not actionable; the reviewer needs the sentence."""
    missed = audit_recall(TEXT, _tdg("2025-05-28"))
    assert "employment terminates" in missed[0].sentence


def test_durations_are_not_reported_by_default():
    """'within 28 days' is an edge in the graph, not a dated fact."""
    text = "The response is due within 28 days of service."
    assert audit_recall(text, _tdg()) == []
    assert audit_recall(text, _tdg(), include_durations=True)


def test_report_is_empty_when_nothing_missed():
    assert format_report({"doc": []}) == []


def test_report_names_the_document_and_the_date():
    lines = format_report({"dismissal_letter": audit_recall(TEXT, _tdg("2025-05-28"))})
    body = "\n".join(lines)
    assert "dismissal_letter" in body
    assert "12 July 2025" in body
