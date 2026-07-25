"""A disagreement must never be reported as agreement.

Dispute detection compared only fully-parsed dates, so a cluster holding
"2019" and "2020", or notice periods of 14 and 21 days, came out labelled
**agreed** — an assurance the sources do not support. In a document whose
purpose is to be checked against its own quotes, a false "agreed" is worse
than a gap: it tells the reader not to look.
"""

from __future__ import annotations

from datetime import date

from tdg_core.tdg import TemporalDependencyGraph, TemporalFact, TimexSpan

from tdg_chrono.chronology import build_chronology


def f(fid, entity, role, value, parsed=None, dur=None, ttype="DATE"):
    return TemporalFact(
        id=fid, entity=entity, role=role,
        timex=TimexSpan(text=str(value), timex_type=ttype, value=value,
                        start_char=0, end_char=0, date_parsed=parsed,
                        duration_days=dur),
        sentence=f"{entity} {value}")


def doc(did, *facts):
    """A document whose body contains the sentences its facts quote.

    Without a body every quote is unverifiable, and unverifiable quotes are
    deliberately not allowed to corroborate one another.
    """
    body = "\n\n".join(f.sentence for f in facts if f.sentence) + "\n"
    return TemporalDependencyGraph(document_id=did, document_type="legal",
                                   source_text=body, facts=list(facts))


def build(a, b):
    return build_chronology({d.document_id: d for d in (a, b)})


def statuses(chron):
    return {e.status for e in chron.events}


def test_conflicting_dates_are_disputed():
    chron = build(
        doc("x", f("a", "employment termination", "END", "2025-07-12", date(2025, 7, 12))),
        doc("y", f("b", "employment termination", "END", "2025-07-14", date(2025, 7, 14))))
    assert chron.meta["counts"]["disputed"] == 1


def test_conflicting_years_are_disputed_not_agreed():
    chron = build(doc("x", f("a", "employment", "START", "2019")),
                  doc("y", f("b", "employment", "START", "2020")))
    assert "agreed" not in statuses(chron)
    assert chron.meta["counts"]["disputed"] == 1


def test_conflicting_months_are_disputed():
    chron = build(doc("x", f("a", "employment termination", "END", "2025-07")),
                  doc("y", f("b", "employment termination", "END", "2025-09")))
    assert chron.meta["counts"]["disputed"] == 1


def test_conflicting_notice_periods_are_disputed():
    """14 days versus 21 days to appeal is a conflict a reader must see."""
    chron = build(
        doc("x", f("a", "appeal deadline", "DURATION", "P14D", None, 14, "DURATION")),
        doc("y", f("b", "appeal deadline", "DURATION", "P21D", None, 21, "DURATION")))
    assert chron.meta["counts"]["disputed"] == 1


def test_a_coarser_value_refined_by_a_finer_one_stays_agreed():
    """July 2025 and 14 July 2025 are consistent, not contradictory."""
    chron = build(
        doc("x", f("a", "employment termination", "END", "2025-07")),
        doc("y", f("b", "employment termination", "END", "2025-07-14", date(2025, 7, 14))))
    assert chron.meta["counts"]["disputed"] == 0
    assert "agreed" in statuses(chron)


def test_disputed_row_keeps_every_value_and_quote():
    chron = build(doc("x", f("a", "employment", "START", "2019")),
                  doc("y", f("b", "employment", "START", "2020")))
    ev = [e for e in chron.events if e.status == "disputed"][0]
    assert {v.value for v in ev.disputed_values} == {"2019", "2020"}
    assert len(ev.sources) == 2
    assert all(s.quote for s in ev.sources)


# ── relative expressions resolve before linking ─────────────────────────

def test_a_derived_date_can_contradict_an_explicit_one():
    """"Within 28 days of service" versus another document's stated deadline.

    Derivation used to run after clustering, so a relative expression never
    met the date it disagreed with and sat unplaced instead.
    """
    from tdg_core.tdg import TemporalDependency

    et1 = TemporalDependencyGraph(
        document_id="et1", document_type="legal", source_text="",
        facts=[f("f1", "service of the claim", "START", "2025-10-06",
                 date(2025, 10, 6)),
               f("f2", "response deadline", "END", None)],
        dependencies=[TemporalDependency(from_id="f1", to_id="f2",
                                         constraint_type="additive",
                                         constraint_expr="within 28 days of",
                                         delta_days=28)])
    grounds = doc("grounds", f("g1", "response deadline", "END", "2025-11-14",
                               date(2025, 11, 14)))
    chron = build_chronology({t.document_id: t for t in (et1, grounds)})
    assert chron.meta["counts"]["disputed"] == 1


def test_a_derived_date_shows_its_working():
    from tdg_core.tdg import TemporalDependency

    et1 = TemporalDependencyGraph(
        document_id="et1", document_type="legal", source_text="",
        facts=[f("f1", "service of the claim", "START", "2025-10-06",
                 date(2025, 10, 6)),
               f("f2", "response deadline", "END", None)],
        dependencies=[TemporalDependency(from_id="f1", to_id="f2",
                                         constraint_type="additive",
                                         constraint_expr="within 28 days of",
                                         delta_days=28)])
    chron = build_chronology({"et1": et1})
    derived = [e for e in chron.events if e.status == "derived"]
    assert derived, "the deadline must reach the timeline"
    assert "2025-11-03" in derived[0].derivation
    assert "within 28 days of" in derived[0].derivation


def test_sources_are_never_mutated_by_resolution():
    """Resolution fills in dates on a copy; the caller's graphs stand."""
    from tdg_core.tdg import TemporalDependency

    et1 = TemporalDependencyGraph(
        document_id="et1", document_type="legal", source_text="",
        facts=[f("f1", "service of the claim", "START", "2025-10-06",
                 date(2025, 10, 6)),
               f("f2", "response deadline", "END", None)],
        dependencies=[TemporalDependency(from_id="f1", to_id="f2",
                                         constraint_type="additive",
                                         constraint_expr="within 28 days of",
                                         delta_days=28)])
    build_chronology({"et1": et1})
    assert et1.facts[1].timex.date_parsed is None
    assert et1.facts[1].timex.value is None


# ── unverifiable quotes must not corroborate ────────────────────────────

def _unchecked(did, sentence, iso):
    """A document asserting a date but shipping no text to check it against."""
    return TemporalDependencyGraph(
        document_id=did, document_type="legal", source_text="…",
        facts=[f("x", "commencement of employment", "START", iso,
                 date(*(int(p) for p in iso.split("-"))))])


def test_an_unverifiable_document_does_not_make_a_date_agreed():
    """This is the exact shape that shipped: a document with no text,
    offsets of zero, and a quote raising another document's date to
    "agreed". Two sources, but only one of them can be checked."""
    checked = doc("letter", f("a", "commencement of employment", "START",
                              "2019-03-03", date(2019, 3, 3)))
    chron = build_chronology({"letter": checked,
                              "response": _unchecked("response", "", "2019-03-03")})
    assert "agreed" not in statuses(chron), (
        "corroboration from an uncheckable quote is not corroboration")


def test_two_verifiable_documents_still_agree():
    a = doc("letter", f("a", "commencement of employment", "START",
                        "2019-03-03", date(2019, 3, 3)))
    b = doc("response", f("b", "commencement of employment", "START",
                          "2019-03-03", date(2019, 3, 3)))
    chron = build_chronology({"letter": a, "response": b})
    assert "agreed" in statuses(chron)


def test_the_run_reports_how_many_quotes_could_not_be_checked():
    chron = build_chronology({
        "letter": doc("letter", f("a", "commencement of employment", "START",
                                  "2019-03-03", date(2019, 3, 3))),
        "response": _unchecked("response", "", "2019-03-03")})
    assert chron.meta["counts"]["unverified_quotes"] == 1
    assert chron.meta["provenance"]["response"]["has_source_text"] is False
