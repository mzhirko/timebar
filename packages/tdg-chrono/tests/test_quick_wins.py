"""Four small fixes, each pinned so it cannot quietly regress."""

from __future__ import annotations

from datetime import date

import pytest

from tdg_core.io import build_tdg
from tdg_core.provenance import check_source_hash, source_hash
from tdg_core.tdg import (TemporalDependency, TemporalDependencyGraph,
                          TemporalFact, TimexSpan)

from tdg_chrono import capabilities
from tdg_chrono.chronology import DEFAULT_MERGE_THRESHOLD, build_chronology

BODY = ("The claim was served on the respondent on 6 October 2025.\n"
        "The response is due within 28 days of service of the claim.\n")


def fact(fid, entity, iso=None, sentence=""):
    parsed = date(*(int(p) for p in iso.split("-"))) if iso else None
    return TemporalFact(
        id=fid, entity=entity, role="START" if iso else "END",
        timex=TimexSpan(text=iso or "", timex_type="DATE", value=iso,
                        start_char=0, end_char=0, date_parsed=parsed),
        sentence=sentence)


@pytest.fixture()
def linked():
    return {"et1": TemporalDependencyGraph(
        document_id="et1", document_type="legal", source_text=BODY,
        facts=[fact("f5", "service of the claim", "2025-10-06",
                    "The claim was served on the respondent on 6 October 2025."),
               fact("f4", "response deadline", None,
                    "The response is due within 28 days of service of the claim.")],
        dependencies=[TemporalDependency(
            from_id="f5", to_id="f4", constraint_type="additive",
            constraint_expr="within 28 days of", delta_days=28)])}


# ── 1. the second merge threshold no longer pretends to be a control ────

def test_the_extra_merge_filter_defaults_to_off():
    """At 0.45 it sat below the linker's own bar and could never reject
    anything, while reading as the threshold that governed merging."""
    assert DEFAULT_MERGE_THRESHOLD == 0.0


def test_raising_the_extra_filter_still_makes_merging_stricter():
    """The parameter is kept because it can still tighten merging.

    Checked against the links' own confidence rather than a guessed number:
    two documents describing one event word for word score a perfect 1.0, so
    a hardcoded 0.99 would have proved nothing.
    """
    from tdg_core.cross_doc import CrossDocLinker

    body = "Employment commenced on 3 March 2019."
    a = TemporalDependencyGraph(
        document_id="a", document_type="legal", source_text=body,
        facts=[fact("x", "commencement of employment", "2019-03-03", body)])
    b = TemporalDependencyGraph(
        document_id="b", document_type="legal", source_text=body,
        facts=[fact("y", "commencement of employment", "2019-03-03", body)])

    linker = CrossDocLinker(composed=True)
    for t in (a, b):
        linker.add_tdg(t)
    links = linker.find_coreferences()
    assert links, "the two documents should link at all"
    above_any_link = max(l.confidence for l in links) + 0.01

    merged = build_chronology({"a": a, "b": b})
    apart = build_chronology({"a": a, "b": b}, merge_threshold=above_any_link)
    assert len(merged.events) == 1
    assert len(apart.events) == 2, "raising the bar above every link splits them"


# ── 2. what-if says when nothing depends on the date ────────────────────

def test_a_date_with_dependents_is_recognised(linked):
    assert capabilities.has_dependents(linked, ("et1", "f5"))


def test_a_date_with_no_dependents_is_recognised(linked):
    assert not capabilities.has_dependents(linked, ("et1", "f4"))


def test_moving_a_date_with_dependents_moves_them(linked):
    out = capabilities.whatif(linked, ("et1", "f5"), date(2025, 10, 13))
    assert len(out["changes"]) > 1


# ── 3. the staleness hook is reachable ──────────────────────────────────

def test_stale_reports_what_a_wrong_date_invalidates(linked):
    out = capabilities.stale_report(linked, ("et1", "f5"))
    assert out["count"] == 1
    assert out["stale"][0]["entity"] == "response deadline"
    assert "within 28 days of" in out["stale"][0]["reason"]


def test_stale_carries_the_quote_for_each_affected_fact(linked):
    out = capabilities.stale_report(linked, ("et1", "f5"))
    assert out["stale"][0]["quote"].startswith("The response is due")


def test_stale_is_empty_when_nothing_depends_on_the_date(linked):
    out = capabilities.stale_report(linked, ("et1", "f4"))
    assert out["count"] == 0


# ── 4. the hash the schema promised is now read ─────────────────────────

def _graph(**kw):
    return TemporalDependencyGraph(document_id="d", document_type="legal", **kw)


def test_a_matching_hash_verifies():
    verdict, _ = check_source_hash(
        _graph(source_text=BODY, source_text_sha256=source_hash(BODY)))
    assert verdict == "match"


def test_a_stale_hash_is_caught():
    verdict, detail = check_source_hash(
        _graph(source_text=BODY, source_text_sha256="f" * 64))
    assert verdict == "mismatch" and "stale" in detail


def test_an_externalised_text_is_distinguished_from_no_provenance():
    """A hash with no text is a bundle keeping its promise; neither is not."""
    assert check_source_hash(_graph(source_text="",
                                    source_text_sha256="a" * 64))[0] == "hash-only"
    assert check_source_hash(_graph(source_text=""))[0] == "neither"


def test_the_hash_round_trips_through_the_format():
    tdg = build_tdg({"document_id": "d", "facts": [], "dependencies": [],
                     "source_text_sha256": "b" * 64})
    assert tdg.source_text_sha256 == "b" * 64
    assert tdg.to_dict()["source_text_sha256"] == "b" * 64


# ── every consumer of additive edges must reject a fabricated one ────────
#
# Guarding only the chronology left three other paths open. A "45 days" edge
# the document never states was correctly refused by the timeline while
# What-if used it to move a termination date by a month, `stale` reported a
# fact as invalidated by it, and the viewer advertised the dependency as
# real. A rejected relation has to be rejected everywhere or it is not
# rejected at all.

FABRICATED_BODY = ("The disciplinary hearing was held on 28 May 2025.\n"
                   "Your employment terminates with effect from 12 July 2025.\n")


@pytest.fixture()
def fabricated():
    """One document whose only relation asserts a period it never states."""
    return {"letter": TemporalDependencyGraph(
        document_id="letter", document_type="legal",
        source_text=FABRICATED_BODY,
        facts=[fact("a", "hearing", "2025-05-28",
                    "The disciplinary hearing was held on 28 May 2025."),
               fact("b", "employment termination", "2025-07-12",
                    "Your employment terminates with effect from 12 July 2025.")],
        dependencies=[TemporalDependency(
            from_id="a", to_id="b", constraint_type="additive",
            constraint_expr="45 days", delta_days=45)])}


def test_the_timeline_rejects_it(fabricated):
    chron = build_chronology(fabricated)
    assert chron.meta["counts"]["unsupported_relations"] == 1
    assert not [e for e in chron.events if e.status == "derived"]


def test_whatif_does_not_move_a_date_on_an_invented_rule(fabricated):
    out = capabilities.whatif(fabricated, ("letter", "a"), date(2025, 6, 28))
    downstream = [c for c in out["changes"] if c["via"] != "user edit (root)"]
    assert downstream == [], "an invented period moved a real date"


def test_the_viewer_does_not_advertise_an_invented_dependency(fabricated):
    assert not capabilities.has_dependents(fabricated, ("letter", "a"))


def test_stale_does_not_report_an_invented_consequence(fabricated):
    assert capabilities.stale_report(fabricated, ("letter", "a"))["count"] == 0


def test_a_genuine_rule_still_works_everywhere(linked):
    """The guard must not cost the real case anything."""
    assert capabilities.has_dependents(linked, ("et1", "f5"))
    out = capabilities.whatif(linked, ("et1", "f5"), date(2025, 10, 13))
    assert len(out["changes"]) > 1
    assert capabilities.stale_report(linked, ("et1", "f5"))["count"] == 1
    assert build_chronology(linked).meta["counts"]["unsupported_relations"] == 0
