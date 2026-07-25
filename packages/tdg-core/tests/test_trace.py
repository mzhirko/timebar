"""Trace tests (Phase 1.6) on a s.111-style fixture:
"within the period of three months beginning with the effective date of
termination" — checked against a case with a dismissal, a claim
presentation, and a later hearing that must be passed over.
"""

from __future__ import annotations

from datetime import date

import pytest

from tdg_core.tdg import (TemporalDependencyGraph, TemporalFact,
                          TemporalDependency, TimexSpan)
from tdg_core.entailment import check_entailment
from tdg_core.trace import render_text, render_html


def _fact(fid, entity, role, sentence, value=None, parsed=None):
    return TemporalFact(
        id=fid, entity=entity, role=role,
        timex=TimexSpan(text=value or "", timex_type="DATE", value=value,
                        start_char=0, end_char=0, date_parsed=parsed),
        sentence=sentence)


STATUTE_SENT = ("A complaint shall be presented to the tribunal within the "
                "period of three months beginning with the effective date "
                "of termination.")


@pytest.fixture()
def statute():
    return TemporalDependencyGraph(
        document_id="era1996_s111", document_type="statute", source_text=STATUTE_SENT,
        facts=[
            _fact("r1", "effective date of termination", "START", STATUTE_SENT),
            _fact("r2", "presentation of the complaint", "END", STATUTE_SENT,
                  value="P3M"),
        ],
        dependencies=[TemporalDependency(
            from_id="r1", to_id="r2", constraint_type="additive",
            constraint_expr="3 months")])


@pytest.fixture()
def case():
    return TemporalDependencyGraph(
        document_id="case_x", document_type="judgment", source_text="…",
        facts=[
            _fact("f1", "effective date of termination", "END",
                  "The claimant was dismissed with effect from 12 July 2025.",
                  value="2025-07-12", parsed=date(2025, 7, 12)),
            _fact("f2", "claim presented to the tribunal", "START",
                  "The claim was presented to the tribunal on 10 October 2025.",
                  value="2025-10-10", parsed=date(2025, 10, 10)),
            _fact("f3", "preliminary hearing", "START",
                  "A preliminary hearing took place on 5 November 2025.",
                  value="2025-11-05", parsed=date(2025, 11, 5)),
        ])


def test_deadline_arithmetic_and_trace(statute, case):
    (r,) = check_entailment(statute, case)
    # 12 Jul + 3 months, anchor day counts ("beginning with") → 11 Oct
    assert r.deadline_computed == "2025-10-11"
    assert r.action_date == "2025-10-10"
    assert r.days_over == -1

    t = r.trace
    assert t["rule"]["statute_sentence"] == STATUTE_SENT
    assert t["rule"]["inclusivity_source"] == "discovered"
    assert "beginning with" in t["rule"]["inclusivity_evidence"]
    assert t["anchor_match"]["date"] == "2025-07-12"
    assert t["arithmetic"]["deadline_effective"] == "2025-10-11"
    assert t["arithmetic"]["margin_days"] == -1
    passed = [p["entity"] for p in t["action_selection"]["passed_over"]]
    assert "preliminary hearing" in passed


def test_render_text_is_a_derivation_not_a_verdict(statute, case):
    (r,) = check_entailment(statute, case)
    text = render_text(r)
    assert "DISCOVERED" in text
    assert "beginning with" in text
    assert "2025-10-11" in text
    assert "passed over" in text and "preliminary hearing" in text
    assert "1 day(s) before the deadline" in text
    # D3: the Desk-facing text states the derivation, never stamps a verdict
    assert "TIMELY" not in text and "LATE" not in text


def test_indeterminate_is_diagnosis_with_request(statute):
    empty_case = TemporalDependencyGraph(
        document_id="case_empty", document_type="judgment", source_text="…",
        facts=[_fact("f1", "some meeting", "START",
                     "A meeting took place.", value=None, parsed=None)])
    (r,) = check_entailment(statute, empty_case)
    assert r.verdict == "INDETERMINATE"
    assert "request" in r.trace
    assert "effective date of termination" in r.trace["request"]
    text = render_text(r)
    assert "cannot answer" in text and "→" in text


def test_tolled_period_extension_in_trace(statute, case):
    """A period the statute does not count against its own limit.

    The engine implements the shape; which statute provides it, and what it
    is called, come from the rule pack. UK early conciliation is one
    instance and is no longer named in the engine.
    """
    from tdg_core.entailment import TollingRule, register_tolling

    register_tolling({"label": "early conciliation",
                      "authority": "ERA 1996 s.207B",
                      "floor_after_end": "P1M"})
    try:
        (r,) = check_entailment(statute, case,
                                tolled_from=date(2025, 8, 1),
                                tolled_to=date(2025, 8, 21))
        assert r.tolling_applied
        c = r.trace["tolling"]
        assert c["extension_days"] >= 20
        assert c["label"] == "early conciliation"
        text = render_text(r)
        assert "clock paused" in text
        assert "ERA 1996 s.207B" in text
    finally:
        register_tolling(None)


def test_engine_names_no_statute_for_tolling(statute, case):
    """With no pack declaring one, the trace stays jurisdiction-neutral."""
    from tdg_core.entailment import register_tolling

    register_tolling(None)
    (r,) = check_entailment(statute, case,
                            tolled_from=date(2025, 8, 1),
                            tolled_to=date(2025, 8, 21))
    assert r.tolling_applied
    assert r.trace["tolling"]["label"] == "tolled period"
    assert "conciliation" not in render_text(r).lower()


def test_html_renders(statute, case):
    (r,) = check_entailment(statute, case)
    page = render_html([r])
    assert "<html>" in page and "beginning with" in page


def test_trace_survives_to_dict_roundtrip(statute, case):
    (r,) = check_entailment(statute, case)
    import json
    d = json.loads(json.dumps(r.to_dict()))
    assert d["trace"]["rule"]["inclusivity_source"] == "discovered"
