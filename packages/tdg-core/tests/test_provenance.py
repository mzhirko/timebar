"""A quote nobody can check must not be counted as corroboration.

The tool's promise is that every date links to a sentence you can verify.
A shipped example carried a document with no text, offsets of zero, and a
quote that existed only as a string in a JSON file, and it raised another
document's date to "agreed" -- the label that tells a reader to stop
looking. Nothing had ever checked that the sentence was really there.
"""

from __future__ import annotations

from datetime import date

from tdg_core.provenance import (check_bundle, check_document, format_report,
                                 fact_is_supported, unverified_keys)
from tdg_core.tdg import TemporalDependencyGraph, TemporalFact, TimexSpan

QUOTE = "The claimant commenced employment with the company on 3 March 2019."


def fact(fid="f1", sentence=QUOTE, raw="2019-03-03"):
    return TemporalFact(
        id=fid, entity="commencement of employment", role="START",
        timex=TimexSpan(text=raw, timex_type="DATE", value="2019-03-03",
                        start_char=0, end_char=0,
                        date_parsed=date(2019, 3, 3)),
        sentence=sentence)


def doc(did, body, *facts):
    return TemporalDependencyGraph(document_id=did, document_type="legal",
                                   source_text=body, facts=list(facts))


def test_a_quote_present_in_the_document_is_supported():
    assert fact_is_supported(fact(), f"Preamble. {QUOTE} And more.")


def test_a_quote_absent_from_the_document_is_not_supported():
    assert not fact_is_supported(fact(), "A wholly unrelated body of text here.")


def test_a_placeholder_body_supports_nothing():
    """The shipped example used a single ellipsis as its source text."""
    assert not fact_is_supported(fact(), "…")
    assert not fact_is_supported(fact(), "")


def test_a_hard_wrapped_quote_still_matches():
    wrapped = ("The claimant commenced employment with the company\n"
               "on 3 March 2019.")
    assert fact_is_supported(fact(), wrapped)


def test_a_short_document_that_really_contains_the_quote_is_supported():
    """Length is not the test; presence of the quote is."""
    assert fact_is_supported(fact(), QUOTE)


def test_either_quote_field_may_carry_the_sentence():
    """Extractors disagree about which field holds the source sentence."""
    swapped = fact(sentence="Section 5 - Details of claim", raw=QUOTE)
    assert fact_is_supported(swapped, f"Heading. {QUOTE}")


def test_report_names_the_document_that_cannot_be_checked():
    reports = check_bundle({
        "letter": doc("letter", QUOTE, fact()),
        "response": doc("response", "…", fact("f2")),
    })
    assert reports["letter"].fully_supported
    assert not reports["response"].fully_supported
    assert unverified_keys(reports) == {("response", "f2")}
    text = "\n".join(format_report(reports))
    assert "response" in text and "no source text" in text


def test_a_clean_bundle_reports_nothing():
    reports = check_bundle({"letter": doc("letter", QUOTE, fact())})
    assert format_report(reports) == []


def test_counts_are_exposed_per_document():
    rep = check_document(doc("d", QUOTE, fact("a"), fact("b", sentence="absent here")))
    assert rep.total == 2
    assert rep.supported == {"a"} and rep.unsupported == {"b"}


# ── relations: is the period the edge asserts actually in the document? ─

from tdg_core.provenance import (check_relations, check_bundle_relations,
                                 format_relation_report, unsupported_relations)
from tdg_core.tdg import TemporalDependency


def dated(fid, entity, iso, sentence=""):
    y, m, d = (int(x) for x in iso.split("-"))
    return TemporalFact(
        id=fid, entity=entity, role="START",
        timex=TimexSpan(text=iso, timex_type="DATE", value=iso, start_char=0,
                        end_char=0, date_parsed=date(y, m, d)),
        sentence=sentence)


def undated(fid, entity, sentence=""):
    return TemporalFact(
        id=fid, entity=entity, role="END",
        timex=TimexSpan(text="", timex_type="DATE", value=None,
                        start_char=0, end_char=0),
        sentence=sentence)


def graph(body, facts, deps):
    return TemporalDependencyGraph(document_id="d", document_type="legal",
                                   source_text=body, facts=facts,
                                   dependencies=deps)


def edge(expr, delta=None, kind="additive"):
    return TemporalDependency(from_id="a", to_id="b", constraint_type=kind,
                              constraint_expr=expr, delta_days=delta)


REAL_RULE = ("The claim was served on the respondent on 6 October 2025. "
             "The response is due within 28 days of service of the claim.")


def test_a_period_stated_in_the_document_is_supported():
    g = graph(REAL_RULE,
              [dated("a", "service of the claim", "2025-10-06"),
               undated("b", "response deadline")],
              [edge("within 28 days of", 28)])
    (check,) = check_relations(g)
    assert check.supported


def test_a_period_the_document_never_states_is_rejected():
    """The failure that shipped: the model measured the gap between two
    dates it had seen and wrote it down as though it were a rule."""
    body = ("The disciplinary hearing was held on 28 May 2025. "
            "Your employment terminates with effect from 12 July 2025.")
    g = graph(body,
              [dated("a", "hearing", "2025-05-28"),
               dated("b", "employment termination", "2025-07-12")],
              [edge("e3 = e2 + 45 days")])
    (check,) = check_relations(g)
    assert check.verdict == "offset-not-in-source"
    assert "never says so" in check.summary


def test_a_real_period_hung_off_the_wrong_anchor_is_rejected():
    """28 days is genuinely in the text, but it runs from service, not from
    the start of employment six years earlier."""
    g = graph(REAL_RULE,
              [dated("a", "employment", "2019-03-03"),
               undated("b", "response deadline")],
              [edge("28 days", 28)])
    (check,) = check_relations(g)
    assert check.verdict == "anchor-not-in-clause"
    assert "from something else" in check.summary


def test_a_period_spelled_out_in_prose_still_matches_a_digit_edge():
    """A statute writes "three months"; the extracted edge writes "3 months".
    Treating those as unrelated would reject correctly extracted rules."""
    body = ("A complaint shall be presented to the tribunal within the period "
            "of three months beginning with the effective date of termination.")
    g = graph(body,
              [dated("a", "effective date of termination", "2025-07-12"),
               undated("b", "presentation of the complaint")],
              [edge("3 months")])
    (check,) = check_relations(g)
    assert check.supported, check.summary


def test_an_edge_asserting_no_period_is_not_checked():
    """A plain ordering constraint claims no arithmetic and needs no check."""
    g = graph("Following the hearing, the decision was taken.",
              [dated("a", "hearing", "2025-05-28"),
               dated("b", "decision", "2025-06-01")],
              [edge(None, None, kind="ordering")])
    assert check_relations(g) == []


def test_the_report_names_every_unsupported_relation():
    g = graph("Nothing here states any period at all, in any form.",
              [dated("a", "hearing", "2025-05-28"),
               dated("b", "termination", "2025-07-12")],
              [edge("45 days")])
    checks = check_bundle_relations({"d": g})
    assert len(unsupported_relations(checks)) == 1
    text = "\n".join(format_relation_report(checks))
    assert "45 days" in text and "not used for arithmetic" in text


def test_a_clean_bundle_reports_nothing():
    g = graph(REAL_RULE,
              [dated("a", "service of the claim", "2025-10-06"),
               undated("b", "response deadline")],
              [edge("within 28 days of", 28)])
    assert format_relation_report(check_bundle_relations({"d": g})) == []
