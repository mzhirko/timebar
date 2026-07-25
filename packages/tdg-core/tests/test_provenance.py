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
