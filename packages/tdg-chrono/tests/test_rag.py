"""Grounded temporal context for retrieval-augmented generation.

What makes this worth putting in a prompt is not that it retrieves facts,
but that it refuses to smooth them: a disagreement stays two dates, a
calculated date shows its working, and the block says what it could not
establish so the model is not left to fill the gap.
"""

from __future__ import annotations

from datetime import date

import pytest

from tdg_core.tdg import (TemporalDependency, TemporalDependencyGraph,
                          TemporalFact, TimexSpan)

from tdg_chrono import rag
from tdg_chrono.chronology import build_chronology

LETTER_BODY = ("You commenced employment with the company on 3 March 2019.\n"
               "Your employment terminates with effect from 12 July 2025.\n")
ET1_BODY = ("The claimant's employment terminated with effect from 14 July 2025.\n"
            "The claim was served on the respondent on 6 October 2025.\n"
            "The response is due within 28 days of service of the claim.\n")


def fact(fid, entity, iso, sentence, role="END"):
    parsed = date(*(int(p) for p in iso.split("-"))) if iso else None
    return TemporalFact(
        id=fid, entity=entity, role=role,
        timex=TimexSpan(text=iso or "", timex_type="DATE", value=iso,
                        start_char=0, end_char=0, date_parsed=parsed),
        sentence=sentence)


@pytest.fixture()
def chron():
    letter = TemporalDependencyGraph(
        document_id="dismissal_letter", document_type="legal",
        source_text=LETTER_BODY,
        facts=[fact("f2", "commencement of employment", "2019-03-03",
                    "You commenced employment with the company on 3 March 2019.",
                    role="START"),
               fact("f1", "effective date of termination", "2025-07-12",
                    "Your employment terminates with effect from 12 July 2025.")])
    et1 = TemporalDependencyGraph(
        document_id="et1", document_type="legal", source_text=ET1_BODY,
        facts=[fact("f1", "effective date of termination", "2025-07-14",
                    "The claimant's employment terminated with effect from 14 July 2025."),
               fact("f5", "service of the claim", "2025-10-06",
                    "The claim was served on the respondent on 6 October 2025.",
                    role="START"),
               fact("f4", "response deadline", None,
                    "The response is due within 28 days of service of the claim.")],
        dependencies=[TemporalDependency(
            from_id="f5", to_id="f4", constraint_type="additive",
            constraint_expr="within 28 days of", delta_days=28)])
    return build_chronology({"dismissal_letter": letter, "et1": et1})


def test_a_disagreement_is_not_flattened_to_one_date(chron):
    """The failure mode this exists to prevent: retrieval returns whichever
    passage ranked first and the answer silently picks a side."""
    text = rag.render(rag.select(chron, about="termination"))
    assert "DISPUTED" in text
    assert "2025-07-12" in text and "2025-07-14" in text
    assert "dismissal_letter says" in text and "et1 says" in text


def test_every_date_carries_its_quote_and_document(chron):
    text = rag.render(rag.select(chron))
    assert "You commenced employment with the company on 3 March 2019." in text
    assert "dismissal_letter" in text


def test_a_calculated_date_is_labelled_and_shows_its_working(chron):
    text = rag.render(rag.select(chron))
    assert "calculated, not stated in any document" in text
    assert "within 28 days of" in text


def test_the_block_states_what_it_could_not_establish(chron):
    text = rag.render(rag.select(chron, about="termination"))
    assert "WHAT IS NOT ESTABLISHED" in text
    assert "do not match this question" in text


def test_a_complete_bundle_says_nothing_is_missing(chron):
    text = rag.render(rag.select(chron))
    assert "Nothing. Every fact" in text


def test_the_block_tells_the_model_not_to_fill_gaps(chron):
    text = rag.render(rag.select(chron))
    assert "Do not calculate a date that is not listed" in text
    assert "do not resolve a DISPUTED date" in text


def test_selecting_by_question_narrows_the_facts(chron):
    everything = rag.select(chron)
    narrowed = rag.select(chron, about="termination")
    assert 0 < narrowed.selected < everything.selected


def test_selecting_by_date_window(chron):
    ctx = rag.select(chron, since=date(2025, 10, 1))
    assert ctx.selected >= 1
    assert all(e.date >= date(2025, 10, 1) for e in ctx.events)


def test_a_question_matching_nothing_says_so_rather_than_inventing(chron):
    text = rag.render(rag.select(chron, about="bankruptcy"))
    assert "(no facts match this question)" in text


def test_the_json_form_carries_offsets_for_citation(chron):
    out = rag.to_dict(rag.select(chron, about="termination"))
    version = out["facts"][0]["versions"][0]
    assert {"document", "fact_id", "value", "quote",
            "start_char", "end_char"} <= set(version)
    assert out["not_established"]["unmatched_events"] > 0


def test_the_json_form_keeps_both_sides_of_a_dispute(chron):
    out = rag.to_dict(rag.select(chron, about="termination"))
    disputed = [f for f in out["facts"] if f["status"] == "disputed"]
    assert disputed and len(disputed[0]["versions"]) == 2


# ── retrieval and asking ────────────────────────────────────────────────

def _bundle():
    letter = TemporalDependencyGraph(
        document_id="dismissal_letter", document_type="legal",
        source_text=LETTER_BODY,
        facts=[fact("f1", "effective date of termination", "2025-07-12",
                    "Your employment terminates with effect from 12 July 2025.")])
    et1 = TemporalDependencyGraph(
        document_id="et1", document_type="legal", source_text=ET1_BODY,
        facts=[fact("f1", "effective date of termination", "2025-07-14",
                    "The claimant's employment terminated with effect from 14 July 2025.")])
    return {"dismissal_letter": letter, "et1": et1}


def test_retrieval_finds_the_relevant_sentence():
    hits = rag.retrieve(_bundle(), "when did employment terminate?", limit=3)
    assert hits
    assert any("terminates with effect" in h.text or "terminated with effect"
               in h.text for h in hits)


def test_every_passage_carries_its_document_and_offsets():
    """An answer has to be able to point at the sentence, not the file."""
    for p in rag.retrieve(_bundle(), "termination", limit=3):
        assert p.doc_id
        source = _bundle()[p.doc_id].source_text
        assert source[p.start_char:p.end_char].strip() == p.text


def test_a_question_matching_nothing_retrieves_nothing():
    assert rag.retrieve(_bundle(), "shipping containers tariff", limit=5) == []


def test_the_prompt_puts_established_dates_before_the_prose(chron):
    """A model that reads the computed dates first is far less likely to
    pull a date out of a sentence and do its own arithmetic on it."""
    prompt, _, _ = rag.build_prompt(chron, _bundle(), "when did employment end?")
    assert prompt.index("TEMPORAL FACTS") < prompt.index("PASSAGES")
    assert prompt.rstrip().endswith("QUESTION: when did employment end?")


def test_the_prompt_forbids_resolving_a_dispute(chron):
    prompt, _, _ = rag.build_prompt(chron, _bundle(), "termination date?")
    assert "DISPUTED" in prompt
    assert "do not resolve a DISPUTED date" in prompt


class _StubClient:
    """Stands in for a model, so these tests need nothing running."""

    def __init__(self):
        self.seen = None
        self.chat = self

    @property
    def completions(self):
        return self

    def create(self, *, model, messages, **kw):
        self.seen = messages
        text = "answer text"

        class _M:
            content = text
        class _C:
            message = _M()
        class _R:
            choices = [_C()]
        return _R()


def test_ask_returns_the_answer_with_what_it_rested_on(chron):
    stub = _StubClient()
    out = rag.ask(chron, _bundle(), "when did employment end?",
                  model="stub", client=stub)
    assert out["answer"] == "answer text"
    assert out["facts_used"]["facts"], "the answer should carry its dates"
    assert out["passages_used"], "and the passages it was given"
    assert out["question"] == "when did employment end?"


def test_ask_instructs_the_model_to_take_dates_from_the_engine(chron):
    stub = _StubClient()
    rag.ask(chron, _bundle(), "termination?", model="stub", client=stub)
    system = stub.seen[0]["content"]
    assert "Take every date from the TEMPORAL FACTS" in system
    assert "do not do arithmetic on dates yourself" in system
    assert "Never pick one" in system
