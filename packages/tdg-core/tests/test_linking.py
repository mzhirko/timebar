"""Cross-document linking by evidence composition.

Two failures bound this module, and both are represented here.

Missing a link: a dismissal letter says "employment termination" ends on
12 July and the ET1 says "employment" ended on 14 July. Under the gated
linker these never met, because the extractor's quote for one of them was
useless and the sentence-overlap gate vetoed the match. The two-day gap
that decides whether a claim is in time never reached the timeline.

Inventing a link: two dismissal letters for different clients share almost
every word, because they share a form. Merging those is worse than missing
a match — it fabricates a dispute between two people who have never met.
"""

from __future__ import annotations

from datetime import date

from tdg_core.linking import (
    BundleStatistics,
    EventLinker,
    document_relatedness,
    temporal_proximity,
    tokenise,
)
from tdg_core.tdg import TemporalDependencyGraph, TemporalFact, TimexSpan


def fact(fid: str, entity: str, role: str, iso: str, sentence: str = "") -> TemporalFact:
    y, m, d = (int(x) for x in iso.split("-"))
    return TemporalFact(
        id=fid, entity=entity, role=role,
        timex=TimexSpan(text=iso, timex_type="DATE", value=iso,
                        start_char=0, end_char=0, date_parsed=date(y, m, d)),
        sentence=sentence,
    )


def doc(doc_id: str, *facts: TemporalFact) -> TemporalDependencyGraph:
    return TemporalDependencyGraph(document_id=doc_id, document_type="legal",
                                   source_text="", facts=list(facts))


# One matter, two documents, disagreeing by two days.
LETTER = doc("dismissal_letter",
             fact("a1", "employment", "START", "2019-03-03",
                  "Ms Okafor commenced employment with Northgate on 3 March 2019."),
             fact("a2", "employment termination", "END", "2025-07-12",
                  "Your employment terminates with effect from 12 July 2025."))

ET1 = doc("et1_claim",
          fact("b1", "employment", "START", "2019-03-03",
               "Ms Okafor was employed by Northgate from 3 March 2019."),
          fact("b2", "employment", "END", "2025-07-14",
               "Section 5 - Details of claim"))

# A different client, a different employer, a different decade.
OTHER_MATTER = doc("brennan_dismissal",
                   fact("c1", "employment", "START", "2004-09-15",
                        "Mr Brennan commenced employment with Halverd Foods."),
                   fact("c2", "employment termination", "END", "2011-01-20",
                        "Mr Brennan's employment with Halverd Foods terminates."))


def _links(*docs):
    tdgs = {d.document_id: d for d in docs}
    return EventLinker(tdgs).link_all(), tdgs


# ── the link that was being missed ──────────────────────────────────────

def test_links_refined_entity_name_to_its_shorter_form():
    """'employment termination' and 'employment' are one event, two names."""
    links, _ = _links(LETTER, ET1)
    matched = {(fa.id, fb.id) for _, fa, _, fb, _, _ in links}
    assert ("a2", "b2") in matched


def test_links_despite_a_useless_quote():
    """The ET1's quote carries no overlap; the match must survive anyway.

    This is the specific veto that made the gated linker miss real disputes:
    quote quality is the extractor's choice, not evidence about the world.
    """
    links, _ = _links(LETTER, ET1)
    pair = [ev for _, fa, _, fb, ev, _ in links if (fa.id, fb.id) == ("a2", "b2")]
    assert pair, "pair must link"
    assert pair[0].sentence < 0.1, "fixture should have no sentence overlap"


def test_matched_dates_two_days_apart_are_kept_as_a_disagreement():
    links, _ = _links(LETTER, ET1)
    pair = [(fa, fb) for _, fa, _, fb, _, _ in links if (fa.id, fb.id) == ("a2", "b2")]
    fa, fb = pair[0]
    assert fa.timex.date_parsed != fb.timex.date_parsed


# ── the link that must never be invented ────────────────────────────────

def test_unrelated_matters_never_link():
    """Same form, same vocabulary, different people. Must not merge."""
    links, _ = _links(LETTER, OTHER_MATTER)
    assert links == []


def test_unrelated_matters_still_isolated_inside_a_larger_bundle():
    """Adding the unrelated file must not contaminate the real matter."""
    links, _ = _links(LETTER, ET1, OTHER_MATTER)
    docs_linked = {(da, db) for da, _, db, _, _, _ in links}
    assert ("brennan_dismissal", "dismissal_letter") not in docs_linked
    assert ("brennan_dismissal", "et1_claim") not in docs_linked
    assert ("dismissal_letter", "et1_claim") in docs_linked


def test_dates_years_apart_are_not_one_event():
    tdgs = {d.document_id: d for d in (LETTER, OTHER_MATTER)}
    linker = EventLinker(tdgs)
    assert linker.temporally_incompatible(LETTER.facts[1], OTHER_MATTER.facts[1])
    assert not linker.temporally_incompatible(LETTER.facts[0], LETTER.facts[0])


# ── one-to-one competition ──────────────────────────────────────────────

def test_a_fact_links_to_at_most_one_fact_per_document():
    """Competition is what makes a permissive threshold safe."""
    crowded = doc("crowded",
                  fact("d1", "employment", "END", "2025-07-13", "one"),
                  fact("d2", "employment", "END", "2025-07-15", "two"))
    links, _ = _links(LETTER, crowded)
    from_ids = [fa.id for _, fa, _, _, _, _ in links]
    to_ids = [fb.id for _, _, _, fb, _, _ in links]
    assert len(from_ids) == len(set(from_ids))
    assert len(to_ids) == len(set(to_ids))


# ── the pieces, individually ────────────────────────────────────────────

def test_tokenise_splits_on_punctuation_and_underscores():
    assert tokenise("disciplinary_hearing") == ["disciplinary", "hearing"]
    assert tokenise("Section 5 - Details") == ["section", "5", "details"]


def test_idf_ranks_a_rare_word_above_a_universal_one():
    stats = BundleStatistics(["employment termination okafor",
                              "employment termination brennan",
                              "employment termination whitaker"])
    assert stats.idf("okafor") > stats.idf("employment")
    assert stats.is_universal("employment")
    assert not stats.is_universal("okafor")


def test_temporal_proximity_decays_with_distance():
    near = temporal_proximity(fact("x", "e", "END", "2025-07-12"),
                              fact("y", "e", "END", "2025-07-14"))
    far = temporal_proximity(fact("x", "e", "END", "2025-07-12"),
                             fact("y", "e", "END", "2011-01-20"))
    assert near > 0.9
    assert far < 0.01


def test_temporal_proximity_is_none_when_incomparable():
    undated = TemporalFact(id="u", entity="e", role="END",
                           timex=TimexSpan(text="later", timex_type="DATE",
                                           value=None, start_char=0, end_char=0),
                           sentence="")
    assert temporal_proximity(undated, fact("y", "e", "END", "2025-07-14")) is None


def test_related_documents_outscore_unrelated_ones():
    related = {d.document_id: d for d in (LETTER, ET1)}
    unrelated = {d.document_id: d for d in (LETTER, OTHER_MATTER)}

    lr = EventLinker(related)
    ur = EventLinker(unrelated)
    r_score = document_relatedness(lr.profiles["dismissal_letter"],
                                   lr.profiles["et1_claim"], lr.stats).score
    u_score = document_relatedness(ur.profiles["dismissal_letter"],
                                   ur.profiles["brennan_dismissal"], ur.stats).score
    assert r_score > u_score


def test_bar_is_flat_by_default():
    """Sliding the bar by relatedness was measured and changed nothing.

    Identical results to a fixed bar on every bundle tested, and at two
    documents the relatedness estimate is actively misleading. The default
    is therefore flat; the slide stays configurable for corpora where
    relatedness is trustworthy.
    """
    linker = EventLinker({d.document_id: d for d in (LETTER, ET1)})
    strict = linker.score_pair(LETTER.facts[1], ET1.facts[1], relatedness=0.0)
    relaxed = linker.score_pair(LETTER.facts[1], ET1.facts[1], relatedness=1.0)
    assert relaxed.threshold == strict.threshold


def test_bar_can_still_be_slid_when_configured():
    linker = EventLinker({d.document_id: d for d in (LETTER, ET1)},
                         unrelated_threshold=0.80, related_threshold=0.30)
    strict = linker.score_pair(LETTER.facts[1], ET1.facts[1], relatedness=0.0)
    relaxed = linker.score_pair(LETTER.facts[1], ET1.facts[1], relatedness=1.0)
    assert relaxed.threshold < strict.threshold
    assert relaxed.score == strict.score  # relatedness moves the bar, not the evidence


# ── matter identity: declared, never inferred ───────────────────────────

def test_documents_in_different_matters_are_never_linked():
    a = doc("okafor_letter",
            fact("m1", "employment termination", "END", "2025-07-12",
                 "Ms Okafor's employment terminates."))
    b = doc("brennan_letter",
            fact("m2", "employment termination", "END", "2025-07-19",
                 "Mr Brennan's employment terminates."))
    a.matter, b.matter = "OKAFOR/2025", "BRENNAN/2025"
    linker = EventLinker({d.document_id: d for d in (a, b)})
    assert linker.link_all() == []
    assert [(a, b) for a, b, _ in linker.blocked_pairs] == \
        [("brennan_letter", "okafor_letter")]
    assert "different declared matters" in linker.blocked_pairs[0][2]


def test_same_declared_matter_links_normally():
    a = doc("letter", fact("m1", "employment termination", "END", "2025-07-12", "x"))
    b = doc("et1", fact("m2", "employment", "END", "2025-07-14", "y"))
    a.matter = b.matter = "OKAFOR/2025"
    assert EventLinker({d.document_id: d for d in (a, b)}).link_all()


def test_absent_matter_links_freely():
    """No declaration is not a declaration of difference.

    The operator put these files in one bundle; that choice is better
    evidence than anything derivable from the text.
    """
    a = doc("letter", fact("m1", "employment termination", "END", "2025-07-12", "x"))
    b = doc("et1", fact("m2", "employment", "END", "2025-07-14", "y"))
    assert a.matter is None and b.matter is None
    assert EventLinker({d.document_id: d for d in (a, b)}).link_all()


def test_one_sided_declaration_does_not_block():
    a = doc("letter", fact("m1", "employment termination", "END", "2025-07-12", "x"))
    b = doc("et1", fact("m2", "employment", "END", "2025-07-14", "y"))
    a.matter = "OKAFOR/2025"
    linker = EventLinker({d.document_id: d for d in (a, b)})
    assert linker.link_all()
    assert linker.matter_coverage == (1, 0, 2)


def test_every_link_carries_its_signals():
    """A merge a reviewer cannot interrogate is not usable in this domain."""
    links, _ = _links(LETTER, ET1)
    for _, _, _, _, ev, rel in links:
        text = str(ev)
        assert "entity=" in text and "sentence=" in text and "time=" in text
        assert 0.0 <= rel.score <= 1.0


# ── the bar follows the configuration ───────────────────────────────────

def test_bar_rises_when_an_embedder_is_configured():
    """An embedder lifts every score, wrong pairs included.

    Measured over labelled pairs, 0.50 gives 83%/83% recall/precision
    lexically and 100%/55% with nomic-embed-text. The bar has to move with
    the configuration or the embedder trades precision away silently.
    """
    from tdg_core.linking import DEFAULT_EMBEDDED_THRESHOLD, DEFAULT_THRESHOLD

    class Stub:
        def similarity(self, a, b):
            return 0.9

    tdgs = {d.document_id: d for d in (LETTER, ET1)}
    assert EventLinker(tdgs).unrelated_threshold == DEFAULT_THRESHOLD
    assert (EventLinker(tdgs, embedder=Stub()).unrelated_threshold
            == DEFAULT_EMBEDDED_THRESHOLD)
    assert DEFAULT_EMBEDDED_THRESHOLD > DEFAULT_THRESHOLD


def test_explicit_threshold_still_wins_over_the_default():
    class Stub:
        def similarity(self, a, b):
            return 0.9

    linker = EventLinker({d.document_id: d for d in (LETTER, ET1)},
                         embedder=Stub(), unrelated_threshold=0.31,
                         related_threshold=0.31)
    assert linker.unrelated_threshold == 0.31


def test_a_broken_embedder_falls_back_to_lexical():
    """An unreachable embedding service must not fail the run."""
    class Broken:
        def similarity(self, a, b):
            raise RuntimeError("connection refused")

    linker = EventLinker({d.document_id: d for d in (LETTER, ET1)},
                         embedder=Broken())
    assert linker.entity_similarity("employment", "employment") > 0


# ── parties: extracted, then compared as sets ───────────────────────────

def _party_docs(pa, pb, iso_a="2025-07-12", iso_b="2025-07-19"):
    a = doc("doc_a", fact("p1", "employment termination", "END", iso_a, "x"))
    b = doc("doc_b", fact("p2", "employment termination", "END", iso_b, "y"))
    a.parties, b.parties = pa, pb
    return EventLinker({d.document_id: d for d in (a, b)})


def test_disjoint_parties_block_the_link():
    """The failure a threshold cannot reach: same words, different clients."""
    linker = _party_docs(["Ms A. Okafor", "Northgate Logistics Ltd"],
                         ["Mr B. Brennan", "Halverd Foods Ltd"])
    assert linker.link_all() == []
    assert "no party in common" in linker.blocked_pairs[0][2]


def test_shared_party_permits_the_link():
    linker = _party_docs(["Ms A. Okafor", "Northgate Logistics Ltd"],
                         ["Northgate Logistics Ltd", "Ms A. Okafor"])
    assert linker.link_all()


def test_a_surname_matches_the_full_name():
    """One document gives the full name, another only the surname."""
    linker = _party_docs(["Ms A. Okafor"], ["Okafor"])
    assert linker.link_all()


def test_organisation_suffix_is_ignored():
    linker = _party_docs(["Northgate Logistics Ltd"], ["Northgate Logistics"])
    assert linker.link_all()


def test_a_shared_honorific_is_not_a_shared_party():
    """'Ms' in common must not make two strangers the same matter."""
    linker = _party_docs(["Ms A. Smith"], ["Ms B. Jones"])
    assert linker.link_all() == []


def test_declared_matter_outranks_parties():
    a = doc("doc_a", fact("p1", "employment termination", "END", "2025-07-12", "x"))
    b = doc("doc_b", fact("p2", "employment termination", "END", "2025-07-14", "y"))
    a.parties, b.parties = ["Ms A. Okafor"], ["Mr B. Brennan"]
    a.matter = b.matter = "CONSOLIDATED/2025"
    linker = EventLinker({d.document_id: d for d in (a, b)})
    assert linker.link_all(), "an explicit matter must win over party names"


def test_naming_no_parties_never_blocks():
    linker = _party_docs([], [])
    assert linker.link_all()


def test_one_document_naming_parties_does_not_block():
    linker = _party_docs(["Ms A. Okafor"], [])
    assert linker.link_all()


def test_party_key_strips_titles_and_initials():
    from tdg_core.linking import party_key
    assert party_key("Ms A. Okafor") == party_key("Okafor")
    assert party_key("Northgate Logistics Ltd") == party_key("Northgate Logistics")
    assert party_key("Mr") == frozenset()


# ── exclusivity is per event, not per fact ──────────────────────────────

def test_a_restated_event_shares_one_slot():
    """A document stating one event twice must not strand the second mention.

    One-fact-per-document matching gave the first mention the link and left
    the restatement behind as its own single-source row.
    """
    letter = doc("letter", fact("a1", "employment termination", "END",
                                "2025-07-12", "terminates 12 July"))
    et1 = doc("et1",
              fact("b1", "employment termination", "END", "2025-07-14",
                   "dismissed 14 July"),
              fact("b2", "employment termination", "END", "2025-07-14",
                   "Schedule 1: dismissal 14 July"))
    links, _ = _links(letter, et1)
    # link_all sorts document ids, so either side may hold the ET1 fact.
    touched = {f.id for _, fa, _, fb, _, _ in links for f in (fa, fb)}
    assert {"b1", "b2"} <= touched, "both restatements must join the row"


def test_two_different_dates_in_one_document_stay_separate():
    """Grouping restatements must not merge genuinely distinct events."""
    et1 = doc("et1",
              fact("b1", "hearing", "CONTAINS", "2025-05-28", "first hearing"),
              fact("b2", "hearing", "CONTAINS", "2025-06-30", "second hearing"))
    linker = EventLinker({"et1": et1})
    groups = linker.within_document_groups("et1")
    assert groups["b1"] != groups["b2"]


def test_restatements_group_together():
    et1 = doc("et1",
              fact("b1", "employment termination", "END", "2025-07-14", "one"),
              fact("b2", "employment termination", "END", "2025-07-14", "two"))
    linker = EventLinker({"et1": et1})
    groups = linker.within_document_groups("et1")
    assert groups["b1"] == groups["b2"]


# ── statistics are scoped to the matter, not the folder ─────────────────

def _okafor_pair():
    a = doc("dismissal_letter",
            fact("e2", "hearing", "START", "2025-05-28",
                 "the outcome of the disciplinary hearing held on 28 May 2025"))
    b = doc("et1_claim",
            fact("e2", "disciplinary hearing", "CONTAINS", "2025-05-28",
                 "A disciplinary hearing took place on 28 May 2025."))
    a.parties, b.parties = ["Ms Okafor"], ["Ms A. Okafor"]
    return a, b


def _stranger(did, party, iso):
    d = doc(did, fact("z1", "employment termination", "END", iso, "terminates"))
    d.parties = [party]
    return d


def test_an_unrelated_file_does_not_move_a_score():
    """A merge must not depend on what else the directory happens to hold.

    Token weight is measured over a corpus; with one shared corpus an
    unrelated case shifted every score enough to flip a borderline pair.
    """
    a, b = _okafor_pair()
    alone = EventLinker({d.document_id: d for d in (a, b)})
    crowded = EventLinker({d.document_id: d for d in
                           (a, b, _stranger("brennan", "Mr Brennan", "2025-07-19"),
                            _stranger("lease", "Acme Realty Plc", "2011-02-01"))})

    def score(linker):
        rel = document_relatedness(linker.profiles["dismissal_letter"],
                                   linker.profiles["et1_claim"],
                                   linker.stats_for("dismissal_letter"))
        return linker.score_pair(a.facts[0], b.facts[0], rel.score,
                                 linker.stats_for("dismissal_letter")).score

    assert score(alone) == score(crowded)


def test_matters_partition_into_separate_groups():
    a, b = _okafor_pair()
    linker = EventLinker({d.document_id: d for d in
                          (a, b, _stranger("brennan", "Mr Brennan", "2025-07-19"))})
    groups = sorted(sorted(docs) for docs in linker.matter_groups.values())
    assert groups == [["brennan"], ["dismissal_letter", "et1_claim"]]


def test_statistics_differ_between_matter_groups():
    a, b = _okafor_pair()
    linker = EventLinker({d.document_id: d for d in
                          (a, b, _stranger("brennan", "Mr Brennan", "2025-07-19"))})
    assert linker.stats_for("dismissal_letter") is linker.stats_for("et1_claim")
    assert linker.stats_for("brennan") is not linker.stats_for("dismissal_letter")


def test_dispute_allowance_scales_to_the_matter_not_the_folder():
    """An old unrelated file must not widen what counts as one event."""
    a, b = _okafor_pair()
    with_old = EventLinker({d.document_id: d for d in
                            (a, b, _stranger("lease", "Acme Realty Plc", "1995-01-01"))})
    assert (with_old.max_dispute_days_for("dismissal_letter")
            < with_old.max_dispute_days)


def test_undeclared_documents_share_one_group():
    """With nothing to separate them, every document is one matter."""
    a, b = _okafor_pair()
    a.parties, b.parties = [], []
    linker = EventLinker({d.document_id: d for d in (a, b)})
    assert len(linker.matter_groups) == 1


# ── the quote signal follows the evidence, not the field name ───────────

def test_the_quote_is_whichever_text_states_the_date():
    """Extractors disagree about which field holds the source sentence.

    One model puts the sentence in `sentence` and the value in `raw_text`;
    another puts the sentence in `raw_text` and a section heading in
    `sentence`. Comparing the nominal field compares a heading with a
    sentence and calls it evidence.
    """
    from tdg_core.linking import supporting_quote

    swapped = TemporalFact(
        id="x", entity="disciplinary hearing", role="CONTAINS",
        timex=TimexSpan(text="A disciplinary hearing took place on 28 May 2025.",
                        timex_type="DATE", value="2025-05-28", start_char=0,
                        end_char=0, date_parsed=date(2025, 5, 28)),
        sentence="Section 5 - Details of claim")
    quote, ok = supporting_quote(swapped)
    assert ok
    assert quote.startswith("A disciplinary hearing")


def test_a_bare_value_is_not_treated_as_a_quote():
    """Comparing "2025-07-12" with "2025-07-14" is not corroboration."""
    from tdg_core.linking import supporting_quote

    f = TemporalFact(
        id="x", entity="employment termination", role="END",
        timex=TimexSpan(text="2025-07-12", timex_type="DATE",
                        value="2025-07-12", start_char=0, end_char=0,
                        date_parsed=date(2025, 7, 12)),
        sentence="")
    quote, ok = supporting_quote(f)
    assert not ok
    assert quote == ""


def test_a_good_sentence_is_preferred_and_marked_supported():
    from tdg_core.linking import supporting_quote

    f = fact("x", "employment termination", "END", "2025-07-12",
             "Your employment terminates with effect from 12 July 2025.")
    quote, ok = supporting_quote(f)
    assert ok and quote.startswith("Your employment")


def test_an_unverifiable_quote_still_counts_against_the_match():
    """Dropping the signal was measured and made precision worse.

    For two facts with near-identical names and adjacent dates the quote
    comparison is the only thing left telling them apart.
    """
    a = doc("a", fact("f1", "hearing", "CONTAINS", "2025-05-28", "no date here"))
    b = doc("b", fact("f2", "hearing", "CONTAINS", "2025-05-28", "nor here"))
    linker = EventLinker({d.document_id: d for d in (a, b)})
    ev = linker.score_pair(a.facts[0], b.facts[0], 0.0)
    assert ev.sentence is not None
