"""Cross-document event linking by evidence composition.

The problem this replaces
-------------------------
Coreference used two hard AND-gates: an entity-name similarity floor and a
source-sentence overlap floor. Any single weak signal vetoed the merge, so a
correct link died whenever one signal happened to be poor — most often the
quote, which is whatever the extractor chose to emit. In a real bundle,
"employment termination" in a dismissal letter and "employment" in the ET1
scored 0.50 on names and 0.00 on sentences, and were never linked, so the
two-day discrepancy that moves a limitation deadline never surfaced.

Raising or lowering a global threshold cannot fix that: lowering it to admit
the true pair also admits every unrelated pair sharing a generic word.

The approach
------------
Three ideas, none of which encode anything about a particular case, statute,
or jurisdiction.

1. **Discriminative weight is measured, not declared.** Token importance
   comes from inverse document frequency over the bundle at hand. A word
   appearing in most documents carries little evidence; a rare one carries
   a lot. This is what a hand-maintained stopword list and boilerplate
   regex were approximating, except it adapts to any corpus in any language
   and needs no maintenance.

2. **Document relatedness sets the bar for fact relatedness.** Documents are
   profiled by what distinguishes them — their rare shared vocabulary and
   the period they cover. Two documents about the same matter earn a lower
   fact-level threshold; unrelated documents that happen to share a common
   word earn nothing. Sharing a *kind* is not relatedness: "both are
   contracts" relates nothing, while "same participants over an overlapping
   period" does.

3. **Competition disambiguates, so the threshold can be low.** Rather than
   judging each pair in isolation, each document pair forms a bipartite
   graph whose edges are weighted by composed evidence, and only a mutual
   best match survives. A document almost never states the same event twice
   with different dates, so one-to-one matching is a strong and general
   constraint. It is what makes a permissive threshold safe: a weak pair
   only links if nothing else competes for either side.

Signals compose by weighted mean over whatever is computable, rather than by
veto. A missing signal reduces confidence; it does not annihilate the match.

Everything here is deterministic and inspectable: every link carries the
signal values that produced it. An embedder may be supplied to improve
entity similarity, and its absence changes scores but not behaviour.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from typing import Iterable, Optional, Protocol

from tdg_core.tdg import TemporalDependencyGraph, TemporalFact
from tdg_core.text_cleaner import mentions_date


# ─── Tokenisation ─────────────────────────────────────────────────────────

def tokenise(text: str) -> list[str]:
    """Split on non-alphanumerics and casefold.

    Deliberately naive and language-agnostic: no stemming, no stopword list,
    no domain vocabulary. Everything that distinguishes a useful token from a
    useless one is learned from the bundle by BundleStatistics.
    """
    out, current = [], []
    for ch in text:
        if ch.isalnum():
            current.append(ch.lower())
        elif current:
            out.append("".join(current))
            current = []
    if current:
        out.append("".join(current))
    return out


class Embedder(Protocol):
    """Anything exposing similarity(a, b) -> float in [0, 1]."""

    def similarity(self, a: str, b: str) -> float: ...


# ─── Party names ──────────────────────────────────────────────────────────

def _load_name_particles() -> frozenset[str]:
    """Honorifics and organisation-form suffixes, from data, not code."""
    import json
    from importlib import resources
    try:
        with resources.files("tdg_core.data").joinpath(
                "name_particles.en.json").open() as fh:
            data = json.load(fh)
    except (FileNotFoundError, ModuleNotFoundError):  # pragma: no cover
        return frozenset()
    return frozenset(t.lower() for key in
                     ("person_titles", "org_suffixes", "generic")
                     for t in data.get(key, []))


_NAME_PARTICLES = _load_name_particles()


def party_key(name: str) -> frozenset[str]:
    """The identifying tokens of a party name.

    "Ms A. Okafor" and "Okafor" reduce to the same key; "Northgate Logistics
    Ltd" and "Northgate Logistics" likewise. Initials are dropped with the
    honorifics: a single letter distinguishes nobody, and keeping it would
    make "A. Smith" and "B. Smith" look like different people when the
    surname is all either document gives.
    """
    tokens = {t for t in tokenise(name)
              if len(t) > 1 and t not in _NAME_PARTICLES and not t.isdigit()}
    return frozenset(tokens)


def _normalise_parties(parties: Iterable[str]) -> set[frozenset[str]]:
    """Party names as comparable keys, dropping any that reduce to nothing."""
    keys = {party_key(p) for p in parties}
    return {k for k in keys if k}


def parties_overlap(a: Iterable[str], b: Iterable[str]) -> set[str]:
    """Names appearing on both sides, as readable strings.

    Two keys count as the same party when either is a subset of the other,
    so a document giving only a surname still matches one giving the full
    name. Requiring equality would treat every abbreviation as a stranger.
    """
    ka, kb = _normalise_parties(a), _normalise_parties(b)
    shared: set[str] = set()
    for x in ka:
        for y in kb:
            if x <= y or y <= x:
                shared.add(" ".join(sorted(x if len(x) <= len(y) else y)))
    return shared


# ─── Corpus statistics ────────────────────────────────────────────────────

class BundleStatistics:
    """Inverse document frequency over the bundle currently being linked.

    This is the piece that removes the need for hand-written stopwords and
    boilerplate patterns. In a bundle of employment documents, "employment"
    appears everywhere and is nearly worthless for telling two events apart,
    while "redundancy" appears once and is highly diagnostic — and in a
    bundle of shipping contracts the same logic elects entirely different
    words, with no code change.
    """

    def __init__(self, documents: Iterable[str]) -> None:
        docs = list(documents)
        self.n_docs = len(docs)
        self._df: Counter[str] = Counter()
        for text in docs:
            for token in set(tokenise(text)):
                self._df[token] += 1

    def idf(self, token: str) -> float:
        """Smoothed IDF. Unseen tokens get the maximum weight."""
        if self.n_docs == 0:
            return 1.0
        df = self._df.get(token, 0)
        return math.log((self.n_docs + 1) / (df + 1)) + 1.0

    def weighted_similarity(self, a: str, b: str) -> float:
        """IDF-weighted overlap of two short strings, in [0, 1].

        A weighted Jaccard: shared evidence over total evidence, where each
        token contributes its own discriminative weight rather than 1.
        """
        ta, tb = set(tokenise(a)), set(tokenise(b))
        if not ta or not tb:
            return 0.0
        shared = ta & tb
        union = ta | tb
        num = sum(self.idf(t) for t in shared)
        den = sum(self.idf(t) for t in union)
        return num / den if den else 0.0

    def weighted_containment(self, a: str, b: str) -> float:
        """Mean IDF-weighted containment of each name in the other, in [0, 1].

        Recognises refinement, which Jaccard punishes. Documents name the
        same event at different granularity all the time — a letter says
        "employment termination" where a pleading says "employment" — and
        under symmetric overlap the extra word counts against a match it
        should support. Averaging the two directions keeps a genuine
        refinement well short of an identical name, so "employment" does not
        become interchangeable with everything it prefixes.
        """
        ta, tb = set(tokenise(a)), set(tokenise(b))
        if not ta or not tb:
            return 0.0
        shared_weight = sum(self.idf(t) for t in (ta & tb))
        wa = sum(self.idf(t) for t in ta)
        wb = sum(self.idf(t) for t in tb)
        if not wa or not wb:
            return 0.0
        return (shared_weight / wa + shared_weight / wb) / 2

    def is_universal(self, token: str) -> bool:
        """True if the token appears in every document of the bundle.

        Such a token cannot distinguish one document from another, so it
        carries no evidence that two documents concern the same matter. This
        is what stops a bundle of unrelated dismissal letters from looking
        related: "employment", "termination" and "dismissal" appear in all
        of them precisely because they share a *form*, and sharing a form is
        not a relationship.
        """
        return self.n_docs > 0 and self._df.get(token, 0) >= self.n_docs

    def discriminative_tokens(self, text: str, *, top: int = 12) -> set[str]:
        """The most distinguishing tokens of a document.

        Used as a proxy for "who and what this document is about" —
        participant names, references and matter-specific nouns rise to the
        top by construction, without naming any of them in code. Tokens
        common to the whole bundle are excluded: they describe the genre,
        not the matter.
        """
        counts = Counter(tokenise(text))
        scored = sorted(counts,
                        key=lambda t: -(self.idf(t) * (1 + math.log(counts[t]))))
        return set(scored[:top])

    def bundle_span_days(self, spans: Iterable[tuple[date, date]]) -> int:
        """Total period the bundle covers, used to scale what counts as near."""
        lows, highs = [], []
        for lo, hi in spans:
            lows.append(lo)
            highs.append(hi)
        if not lows:
            return 0
        return max(0, (max(highs) - min(lows)).days)


# ─── Document profiles ────────────────────────────────────────────────────

@dataclass
class DocumentProfile:
    """What a document is about, in terms usable for comparison."""

    doc_id: str
    tokens: set[str]
    discriminative: set[str]
    span: Optional[tuple[date, date]]

    @property
    def has_span(self) -> bool:
        return self.span is not None


def profile_document(tdg: TemporalDependencyGraph,
                     stats: BundleStatistics) -> DocumentProfile:
    """Summarise a document by its distinguishing vocabulary and date span."""
    text = document_text(tdg)
    dates = [f.timex.date_parsed for f in tdg.facts if f.timex.date_parsed]
    span = (min(dates), max(dates)) if dates else None
    return DocumentProfile(
        doc_id=tdg.document_id,
        tokens=set(tokenise(text)),
        discriminative=stats.discriminative_tokens(text),
        span=span,
    )


def document_text(tdg: TemporalDependencyGraph) -> str:
    """The text a document contributes to corpus statistics.

    Entities and source sentences, not raw document text: these are the
    surfaces linking actually compares, so they are what should be weighted.
    """
    parts = [tdg.document_id.replace("_", " "), tdg.document_type or ""]
    for f in tdg.facts:
        parts.append(f.entity or "")
        parts.append(f.sentence or "")
    return " ".join(p for p in parts if p)


def _span_compatibility(a: tuple[date, date], b: tuple[date, date],
                        *, half_life_days: float = 365.0) -> float:
    """Do two documents cover the same period? In [0, 1].

    Intersecting spans score 1.0; disjoint spans decay with the gap between
    them. Plain interval Jaccard is wrong here because a document quoting a
    single date has a zero-length span, which makes every intersection zero
    even when that date sits squarely inside the other document's period —
    the common case for a letter cited by a pleading.
    """
    if a[0] <= b[1] and b[0] <= a[1]:
        return 1.0
    gap = (b[0] - a[1]).days if b[0] > a[1] else (a[0] - b[1]).days
    return 0.5 ** (max(0, gap) / half_life_days)


@dataclass
class Relatedness:
    """How strongly two documents look like parts of the same matter."""

    score: float
    shared_terms: set[str] = field(default_factory=set)
    span_overlap: float = 0.0

    @property
    def explanation(self) -> str:
        terms = ", ".join(sorted(self.shared_terms)[:5]) or "none"
        return (f"shared terms: {terms}; "
                f"period overlap: {self.span_overlap:.2f}")


def document_relatedness(a: DocumentProfile, b: DocumentProfile,
                         stats: BundleStatistics) -> Relatedness:
    """Evidence that two documents concern the same matter.

    Two independent kinds of evidence, combined so that neither alone is
    conclusive: distinguishing vocabulary in common (roughly, the same
    people and references) and an overlapping period. Documents of the same
    *type* with nothing else in common score near zero, which is the point —
    a bundle of unrelated dismissal letters must not cross-link.
    """
    def containment(src: DocumentProfile, dst: DocumentProfile) -> float:
        """How much of src's distinguishing vocabulary appears in dst at all.

        Containment rather than set intersection, because the informative
        question is not "do their summaries coincide" but "does what makes
        this document specific — the party, the employer, the reference —
        turn up in the other document". Two dismissal letters for different
        clients each carry names the other never mentions, and score low
        however alike their boilerplate. Two documents about one matter name
        the same people, and score high.
        """
        if not src.discriminative:
            return 0.0
        total = sum(stats.idf(t) for t in src.discriminative)
        found = sum(stats.idf(t) for t in src.discriminative if t in dst.tokens)
        return found / total if total else 0.0

    # Symmetric: a pleading citing a letter, and the letter, should relate
    # equally in both directions.
    lexical = (containment(a, b) + containment(b, a)) / 2
    shared = {t for t in (a.discriminative & b.discriminative)
              if not stats.is_universal(t)}

    overlap = (_span_compatibility(a.span, b.span)
               if a.has_span and b.has_span else 0.0)

    # Both signals contribute; neither is a gate. Vocabulary weighs more
    # because two unrelated matters can easily share a period.
    score = 0.65 * lexical + 0.35 * overlap
    return Relatedness(score=score, shared_terms=shared, span_overlap=overlap)


# ─── Fact-level signals ───────────────────────────────────────────────────

def temporal_proximity(a: TemporalFact, b: TemporalFact,
                       *, half_life_days: float = 21.0) -> Optional[float]:
    """How close two facts sit in time, in [0, 1], or None if incomparable.

    Decays smoothly with the gap rather than thresholding, so a two-day
    discrepancy between two accounts of one event reads as strong evidence
    of sameness while a six-month gap reads as near-zero. This is the signal
    a disputed date most needs: the disagreement is small precisely because
    both documents are describing the same event.
    """
    if a.timex.date_parsed and b.timex.date_parsed:
        gap = abs((a.timex.date_parsed - b.timex.date_parsed).days)
        return 0.5 ** (gap / half_life_days)
    if a.timex.duration_days is not None and b.timex.duration_days is not None:
        gap = abs(a.timex.duration_days - b.timex.duration_days)
        return 0.5 ** (gap / half_life_days)
    return None


def role_compatibility(a: str, b: str) -> float:
    """Can two facts in these roles describe one event? In [0, 1].

    Requiring identical roles was another veto: extractors disagree about
    role for the same event constantly — one model tags a hearing END and
    another CONTAINS — and the pair was then never considered at all.

    The distinctions that genuinely matter are kept. START against END is
    zero: those are opposite ends of a period and merging them would collapse
    a span into a point. DURATION against anything else is zero: a length is
    not a moment. Everything else is a difference of description, scored
    below an exact agreement so it must earn the match on other evidence.

    This is about the TDG's own closed role vocabulary, not about any
    jurisdiction or case, so it stays in code rather than in a data file.
    """
    a, b = (a or "UNKNOWN").upper(), (b or "UNKNOWN").upper()
    if a == b:
        return 1.0
    pair = {a, b}
    if pair == {"START", "END"}:
        return 0.0
    if "DURATION" in pair:
        return 0.0
    if "UNKNOWN" in pair:
        # Absence of a determined role is not evidence against the match.
        return 0.85
    # CONTAINS against START or END: a point event described as bounding a
    # period in one document and falling within it in the other. A mild
    # disagreement about description, so a mild discount — a steeper one
    # compounds with a poor quote and sinks pairs that are plainly the same
    # event under both names.
    return 0.85


def supporting_quote(fact: TemporalFact) -> tuple[str, bool]:
    """The text that evidences this fact, and whether it really does.

    Extractors disagree about which field holds the quote. One model puts the
    source sentence in ``sentence`` and the bare value in ``raw_text``;
    another puts the sentence in ``raw_text`` and a section heading in
    ``sentence``. Comparing whichever field is nominally the quote therefore
    compares a heading against a sentence and calls the result evidence.

    So the fact's own date decides: whichever text states it is the quote. A
    text that does not state it is bookkeeping, and the caller is told so —
    an unverifiable quote should count as *no* evidence about whether two
    facts match, not as evidence against.
    """
    when = fact.timex.date_parsed
    # A bare value restated is not a quote. Requiring some prose around the
    # date keeps "2025-07-12" out of the running: comparing two such strings
    # scores partial overlap on the shared year and reads as corroboration
    # where there is none.
    candidates = [t for t in (fact.sentence, fact.timex.text)
                  if t and t.strip() and len(tokenise(t)) >= 4]
    if when is not None:
        for text in candidates:
            if mentions_date(text, when):
                return text, True
    elif fact.timex.value:
        value = str(fact.timex.value).lower()
        for text in candidates:
            if value in text.lower():
                return text, True
    # Nothing corroborates the fact. Fall back to the nominal quote field so
    # behaviour is unchanged where this cannot help.
    return (fact.sentence or ""), False


@dataclass
class PairEvidence:
    """Every signal behind one candidate link, kept for the audit trail."""

    entity: float
    sentence: float
    temporal: Optional[float]
    score: float
    threshold: float

    @property
    def passes(self) -> bool:
        return self.score >= self.threshold

    def __str__(self) -> str:
        t = f"{self.temporal:.2f}" if self.temporal is not None else "n/a"
        return (f"entity={self.entity:.2f}, sentence={self.sentence:.2f}, "
                f"time={t} → {self.score:.2f} (needed {self.threshold:.2f})")


# Weights over the three fact-level signals. Renormalised across whichever
# signals are computable for a given pair, so an absent signal lowers
# confidence instead of vetoing the match.
_WEIGHTS = {"entity": 0.45, "sentence": 0.25, "temporal": 0.30}

# A pair with no lexical evidence at all is not a candidate, however close
# in time — otherwise everything happening on one day would merge.
_LEXICAL_FLOOR = 0.12

# The bar depends on whether an embedder is running, because an embedder
# lifts every score, wrong pairs included. Measured over labelled pairs:
#
#   threshold   lexical recall/precision   nomic-embed-text recall/precision
#     0.50           83% / 83%                    100% / 55%
#     0.60           50% / 75%                     83% / 62%
#     0.65           50% / 75%                     50% / 75%
#
# 0.50 is the optimum without an embedder and much too permissive with one:
# semantic similarity puts unrelated-but-adjacent concepts ("disciplinary
# hearing" against "grievance meeting", 0.62) within reach of the same bar
# that admits genuine paraphrase. Neither setting separates cleanly — the
# residual errors are cross-matter pairs, which only a declared matter key
# fixes.
DEFAULT_THRESHOLD = 0.50
DEFAULT_EMBEDDED_THRESHOLD = 0.60


class EventLinker:
    """Links facts across documents by composed evidence and competition."""

    def __init__(
        self,
        tdgs: dict[str, TemporalDependencyGraph],
        *,
        unrelated_threshold: Optional[float] = None,
        related_threshold: Optional[float] = None,
        min_gap_days: int = 60,
        gap_fraction: float = 0.25,
        embedder: Optional[Embedder] = None,
    ) -> None:
        """
        Args:
            unrelated_threshold, related_threshold: the score a pair must
                reach, at zero and at full document relatedness. Left unset
                they default by configuration — see DEFAULT_THRESHOLD and
                DEFAULT_EMBEDDED_THRESHOLD — because the right bar depends on
                whether an embedder is running, not on taste.

                A flat bar is the default: sliding it by relatedness was
                measured and changed nothing, identical to a fixed bar on
                every bundle tested, while at two documents the relatedness
                estimate is actively misleading. The two remain separate
                parameters so the slide can be re-enabled where relatedness
                is trustworthy.

                Within one matter the error costs invert: a wrong merge
                appears as a disputed row a reviewer reads and can split,
                while a missed merge silently hides a real conflict. Guarding
                against wrong merges is the job of role compatibility, the
                temporal disqualifier and one-to-one competition — not of
                this number. Cross-matter merges are prevented by declared
                matter identity; no threshold separates them at all (see
                PATCH-NOTES.md).
            min_gap_days, gap_fraction: how far apart two precise dates may
                be and still be one event told twice. Two accounts of one
                event differ by days or weeks — transcription, or a
                different legal characterisation of the same moment — never
                by years. The allowance is the larger of min_gap_days and
                gap_fraction of the whole bundle's timespan, so it adapts to
                a matter spanning a fortnight or a decade without either
                being written into the code.
            embedder: optional semantic similarity for entity names. Improves
                scores for paraphrases the lexical signal cannot see
                ("termination" vs "dismissal"); its absence is not fatal.
        """
        default = DEFAULT_EMBEDDED_THRESHOLD if embedder else DEFAULT_THRESHOLD
        self.tdgs = tdgs
        self.unrelated_threshold = (unrelated_threshold if unrelated_threshold
                                    is not None else default)
        self.related_threshold = (related_threshold if related_threshold
                                  is not None else default)
        self.min_gap_days = min_gap_days
        self.gap_fraction = gap_fraction
        self._embedder = embedder

        # Statistics are scoped to the matter, not the folder. Token weight is
        # measured over a corpus, so with one shared corpus an unrelated case
        # file sitting in the same directory shifts every score slightly —
        # enough to flip a borderline pair. Measured: two documents describing
        # one hearing scored 0.6013 alone and 0.5964 once an unrelated matter
        # joined the folder, against a 0.60 bar. Whether two documents describe
        # the same event cannot depend on what else the directory holds.
        self.matter_groups = self._partition_by_matter()
        self._stats_by_group = {
            gid: BundleStatistics(document_text(self.tdgs[d]) for d in docs)
            for gid, docs in self.matter_groups.items()}
        self._group_of = {d: gid for gid, docs in self.matter_groups.items()
                          for d in docs}
        # Whole-bundle statistics remain available for callers reasoning about
        # the directory as a whole rather than about one matter.
        self.stats = BundleStatistics(document_text(t) for t in tdgs.values())
        self.profiles = {d: profile_document(t, self.stats_for(d))
                         for d, t in tdgs.items()}
        self._span_by_group = {
            gid: self.stats.bundle_span_days(
                [self.profiles[d].span for d in docs if self.profiles[d].span])
            for gid, docs in self.matter_groups.items()}
        spans = [p.span for p in self.profiles.values() if p.span]
        self.bundle_span = self.stats.bundle_span_days(spans)

    def _partition_by_matter(self) -> dict[int, list[str]]:
        """Group documents that may be linked, by connected components.

        Membership follows the same rule linking does, so the statistics a
        pair is scored against are drawn from exactly the documents it could
        have been linked to.
        """
        parent = {d: d for d in self.tdgs}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        ids = sorted(self.tdgs)
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                if self.same_matter(a, b)[0]:
                    parent[find(a)] = find(b)

        groups: dict[str, list[str]] = {}
        for d in ids:
            groups.setdefault(find(d), []).append(d)
        return {i: docs for i, docs in enumerate(groups.values())}

    def stats_for(self, doc_id: str) -> BundleStatistics:
        """Token statistics for the matter this document belongs to."""
        gid = self._group_of.get(doc_id)
        return self._stats_by_group.get(gid, self.stats)

    def span_for(self, doc_id: str) -> int:
        """Timespan of the matter this document belongs to."""
        gid = self._group_of.get(doc_id)
        return self._span_by_group.get(gid, self.bundle_span)

    @property
    def max_dispute_days(self) -> float:
        """Largest date difference still attributable to one event."""
        return max(self.min_gap_days, self.gap_fraction * self.bundle_span)

    def max_dispute_days_for(self, doc_id: str) -> float:
        """As max_dispute_days, scaled to the matter rather than the folder.

        An unrelated decade-old file in the directory must not widen what
        counts as one event inside an unrelated three-month matter.
        """
        return max(self.min_gap_days, self.gap_fraction * self.span_for(doc_id))

    def temporally_incompatible(self, a: TemporalFact, b: TemporalFact,
                                doc_id: Optional[str] = None) -> bool:
        """True when two precise dates are too far apart to be one event.

        This is the signal that separates a genuine dispute from a false
        merge. Two documents disagreeing by two days about when employment
        ended are describing one termination; two documents naming dates
        fourteen years apart are describing two different people's careers.
        Only applied when both dates are precise — an unresolved or partial
        value is not evidence of anything.
        """
        if not (a.timex.date_parsed and b.timex.date_parsed):
            return False
        gap = abs((a.timex.date_parsed - b.timex.date_parsed).days)
        allowance = (self.max_dispute_days_for(doc_id) if doc_id
                     else self.max_dispute_days)
        return gap > allowance

    # ── signals ────────────────────────────────────────────────────────

    def entity_similarity(self, a: str, b: str,
                          stats: Optional[BundleStatistics] = None) -> float:
        """IDF-weighted lexical agreement, lifted by an embedder when present."""
        stats = stats or self.stats
        lexical = max(stats.weighted_similarity(a, b),
                      stats.weighted_containment(a, b))
        if self._embedder is None:
            return lexical
        try:
            semantic = self._embedder.similarity(a, b)
        except Exception:  # noqa: BLE001 — an unavailable embedder must not break linking
            return lexical
        return max(lexical, semantic)

    def score_pair(self, fa: TemporalFact, fb: TemporalFact,
                   relatedness: float,
                   stats: Optional[BundleStatistics] = None) -> PairEvidence:
        """Compose all computable signals into one score plus its threshold."""
        stats = stats or self.stats
        entity = self.entity_similarity(fa.entity, fb.entity, stats)
        quote_a, ok_a = supporting_quote(fa)
        quote_b, ok_b = supporting_quote(fb)
        sentence = stats.weighted_similarity(quote_a, quote_b)
        temporal = temporal_proximity(fa, fb)

        # The quote signal is always weighed, even when neither text could be
        # corroborated. Dropping it was tried and measured: for two facts with
        # near-identical names and adjacent dates the quote comparison is the
        # only remaining thing that tells them apart, so removing it inflated
        # false pairs more than true ones — lexical precision fell from 83% to
        # 50% at the same bar. Choosing the right text is the improvement;
        # discarding the signal is not.
        available = {"entity": entity, "sentence": sentence}
        if temporal is not None:
            available["temporal"] = temporal
        total_weight = sum(_WEIGHTS[k] for k in available)
        score = sum(_WEIGHTS[k] * v for k, v in available.items()) / total_weight
        # Role disagreement discounts rather than vetoes, except where the
        # roles are genuinely contradictory, where it is zero.
        score *= role_compatibility(fa.role, fb.role)

        # The bar slides from "unrelated" down to "related" in proportion to
        # document relatedness. Unlike a fixed threshold with a discount, it
        # can also move *up*: strangers must clear a higher bar than a fixed
        # base would ever impose.
        span = self.unrelated_threshold - self.related_threshold
        threshold = self.unrelated_threshold - span * max(0.0, min(1.0, relatedness))
        return PairEvidence(entity=entity, sentence=sentence,
                            temporal=temporal, score=score, threshold=threshold)

    # ── matching ───────────────────────────────────────────────────────

    def _candidates(self, doc_a: str, doc_b: str, relatedness: float
                    ) -> list[tuple[float, TemporalFact, TemporalFact, PairEvidence]]:
        out = []
        stats = self.stats_for(doc_a)
        for fa in self.tdgs[doc_a].facts:
            if fa.is_duplicate_of is not None:
                continue
            for fb in self.tdgs[doc_b].facts:
                if fb.is_duplicate_of is not None:
                    continue
                if role_compatibility(fa.role, fb.role) == 0.0:
                    continue
                if self.temporally_incompatible(fa, fb, doc_a):
                    continue
                ev = self.score_pair(fa, fb, relatedness, stats)
                if max(ev.entity, ev.sentence) < _LEXICAL_FLOOR:
                    continue
                if not ev.passes:
                    continue
                out.append((ev.score, fa, fb, ev))
        return out

    def within_document_groups(self, doc_id: str) -> dict[str, str]:
        """Map each fact to the event it belongs to inside its own document.

        Exclusivity has to be per event, not per fact. A document often
        states one event more than once — a letter's summary and its
        operative clause, a pleading's narrative and its schedule — and with
        one-fact-per-document matching the second mention is left stranded
        while the first takes the link. Grouping restatements first makes the
        constraint mean what it should: one event here matches one event
        there.

        Grouping is deliberately strict. Facts join only on the same resolved
        value and compatible roles, so restatements combine while two
        genuinely different dates stay apart — inside one document, two dates
        for one concept are far more likely to be two events than a
        disagreement with itself.
        """
        tdg = self.tdgs[doc_id]
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for f in tdg.facts:
            parent.setdefault(f.id, f.id)
        # The extractor's own duplicate flags are authoritative.
        for f in tdg.facts:
            if f.is_duplicate_of and f.is_duplicate_of in parent:
                parent[find(f.id)] = find(f.is_duplicate_of)

        facts = [f for f in tdg.facts if f.is_duplicate_of is None]
        for i, fa in enumerate(facts):
            for fb in facts[i + 1:]:
                if role_compatibility(fa.role, fb.role) == 0.0:
                    continue
                va, vb = fa.timex.value, fb.timex.value
                if not va or va != vb:
                    continue
                if self.entity_similarity(fa.entity, fb.entity,
                                          self.stats_for(doc_id)) < 0.6:
                    continue
                parent[find(fa.id)] = find(fb.id)

        return {fid: find(fid) for fid in parent}

    def link_pair(self, doc_a: str, doc_b: str) -> list[tuple[TemporalFact, TemporalFact, PairEvidence, Relatedness]]:
        """Best one-to-one matching between two documents' events.

        Greedy on descending score, each *event* consumed once — restatements
        of one event inside a document share a slot rather than competing for
        separate ones. Greedy equals optimal here often enough, and unlike a
        full assignment it degrades predictably: the strongest evidence is
        always honoured first, and a weak pair survives only when nothing
        better wants either side.
        """
        rel = document_relatedness(self.profiles[doc_a], self.profiles[doc_b],
                                   self.stats_for(doc_a))
        candidates = self._candidates(doc_a, doc_b, rel.score)
        candidates.sort(key=lambda c: -c[0])

        groups_a = self.within_document_groups(doc_a)
        groups_b = self.within_document_groups(doc_b)

        by_id_a = {f.id: f for f in self.tdgs[doc_a].facts}
        by_id_b = {f.id: f for f in self.tdgs[doc_b].facts}

        used_a: set[str] = set()
        used_b: set[str] = set()
        matched = []
        for _, fa, fb, ev in candidates:
            ga, gb = groups_a.get(fa.id, fa.id), groups_b.get(fb.id, fb.id)
            if ga in used_a or gb in used_b:
                continue
            used_a.add(ga)
            used_b.add(gb)
            matched.append((fa, fb, ev, rel))
            # Every restatement of the same event joins the same row, rather
            # than being left over as a separate one-source event.
            for other, group_map, by_id, is_left in (
                    (fb, groups_a, by_id_a, True), (fa, groups_b, by_id_b, False)):
                anchor = fa if is_left else fb
                for fid, gid in group_map.items():
                    if gid != group_map.get(anchor.id) or fid == anchor.id:
                        continue
                    sibling = by_id.get(fid)
                    if sibling is None:
                        continue
                    pair = ((sibling, other) if is_left else (other, sibling))
                    matched.append((pair[0], pair[1], ev, rel))
        return matched

    def same_matter(self, doc_a: str, doc_b: str) -> tuple[bool, str]:
        """May these two documents be linked at all, and on what grounds?

        Two sources of evidence, both read from the documents rather than
        deduced from them.

        A declared ``matter`` is decisive. Failing that, the named parties
        are: two documents that each name people and share none of them are
        different matters. Naming the parties is a reading task, which is
        the extractor's job; deciding what the names imply is arithmetic on
        sets, which is code's job. Statistical inference of matter identity
        was tried and measured — relatedness produced more false
        cross-matter links than true within-matter ones, and a rare-token
        partition split genuine single-matter bundles.

        Silence never blocks. A document declaring no matter and naming no
        party links freely, because the operator put these files in one
        bundle and that choice outranks anything derivable from the text.
        """
        a, b = self.tdgs[doc_a], self.tdgs[doc_b]

        if a.matter is not None and b.matter is not None:
            if a.matter == b.matter:
                return True, f"same declared matter ({a.matter})"
            return False, f"different declared matters ({a.matter} / {b.matter})"

        if a.parties and b.parties:
            shared = parties_overlap(a.parties, b.parties)
            if shared:
                return True, f"shared parties ({', '.join(sorted(shared))})"
            return False, ("no party in common ("
                           f"{'; '.join(a.parties)} / {'; '.join(b.parties)})")

        return True, "no matter or parties declared — linked freely"

    @property
    def matter_coverage(self) -> tuple[int, int, int]:
        """(documents declaring a matter, documents naming parties, total)."""
        declared = sum(1 for t in self.tdgs.values() if t.matter is not None)
        named = sum(1 for t in self.tdgs.values() if t.parties)
        return declared, named, len(self.tdgs)

    def link_all(self) -> list[tuple[str, TemporalFact, str, TemporalFact, PairEvidence, Relatedness]]:
        """Link every document pair. Returns flat tuples for the caller to shape."""
        out = []
        self.blocked_pairs: list[tuple[str, str, str]] = []
        doc_ids = sorted(self.tdgs)
        for i, doc_a in enumerate(doc_ids):
            for doc_b in doc_ids[i + 1:]:
                allowed, why = self.same_matter(doc_a, doc_b)
                if not allowed:
                    self.blocked_pairs.append((doc_a, doc_b, why))
                    continue
                for fa, fb, ev, rel in self.link_pair(doc_a, doc_b):
                    out.append((doc_a, fa, doc_b, fb, ev, rel))
        return out
