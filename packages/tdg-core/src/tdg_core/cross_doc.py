"""
Cross-document temporal linking.

Given multiple TDGs (from different legal documents), detects:
1. Entity coreference — same temporal concept across documents
2. Temporal entailment — one document's facts satisfy another's rules
3. Contradiction — incompatible temporal constraints across documents
4. Structural analogy — shared temporal patterns (same role+type shape)

This is the implementation layer for:
  - Use case B: "Does this case satisfy that statute?"
  - Use case C: "Do these contracts have contradictory notice periods?"
  - Use case D: "Which RAG chunks become stale when a date changes?"

Usage:
    from tdg_core.cross_doc import CrossDocLinker

    linker = CrossDocLinker()
    linker.add_tdg(contract_tdg)
    linker.add_tdg(statute_tdg)
    linker.add_tdg(judgment_tdg)

    links = linker.find_all_links()
    for link in links:
        print(link)

    # Staleness propagation
    stale = linker.propagate_staleness("contract_001", "e1", delta_days=30)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from datetime import date, timedelta
from typing import Optional, Literal

from tdg_core.tdg import (
    TemporalDependencyGraph,
    TemporalFact,
    TemporalDependency,
)
from tdg_core.io import DEFAULT_SIMILARITY_THRESHOLD
from tdg_core.embeddings import EmbeddingSimilarity, normalise_entity


# ─── Sentence-level text overlap ──────────────────────────────────────────

_STOP = {"the", "of", "a", "an", "in", "on", "to", "for", "and", "or",
         "is", "are", "was", "were", "be", "been", "shall", "will",
         "that", "this", "by", "with", "from", "at", "as", "its",
         "it", "not", "no", "any", "all", "each", "such", "may"}


def _text_overlap(text_a: str, text_b: str) -> float:
    """Token-based Jaccard overlap between two text passages.

    Used to confirm whether two facts with similar entity names actually
    refer to the same clause. High overlap (>0.2) = same or paraphrased
    clause. Low overlap (<0.1) = different clauses that share a generic name.
    """
    if not text_a or not text_b:
        return 0.0
    tokens_a = set(text_a.lower().split()) - _STOP
    tokens_b = set(text_b.lower().split()) - _STOP
    if not tokens_a or not tokens_b:
        return 0.0
    overlap = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return overlap / union if union else 0.0


# ─── Data structures ──────────────────────────────────────────────────────

LinkType = Literal[
    "coreference",         # same concept across documents
    "entailment",          # one doc's facts satisfy another's constraints
    "contradiction",       # incompatible RULE-level values across documents
    "parallel_application",# same concept applied to different cases (not a contradiction)
    "structural_analogy",  # shared temporal pattern shape
]


@dataclass
class CrossDocLink:
    """A detected relationship between facts in different documents."""
    link_type: LinkType
    from_doc: str           # document_id
    from_fact: str          # fact id
    to_doc: str
    to_fact: str
    confidence: float
    explanation: str
    # For contradiction links: what the conflict is
    value_a: Optional[str] = None
    value_b: Optional[str] = None
    delta_days: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def __repr__(self) -> str:
        return (f"CrossDocLink({self.link_type}: "
                f"{self.from_doc}/{self.from_fact} ↔ "
                f"{self.to_doc}/{self.to_fact}, "
                f"conf={self.confidence:.2f})")


@dataclass
class StaleFact:
    """A fact that became stale due to an upstream edit."""
    doc_id: str
    fact_id: str
    old_value: Optional[str]
    reason: str             # how this was reached (chain or coreference)
    hop_distance: int       # how many edges from the edit source


# ─── Entity normalisation ─────────────────────────────────────────────────

# Re-export normalise_entity for backward compatibility
_normalise_entity = normalise_entity


def _entity_similarity(
    a: str,
    b: str,
    embedder: Optional[EmbeddingSimilarity] = None,
) -> float:
    """
    Compute similarity between two entity names.
    Returns 0.0-1.0.

    Uses embedding cosine similarity when an embedder is provided and
    the service is reachable. Falls back to token Jaccard otherwise.
    """
    if embedder is not None:
        return embedder.similarity(a, b)

    # Fallback: normalise + exact match, then token Jaccard
    from tdg_core.embeddings import _token_jaccard
    na = normalise_entity(a)
    nb = normalise_entity(b)
    if na == nb:
        return 1.0
    return _token_jaccard(na, nb)


# ─── Value comparison ─────────────────────────────────────────────────────

def _duration_to_days(value: str) -> Optional[int]:
    """Parse ISO duration or 'N days' strings to days."""
    if not value:
        return None
    total = 0
    found = False
    for m in re.finditer(r"(\d+(?:\.\d+)?)([YMWD])", value):
        n = float(m.group(1))
        u = m.group(2)
        if u == "Y": total += int(n * 365)
        elif u == "M": total += int(n * 30)
        elif u == "W": total += int(n * 7)
        elif u == "D": total += int(n)
        found = True
    return total if found else None


def _values_match(fact_a: TemporalFact, fact_b: TemporalFact,
                  tolerance_days: int = 5) -> tuple[bool, Optional[int]]:
    """
    Check if two facts have matching temporal values.
    Returns (match, delta_days).
    """
    # Both have parsed dates
    if fact_a.timex.date_parsed and fact_b.timex.date_parsed:
        delta = abs((fact_a.timex.date_parsed - fact_b.timex.date_parsed).days)
        return delta <= tolerance_days, delta

    # Both have durations
    if fact_a.timex.duration_days is not None and fact_b.timex.duration_days is not None:
        delta = abs(fact_a.timex.duration_days - fact_b.timex.duration_days)
        return delta <= tolerance_days, delta

    # Both have ISO values that look like durations
    val_a = fact_a.timex.value or ""
    val_b = fact_b.timex.value or ""
    if val_a.startswith("P") and val_b.startswith("P"):
        days_a = _duration_to_days(val_a)
        days_b = _duration_to_days(val_b)
        if days_a is not None and days_b is not None:
            delta = abs(days_a - days_b)
            return delta <= tolerance_days, delta

    return False, None


def _values_contradict(fact_a: TemporalFact, fact_b: TemporalFact,
                       tolerance_days: int = 5) -> tuple[bool, Optional[int]]:
    """
    Check if two facts that SHOULD be the same have different values.
    Only meaningful when both facts have computable values.
    """
    match, delta = _values_match(fact_a, fact_b, tolerance_days)
    if delta is None:
        return False, None  # can't tell — not a contradiction, just unknown
    return not match, delta


# ─── Main linker ──────────────────────────────────────────────────────────

class CrossDocLinker:
    """
    Finds temporal relationships across multiple TDGs.

    Add TDGs with add_tdg(), then call find_all_links() to detect
    coreference, entailment, contradiction, and structural analogy.
    """

    def __init__(
        self,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        sentence_threshold: float = 0.2,
        embedder: Optional[EmbeddingSimilarity] = None,
        composed: bool = False,
    ):
        """
        Args:
            composed: use evidence composition (tdg_core.linking) instead of
                the two hard AND-gates. The gated path requires entity name
                *and* sentence overlap to clear independent floors, so one
                poor signal — usually the quote, which is whatever the
                extractor emitted — vetoes a correct link. Composition
                weighs the same signals plus temporal proximity, lowers the
                bar for documents established to concern the same matter,
                and resolves competition one-to-one.
        """
        self.tdgs: dict[str, TemporalDependencyGraph] = {}
        self.similarity_threshold = similarity_threshold
        self.sentence_threshold = sentence_threshold
        self._embedder = embedder
        self.composed = composed

    def add_tdg(self, tdg: TemporalDependencyGraph) -> None:
        self.tdgs[tdg.document_id] = tdg

    def _all_fact_pairs(self):
        """Yield all cross-document fact pairs."""
        doc_ids = list(self.tdgs.keys())
        for i, doc_a in enumerate(doc_ids):
            for doc_b in doc_ids[i + 1:]:
                for fa in self.tdgs[doc_a].facts:
                    if fa.is_duplicate_of is not None:
                        continue
                    for fb in self.tdgs[doc_b].facts:
                        if fb.is_duplicate_of is not None:
                            continue
                        yield doc_a, fa, doc_b, fb

    # ── Coreference detection ──────────────────────────────────────────

    # DESIGN TODO (staged linking): before fact-level comparison, tag each
    # document with a SET of labels (parties, case number, domain, date
    # range). A document may carry many tags, and sharing a tag does not
    # by itself make two documents related — "both contracts" is not a
    # relation, "same parties + overlapping period" is. Event linking
    # should only run on pairs whose tag overlap is of a relating kind.
    # This cuts false merges when unrelated files land in one bundle.

    def find_coreferences(self) -> list[CrossDocLink]:
        """
        Find facts across documents that refer to the same temporal concept.

        Two-signal matching:
        1. Entity name similarity (are the names similar?)
        2. Source sentence overlap (are they describing the same clause?)

        Both signals must be present. Entity name alone produces false
        matches ("agreement" appears in 23/45 EU contracts but refers to
        different agreements). Sentence overlap confirms the match.

        With composed=True this is replaced by evidence composition, where
        that same observation is handled by measuring how discriminative a
        word is in the bundle at hand rather than by requiring a second
        signal to clear a fixed floor.
        """
        if self.composed:
            return self._find_coreferences_composed()

        links = []
        seen = set()

        for doc_a, fa, doc_b, fb in self._all_fact_pairs():
            if fa.role != fb.role:
                continue

            entity_sim = _entity_similarity(fa.entity, fb.entity, self._embedder)
            if entity_sim < self.similarity_threshold:
                continue

            pair_key = tuple(sorted([(doc_a, fa.id), (doc_b, fb.id)]))
            if pair_key in seen:
                continue
            seen.add(pair_key)

            # Sentence overlap confirms the match is real
            sentence_sim = _text_overlap(fa.sentence, fb.sentence)

            # Combined confidence: entity name gets you candidates,
            # sentence overlap confirms or rejects
            confidence = entity_sim * 0.5 + sentence_sim * 0.5

            # Skip pairs with no sentence-level evidence unless entity
            # names are effectively identical (normalised exact match)
            na = normalise_entity(fa.entity)
            nb = normalise_entity(fb.entity)
            exact_name = (na == nb) and len(na.split()) > 1  # multi-word exact match
            if sentence_sim < 0.1 and not exact_name:
                continue

            val_match, delta = _values_match(fa, fb)
            if val_match and delta is not None:
                confidence = min(confidence + 0.15, 1.0)
                explanation = (
                    f"Same concept: {na} ({fa.role}) "
                    f"with matching values"
                    f"{f' (Δ{delta}d)' if delta else ''}"
                    f" [entity={entity_sim:.2f}, sentence={sentence_sim:.2f}]"
                )
            elif delta is not None and not val_match:
                # Values disagree. If sentences overlap strongly, the
                # contradiction detector will handle it. If sentences
                # don't overlap, these are different concepts with the
                # same name — still coreference at the type level.
                if sentence_sim >= self.sentence_threshold:
                    continue  # same clause, different values → contradiction detector
                else:
                    confidence *= 0.6
                    explanation = (
                        f"Same concept type: {na} ({fa.role}) "
                        f"in different contexts"
                        f" [entity={entity_sim:.2f}, sentence={sentence_sim:.2f}]"
                    )
            else:
                explanation = (
                    f"Same concept: {na} ({fa.role}), "
                    f"values not directly comparable"
                    f" [entity={entity_sim:.2f}, sentence={sentence_sim:.2f}]"
                )

            if confidence < self.similarity_threshold:
                continue

            val_a = fa.timex.value or (fa.timex.date_parsed.isoformat() if fa.timex.date_parsed else None)
            val_b = fb.timex.value or (fb.timex.date_parsed.isoformat() if fb.timex.date_parsed else None)

            links.append(CrossDocLink(
                link_type="coreference",
                from_doc=doc_a, from_fact=fa.id,
                to_doc=doc_b, to_fact=fb.id,
                confidence=confidence,
                explanation=explanation,
                value_a=val_a,
                value_b=val_b,
                delta_days=delta,
            ))

        return links

    def _find_coreferences_composed(self) -> list[CrossDocLink]:
        """Coreference via tdg_core.linking.

        Emits contradiction links directly for matched pairs whose dates
        disagree: once two facts are established as the same event, a
        difference in their dates *is* the dispute, and no separate
        sentence-overlap test is needed to decide whether they ought to
        have agreed.
        """
        from tdg_core.linking import EventLinker

        linker = EventLinker(self.tdgs, embedder=self._embedder)
        self.last_linker = linker
        links: list[CrossDocLink] = []
        for doc_a, fa, doc_b, fb, ev, rel in linker.link_all():
            val_a = fa.timex.value or (fa.timex.date_parsed.isoformat()
                                       if fa.timex.date_parsed else None)
            val_b = fb.timex.value or (fb.timex.date_parsed.isoformat()
                                       if fb.timex.date_parsed else None)
            match, delta = _values_match(fa, fb)
            contradicts = delta is not None and not match
            links.append(CrossDocLink(
                link_type="contradiction" if contradicts else "coreference",
                from_doc=doc_a, from_fact=fa.id,
                to_doc=doc_b, to_fact=fb.id,
                confidence=ev.score,
                explanation=(
                    f"{'Conflicting values' if contradicts else 'Same event'} "
                    f"for {fa.entity!r} / {fb.entity!r} ({fa.role})"
                    f"{f': {val_a} vs {val_b} (Δ{delta}d)' if contradicts else ''}"
                    f" [{ev}; documents: {rel.explanation}]"
                ),
                value_a=val_a, value_b=val_b, delta_days=delta,
            ))
        return links

    # ── Contradiction detection ────────────────────────────────────────

    def find_contradictions(self) -> list[CrossDocLink]:
        """
        Find facts across documents that refer to the same clause but
        have incompatible values.

        Uses sentence overlap to determine whether two facts SHOULD agree.
        If two facts quote the same clause (high sentence overlap) but have
        different values, that's a contradiction. If they have similar entity
        names but describe different clauses (low sentence overlap), that's
        parallel application — expected, not contradictory.

        This replaces the old document-type classification (INSTANCE_DOC_TYPES)
        with a signal derived from the data itself.
        """
        if self.composed:
            # Composition emits contradictions alongside coreference, from a
            # single matching pass — a matched pair with disagreeing dates is
            # the contradiction. Re-running the gated detector here would
            # double-report them under different confidences.
            return []

        links = []
        seen = set()

        for doc_a, fa, doc_b, fb in self._all_fact_pairs():
            if fa.role != fb.role:
                continue

            entity_sim = _entity_similarity(fa.entity, fb.entity, self._embedder)
            if entity_sim < self.similarity_threshold:
                continue

            pair_key = tuple(sorted([(doc_a, fa.id), (doc_b, fb.id)]))
            if pair_key in seen:
                continue
            seen.add(pair_key)

            is_contra, delta = _values_contradict(fa, fb)
            if not is_contra:
                continue

            val_a = fa.timex.value or (fa.timex.date_parsed.isoformat() if fa.timex.date_parsed else None)
            val_b = fb.timex.value or (fb.timex.date_parsed.isoformat() if fb.timex.date_parsed else None)

            # Sentence overlap determines whether these facts SHOULD agree.
            # High overlap = same clause = values should match = real contradiction.
            # Low overlap = different clauses = different values are expected.
            sentence_sim = _text_overlap(fa.sentence, fb.sentence)
            na = normalise_entity(fa.entity)

            if sentence_sim >= self.sentence_threshold:
                # Same or paraphrased clause — values should agree but don't.
                confidence = entity_sim * 0.5 + sentence_sim * 0.5
                links.append(CrossDocLink(
                    link_type="contradiction",
                    from_doc=doc_a, from_fact=fa.id,
                    to_doc=doc_b, to_fact=fb.id,
                    confidence=confidence,
                    explanation=(
                        f"Contradictory values for {na} ({fa.role}): "
                        f"{val_a} vs {val_b} (Δ{delta}d)"
                        f" [entity={entity_sim:.2f}, sentence={sentence_sim:.2f}]"
                    ),
                    value_a=val_a,
                    value_b=val_b,
                    delta_days=delta,
                ))
            else:
                # Different clauses that happen to share an entity name.
                # "Agreement" in two different EU treaties having different
                # dates is expected, not contradictory.
                confidence = entity_sim * 0.3 + sentence_sim * 0.2
                if confidence >= 0.2:  # only report if minimally interesting
                    links.append(CrossDocLink(
                        link_type="parallel_application",
                        from_doc=doc_a, from_fact=fa.id,
                        to_doc=doc_b, to_fact=fb.id,
                        confidence=confidence,
                        explanation=(
                            f"Same concept type in different contexts: "
                            f"{na} ({fa.role}) "
                            f"({val_a} vs {val_b}, Δ{delta}d)"
                            f" [entity={entity_sim:.2f}, sentence={sentence_sim:.2f}]"
                        ),
                        value_a=val_a,
                        value_b=val_b,
                        delta_days=delta,
                    ))

        return links

    # ── Structural analogy ─────────────────────────────────────────────

    def find_structural_analogies(self) -> list[CrossDocLink]:
        """
        Find documents that share temporal pattern shapes.

        A structural analogy exists when two documents have the same
        dependency topology: e.g. both have START →(additive)→ END
        with a DURATION fact, indicating the same "fixed term + computed
        expiry" pattern.

        This is a document-level link, not fact-level.
        """
        links = []

        def _dep_signature(tdg: TemporalDependencyGraph) -> set[tuple[str, str, str]]:
            """Extract the role-pair-type triples as a structural fingerprint."""
            from tdg_core.provenance import trusted_dependencies

            fact_map = {f.id: f for f in tdg.facts}
            sig = set()
            # An invented edge would contribute a shape the document never
            # had, nudging a similarity score on evidence that is not there.
            for dep in trusted_dependencies(tdg):
                fa = fact_map.get(dep.from_id)
                fb = fact_map.get(dep.to_id)
                if fa and fb:
                    sig.add((fa.role, fb.role, dep.constraint_type))
            return sig

        doc_ids = list(self.tdgs.keys())
        for i, doc_a in enumerate(doc_ids):
            sig_a = _dep_signature(self.tdgs[doc_a])
            if not sig_a:
                continue
            for doc_b in doc_ids[i + 1:]:
                sig_b = _dep_signature(self.tdgs[doc_b])
                if not sig_b:
                    continue

                shared = sig_a & sig_b
                if not shared:
                    continue

                total = sig_a | sig_b
                overlap = len(shared) / len(total) if total else 0

                if overlap >= 0.2:  # at least 20% pattern overlap
                    links.append(CrossDocLink(
                        link_type="structural_analogy",
                        from_doc=doc_a, from_fact="*",
                        to_doc=doc_b, to_fact="*",
                        confidence=overlap,
                        explanation=(
                            f"Shared patterns: {', '.join(f'{a}→{b} [{t}]' for a, b, t in shared)}"
                        ),
                    ))

        return links

    # ── All links ──────────────────────────────────────────────────────

    def find_all_links(self) -> list[CrossDocLink]:
        """Run all cross-doc analyses and return combined results."""
        links = []
        links.extend(self.find_coreferences())
        links.extend(self.find_contradictions())
        links.extend(self.find_structural_analogies())
        # Sort by confidence descending
        links.sort(key=lambda l: -l.confidence)
        return links

    # ── Staleness propagation (Use case D) ─────────────────────────────

    def propagate_staleness(
        self,
        edited_doc: str,
        edited_fact_id: str,
        delta_days: int = 0,
    ) -> list[StaleFact]:
        """
        Given an edit to a fact in one document, find all facts
        (within and across documents) that become stale.

        Strategy:
        1. BFS through intra-doc additive dependencies
        2. For each affected fact, check cross-doc coreferences
        3. Those coreferent facts are also stale
        """
        if edited_doc not in self.tdgs:
            return []

        tdg = self.tdgs[edited_doc]
        fact_map = {f.id: f for f in tdg.facts}

        stale: list[StaleFact] = []
        visited = {edited_fact_id}
        queue = [(edited_fact_id, 0)]

        # Phase 1: intra-doc BFS through additive edges. Only those the
        # document supports: reporting a fact as invalidated by a period the
        # document never states is an invented consequence, and reads exactly
        # like a real one.
        from tdg_core.provenance import trusted_dependencies

        trusted = trusted_dependencies(tdg)
        while queue:
            current_id, depth = queue.pop(0)
            current_fact = fact_map.get(current_id)

            for dep in trusted:
                if dep.from_id != current_id:
                    continue
                if dep.constraint_type != "additive":
                    continue
                if dep.to_id in visited:
                    continue

                visited.add(dep.to_id)
                target = fact_map.get(dep.to_id)
                if target:
                    stale.append(StaleFact(
                        doc_id=edited_doc,
                        fact_id=dep.to_id,
                        old_value=target.timex.value,
                        reason=f"additive chain from {edited_fact_id}: {dep.constraint_expr}",
                        hop_distance=depth + 1,
                    ))
                    queue.append((dep.to_id, depth + 1))

        # Phase 2: cross-doc via coreferences
        coref_links = self.find_coreferences()
        all_stale_facts = {edited_fact_id} | {s.fact_id for s in stale}
        stale_in_doc = {(edited_doc, fid) for fid in all_stale_facts}

        for link in coref_links:
            src = (link.from_doc, link.from_fact)
            dst = (link.to_doc, link.to_fact)

            if src in stale_in_doc:
                target_doc = link.to_doc
                target_fact_id = link.to_fact
            elif dst in stale_in_doc:
                target_doc = link.from_doc
                target_fact_id = link.from_fact
            else:
                continue

            target_tdg = self.tdgs.get(target_doc)
            if not target_tdg:
                continue

            target_fact = next(
                (f for f in target_tdg.facts if f.id == target_fact_id), None
            )
            if target_fact:
                stale.append(StaleFact(
                    doc_id=target_doc,
                    fact_id=target_fact_id,
                    old_value=target_fact.timex.value,
                    reason=f"cross-doc coreference with {edited_doc}/{link.from_fact if target_doc == link.to_doc else link.to_fact}",
                    hop_distance=99,  # cross-doc hops are conceptually "far"
                ))

        return stale