"""
Entity-time linking.

Determines which entity a temporal expression is about, using:
  1. spaCy NER to find named entities
  2. spaCy dep parse to find the syntactic subject of the governing verb
  3. coreferee for coreference resolution ("The agreement... it expires on...")
  4. Proximity fallback when syntactic linking fails

Two backends:
  1. SpacyEntityLinker     — spaCy NER + dep parse + coreferee (default)
  2. HeuristicEntityLinker — capitalization heuristics fallback

Install:
    pip install spacy coreferee
    python -m spacy download en_core_web_trf
    python -m coreferee install en
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import spacy


@dataclass
class EntityMention:
    """A named entity or event mention in text."""
    text: str
    entity_type: str   # PERSON, ORG, EVENT, GPE, DATE, etc. (spaCy label)
    start_char: int
    end_char: int


# ---------------------------------------------------------------------------
# spaCy-based entity linker (recommended)
# ---------------------------------------------------------------------------

class SpacyEntityLinker:
    """
    Link temporal expressions to entities using spaCy NER + dep parse.

    Strategy (in priority order):
    1. If document_entity is set, always return that
    2. Use coreferee to resolve pronoun/nominal references to their antecedent
    3. Find the syntactic subject of the temporal expression's governing verb
    4. Fall back to nearest named entity in the same sentence
    5. Fall back to nearest named entity in the document

    The dep-parse approach correctly handles:
    - "Germany invaded Poland on September 1, 1939"
      → invade.nsubj = Germany → START linked to Germany/invasion
    - "The patient was discharged on March 17"
      → discharge.nsubj = patient → END linked to patient
    - "The agreement terminates on June 30"
      → terminate.nsubj = agreement → END linked to agreement
    """

    def __init__(self, document_entity: Optional[str] = None, nlp=None):
        """
        Args:
            document_entity: If set, all timex are linked to this entity.
            nlp: Loaded spaCy model. If None, loads en_core_web_trf on first use.
        """
        self.document_entity = document_entity
        self._nlp = nlp
        self._doc_cache: dict[str, object] = {}
        self._coreferee_available = None  # lazy check

    @property
    def nlp(self):
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load("en_core_web_trf")
            self._try_add_coreferee()
        return self._nlp

    def _try_add_coreferee(self):
        """Attempt to add coreferee to the pipeline. Silently skip if unavailable."""
        try:
            import coreferee
            if "coreferee" not in self._nlp.pipe_names:
                self._nlp.add_pipe("coreferee")
            self._coreferee_available = True
        except (ImportError, Exception) as e:
            print(f"[SpacyEntityLinker] coreferee unavailable: {e}. Coreference disabled.")
            self._coreferee_available = False

    def _get_doc(self, text: str):
        if text not in self._doc_cache:
            self._doc_cache[text] = self.nlp(text)
        return self._doc_cache[text]

    def link(
        self,
        text: str,
        timex_start: int,
        timex_end: int,
        entities: Optional[list[EntityMention]] = None,
    ) -> str:
        """Find the entity that a temporal expression at [timex_start:timex_end] refers to."""
        if self.document_entity:
            return self.document_entity

        doc = self._get_doc(text)

        # Find tokens in the timex span
        timex_tokens = [
            t for t in doc
            if not (t.idx + len(t.text) <= timex_start or t.idx >= timex_end)
        ]
        if not timex_tokens:
            return self._fallback_entity(doc, timex_start)

        first_tok = timex_tokens[0]

        # --- Strategy 1: Find governing verb → find its subject ---
        entity = self._find_subject_of_governor(first_tok)
        if entity:
            return entity

        # --- Strategy 2: coreferee resolution of nearby nominal/pronoun ---
        if self._coreferee_available:
            entity = self._resolve_coref(doc, timex_start, timex_end)
            if entity:
                return entity

        # --- Strategy 3: Nearest NER entity in same sentence ---
        entity = self._nearest_ner_in_sentence(doc, timex_start, timex_end)
        if entity:
            return entity

        # --- Strategy 4: Nearest NER entity in document ---
        return self._fallback_entity(doc, timex_start)

    def _find_subject_of_governor(self, timex_token) -> Optional[str]:
        """
        Walk up the dep tree from the timex token to find the governing verb,
        then return its nominal subject (nsubj).
        """
        current = timex_token
        for _ in range(4):  # max 4 hops up
            head = current.head
            if head == current:
                break

            if head.pos_ in ("VERB", "AUX"):
                # Found the governing verb — look for its subject
                for child in head.children:
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        # Return the full noun phrase span
                        return self._expand_noun_phrase(child)
                # No direct subject — check via passive or xcomp
                for child in head.children:
                    if child.dep_ in ("agent",):  # "by [agent]" in passive
                        return self._expand_noun_phrase(child)

            current = head

        return None

    def _expand_noun_phrase(self, token) -> str:
        """
        Get the full noun phrase for a token using its subtree,
        filtered to just the relevant noun chunk.
        """
        # Try to find the noun chunk containing this token
        for chunk in token.doc.noun_chunks:
            if chunk.start <= token.i < chunk.end:
                return chunk.text
        # Fall back to just the token text
        return token.text

    def _resolve_coref(self, doc, timex_start: int, timex_end: int) -> Optional[str]:
        """
        Use coreferee to resolve any pronoun or nominal near the timex
        to its antecedent named entity.
        """
        if not hasattr(doc._, "coref_chains"):
            return None

        # Find tokens near the timex (same sentence, before the timex)
        for token in doc:
            if token.idx >= timex_end:
                break
            if not hasattr(token._, "coref_chains"):
                continue
            chain = token._.coref_chains
            if chain:
                # Get the most representative mention in the chain
                for mention in chain[0].mentions:
                    mention_tokens = [doc[i] for i in mention]
                    mention_text = " ".join(t.text for t in mention_tokens)
                    # Check if any token in the mention is a named entity
                    for t in mention_tokens:
                        if t.ent_type_ in ("PERSON", "ORG", "GPE", "EVENT", "PRODUCT", "FAC"):
                            return mention_text
        return None

    def _nearest_ner_in_sentence(self, doc, timex_start: int, timex_end: int) -> Optional[str]:
        """Find the nearest named entity in the same sentence as the timex."""
        # Find containing sentence
        sent = None
        for s in doc.sents:
            if s.start_char <= timex_start < s.end_char:
                sent = s
                break

        if sent is None:
            return None

        # Filter to meaningful entity types (exclude DATE/TIME/CARDINAL)
        candidate_types = {"PERSON", "ORG", "GPE", "EVENT", "PRODUCT", "FAC", "NORP", "LOC"}
        ents_in_sent = [
            e for e in doc.ents
            if e.start_char >= sent.start_char
            and e.end_char <= sent.end_char
            and e.label_ in candidate_types
        ]

        if not ents_in_sent:
            return None

        if len(ents_in_sent) == 1:
            return ents_in_sent[0].text

        # Prefer entities that appear BEFORE the timex (likely the subject)
        before = [e for e in ents_in_sent if e.end_char <= timex_start]
        if before:
            return max(before, key=lambda e: e.end_char).text

        return min(ents_in_sent, key=lambda e: abs(e.start_char - timex_start)).text

    def _fallback_entity(self, doc, timex_start: int) -> str:
        """Nearest named entity anywhere in the document."""
        candidate_types = {"PERSON", "ORG", "GPE", "EVENT", "PRODUCT", "FAC", "NORP", "LOC"}
        ents = [e for e in doc.ents if e.label_ in candidate_types]
        if not ents:
            return "UNKNOWN_ENTITY"
        return min(ents, key=lambda e: abs(e.start_char - timex_start)).text

    def extract_and_link(
        self, text: str, timex_start: int, timex_end: int
    ) -> tuple[str, list[EntityMention]]:
        """Extract entities and link in one call. Returns (entity_name, all_entities)."""
        doc = self._get_doc(text)
        entities = [
            EntityMention(
                text=e.text,
                entity_type=e.label_,
                start_char=e.start_char,
                end_char=e.end_char,
            )
            for e in doc.ents
        ]
        name = self.link(text, timex_start, timex_end)
        return name, entities

    def clear_cache(self):
        self._doc_cache.clear()


# ---------------------------------------------------------------------------
# Heuristic fallback (original, kept for when spaCy is unavailable)
# ---------------------------------------------------------------------------

def _simple_sentence_split(text: str) -> list[tuple[str, int, int]]:
    sentences = []
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    offset = 0
    for part in parts:
        start = text.find(part, offset)
        if start == -1:
            start = offset
        end = start + len(part)
        sentences.append((part, start, end))
        offset = end
    return sentences


def _extract_entities_heuristic(text: str) -> list[EntityMention]:
    """Simple entity extraction using capitalization heuristics."""
    entities: list[EntityMention] = []
    seen: set[tuple[int, int]] = set()

    for m in re.finditer(
        r"\b([A-Z][a-z]+(?:\s+(?:of|the|and|in|de|von|van|for|to)\s+)?(?:[A-Z][a-z]+|\b[IVX]+\b)(?:\s+(?:[A-Z][a-z]+|\b[IVX]+\b))*)\b",
        text
    ):
        span = (m.start(), m.end())
        if span not in seen:
            seen.add(span)
            entities.append(EntityMention(
                text=m.group(0), entity_type="ENTITY",
                start_char=m.start(), end_char=m.end(),
            ))

    for m in re.finditer(
        r"\b(?:the|this|said)\s+(war|agreement|contract|treaty|battle|conflict|"
        r"siege|campaign|operation|merger|trial|hearing|experiment|study|"
        r"term|patient|company|firm|service|project)\b",
        text, re.IGNORECASE
    ):
        span = (m.start(), m.end())
        if span not in seen:
            seen.add(span)
            entities.append(EntityMention(
                text=m.group(0), entity_type="EVENT",
                start_char=m.start(), end_char=m.end(),
            ))

    return entities


class HeuristicEntityLinker:
    """Capitalization-heuristic entity linker. Fallback for when spaCy is unavailable."""

    def __init__(self, document_entity: Optional[str] = None):
        self.document_entity = document_entity

    def link(
        self,
        text: str,
        timex_start: int,
        timex_end: int,
        entities: Optional[list[EntityMention]] = None,
    ) -> str:
        if self.document_entity:
            return self.document_entity

        if entities is None:
            entities = _extract_entities_heuristic(text)
        if not entities:
            return "UNKNOWN_ENTITY"

        sentences = _simple_sentence_split(text)
        sent_start, sent_end = 0, len(text)
        for _, s_start, s_end in sentences:
            if s_start <= timex_start < s_end:
                sent_start, sent_end = s_start, s_end
                break

        same_sent = [e for e in entities if
                     not (e.end_char <= sent_start or e.start_char >= sent_end)]

        if not same_sent:
            return min(entities, key=lambda e: abs(e.start_char - timex_start)).text
        if len(same_sent) == 1:
            return same_sent[0].text

        before = [e for e in same_sent if e.end_char <= timex_start]
        if before:
            return max(before, key=lambda e: e.end_char).text
        return min(same_sent, key=lambda e: abs(e.start_char - timex_start)).text

    def extract_and_link(
        self, text: str, timex_start: int, timex_end: int
    ) -> tuple[str, list[EntityMention]]:
        entities = _extract_entities_heuristic(text)
        name = self.link(text, timex_start, timex_end, entities)
        return name, entities
