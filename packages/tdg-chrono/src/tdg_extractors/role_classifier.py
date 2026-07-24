"""
Temporal role classification.

Determines whether a temporal expression is START, END, DURATION,
CONTAINS, or UNKNOWN — using spaCy's dependency parse to find the
syntactic governor of the date token.

Key insight: the syntactic head verb of a date span is a far more
reliable signal than verb-proximity heuristics. "The agreement,
which was signed on January 15, terminates on June 30" — proximity
would assign both dates to "terminates"; dep parse correctly assigns
"signed" to January 15 and "terminates" to June 30.

Two backends:
  1. SpacyRoleClassifier  — uses dep parse + lemmatization (default)
  2. PatternRoleClassifier — regex fallback (no spaCy needed)

Verb sets use LEMMA form only — spaCy's lemmatizer handles all
morphological variants (began/begin/beginning → begin).
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from tdg_core.tdg import TemporalRole

if TYPE_CHECKING:
    import spacy


@dataclass
class RoleSignal:
    """Linguistic evidence for a temporal role assignment."""
    role: TemporalRole
    verb: Optional[str] = None
    prep: Optional[str] = None
    confidence: float = 0.5
    reason: str = ""


# ---------------------------------------------------------------------------
# Verb/preposition lexicons — LEMMA FORM ONLY
# spaCy lemmatizes before lookup, so no need to list inflected forms.
# ---------------------------------------------------------------------------

START_VERB_LEMMAS = {
    # Generic inception
    "begin", "start", "commence", "initiate", "launch", "open",
    "inaugurate", "establish", "found", "incorporate",
    # Domain-crossing
    "admit",        # medical: patient admitted
    "diagnose",     # medical
    "enroll",       # education
    "deploy",       # military/tech
    "bear",         # biographical: born (lemma of "born" → "bear" in spaCy)
    "enter",        # generic
    "invade",       # military
    "sign",         # legal: signing = start of agreement
    "execute",      # legal
    "file",         # legal
    "hire",         # corporate
    "appoint",      # corporate/political
    "elect",        # political
    "accede",       # political
    "ascend",       # political
    "announce",     # corporate/general
    "award",        # biographical
    "take",         # legal: "takes effect on"
    "take effect",  # legal — handled as prep pattern instead
}

END_VERB_LEMMAS = {
    "end", "finish", "terminate", "expire", "close", "conclude",
    "dissolve", "abolish", "complete", "cease", "stop", "adjourn",
    # Domain-crossing
    "discharge",    # medical
    "die",          # biographical
    "resign",       # corporate/political
    "retire",       # corporate
    "abdicate",     # political
    "surrender",    # military
    "repeal",       # legal
    "dismiss",      # legal
    "graduate",     # education
    "demolish",     # construction
    "fall",         # military (fall of a city)
    "defeat",       # military
    "cure",         # medical
}

DURATION_VERB_LEMMAS = {
    "last", "continue", "span", "endure", "persist", "extend",
    "run",      # "the contract runs for 2 years"
    "serve",    # "served for 8 years"
    "reign",    # "reigned for 30 years"
    "rule",     # political
    "survive",  # "survived for 3 months"
}

# Prepositions — checked against the dep parse prep/mark relation
START_PREPS = {"from", "since", "starting", "beginning", "as of", "effective"}
# Note: "to" is excluded here — it's too ambiguous ("moved to Paris in 1891").
# "to" is only treated as END when it appears in a date range ("from X to Y"),
# which is handled by the dep parse range detection, not the prep lookup.
END_PREPS = {"until", "till", "through", "by", "before", "no later than", "ending"}
DURATION_PREPS = {"for", "over", "spanning", "within"}
CONTAINS_PREPS = {"during", "throughout", "amid", "while"}

# Noun heads that carry temporal role (dep parse: timex is child of these nouns)
START_NOUNS = {
    "start", "beginning", "commencement", "inception", "onset", "outset",
    "founding", "establishment", "admission", "diagnosis", "birth",
    "inauguration", "signing", "execution", "opening", "launch",
}
END_NOUNS = {
    "end", "conclusion", "termination", "expiration", "expiry", "closure",
    "death", "discharge", "dissolution", "completion", "surrender",
    "resignation", "retirement", "graduation",
}
DURATION_NOUNS = {
    "duration", "length", "period", "span", "term", "tenure", "reign",
}


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class RoleClassifier(ABC):
    @abstractmethod
    def classify(self, text: str, timex_start: int, timex_end: int) -> RoleSignal:
        ...


# ---------------------------------------------------------------------------
# spaCy-based classifier (dep parse + lemmatization)
# ---------------------------------------------------------------------------

class SpacyRoleClassifier(RoleClassifier):
    """
    Classify temporal role using spaCy's dependency parse.

    Strategy:
    1. Find all tokens that overlap with the timex span
    2. Walk up the dep tree to find the governing verb
    3. Check the governor's lemma against role verb sets
    4. If governor is a prep/mark, check its head instead
    5. Fall back to prep-based and noun-based signals
    """

    def __init__(self, nlp=None):
        """
        Args:
            nlp: Loaded spaCy model. If None, loads en_core_web_trf on first use.
        """
        self._nlp = nlp
        self._doc_cache: dict[str, object] = {}  # text → spaCy Doc

    @property
    def nlp(self):
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load("en_core_web_trf")
        return self._nlp

    def _get_doc(self, text: str):
        """Get or create a spaCy Doc, caching by text."""
        if text not in self._doc_cache:
            self._doc_cache[text] = self.nlp(text)
        return self._doc_cache[text]

    def classify(self, text: str, timex_start: int, timex_end: int) -> RoleSignal:
        doc = self._get_doc(text)

        # Find tokens overlapping with the timex span
        timex_tokens = [
            t for t in doc
            if t.idx >= timex_start or (t.idx < timex_end and t.idx + len(t.text) > timex_start)
        ]
        # More precise overlap
        timex_tokens = [
            t for t in doc
            if not (t.idx + len(t.text) <= timex_start or t.idx >= timex_end)
        ]

        if not timex_tokens:
            return RoleSignal("UNKNOWN", confidence=0.2, reason="no tokens found in span")

        # --- Signal 1: Check immediate preposition/marker before timex ---
        first_tok = timex_tokens[0]
        prep_signal = self._check_prep(first_tok)
        if prep_signal:
            return prep_signal

        # --- Signal 2: Walk dep tree to find governing verb ---
        # Try each timex token as a candidate head
        for tok in timex_tokens:
            verb_signal = self._walk_to_verb(tok)
            if verb_signal and verb_signal.role != "UNKNOWN":
                return verb_signal

        # --- Signal 3: Check noun head of the timex ---
        noun_signal = self._check_noun_head(timex_tokens[0])
        if noun_signal:
            return noun_signal

        # --- Signal 4: Genitive "of [DATE]" — common in legal text ---
        # "decision of 18 March 1992", "Directive of 15 July 1975"
        # The date's role is the role of its head noun
        if first_tok.i > 0:
            prev = doc[first_tok.i - 1]
            if prev.text.lower() == "of":
                head_noun = prev.head
                if head_noun.pos_ in ("NOUN", "PROPN"):
                    head_lemma = head_noun.lemma_.lower()
                    if head_lemma in END_NOUNS:
                        return RoleSignal("END", prep="of", confidence=0.6,
                                          reason=f"genitive: '{head_lemma} of [DATE]'")
                    if head_lemma in START_NOUNS | {"decision", "directive", "regulation",
                                                     "order", "judgment", "decree", "act",
                                                     "treaty", "convention", "communication",
                                                     "notice", "letter", "application", "action"}:
                        return RoleSignal("START", prep="of", confidence=0.6,
                                          reason=f"genitive: '{head_lemma} of [DATE]'")

        # --- Signal 5: Check "on [DATE]" pattern — common but ambiguous ---
        if first_tok.i > 0:
            prev = doc[first_tok.i - 1]
            if prev.text.lower() == "on":
                return RoleSignal("UNKNOWN", prep="on", confidence=0.3,
                                  reason="ambiguous prep 'on'")

        return RoleSignal("UNKNOWN", confidence=0.2, reason="no dep parse signal found")

    def _check_prep(self, token) -> Optional[RoleSignal]:
        """Check if the token before this one is a role-signaling preposition."""
        if token.i == 0:
            return None

        # Look back up to 3 tokens for a preposition
        for i in range(max(0, token.i - 3), token.i):
            prev = token.doc[i]
            lemma = prev.lemma_.lower()
            text = prev.text.lower()

            for prep in sorted(START_PREPS, key=len, reverse=True):
                if text == prep or lemma == prep:
                    return RoleSignal("START", prep=prep, confidence=0.8,
                                      reason=f"prep '{prep}' before timex")

            for prep in sorted(END_PREPS, key=len, reverse=True):
                if text == prep or lemma == prep:
                    return RoleSignal("END", prep=prep, confidence=0.8,
                                      reason=f"prep '{prep}' before timex")

            for prep in sorted(DURATION_PREPS, key=len, reverse=True):
                if text == prep or lemma == prep:
                    return RoleSignal("DURATION", prep=prep, confidence=0.75,
                                      reason=f"prep '{prep}' before timex")

            for prep in sorted(CONTAINS_PREPS, key=len, reverse=True):
                if text == prep or lemma == prep:
                    return RoleSignal("CONTAINS", prep=prep, confidence=0.75,
                                      reason=f"prep '{prep}' before timex")

        return None

    def _walk_to_verb(self, token, max_hops: int = 4) -> Optional[RoleSignal]:
        """
        Walk up the dep tree from a token to find its governing verb.
        Returns a RoleSignal if a role verb is found.
        """
        current = token
        for hop in range(max_hops):
            head = current.head
            if head == current:
                break  # reached root

            head_lemma = head.lemma_.lower()

            # Direct verb check
            if head.pos_ in ("VERB", "AUX"):
                role = self._lemma_to_role(head_lemma)
                if role != "UNKNOWN":
                    conf = max(0.4, 0.85 - hop * 0.1)  # confidence degrades with distance
                    return RoleSignal(
                        role, verb=head_lemma, confidence=conf,
                        reason=f"dep parse: head verb '{head_lemma}' ({hop+1} hop(s))"
                    )

            # Preposition/marker — check its head
            if head.dep_ in ("prep", "mark", "advmod") and head.head.pos_ == "VERB":
                prep_text = head.text.lower()
                verb_lemma = head.head.lemma_.lower()
                # Prep takes priority for role signal
                for prep in START_PREPS:
                    if prep_text == prep:
                        return RoleSignal("START", prep=prep, verb=verb_lemma,
                                          confidence=0.82,
                                          reason=f"dep: prep '{prep}' → verb '{verb_lemma}'")
                for prep in END_PREPS:
                    if prep_text == prep:
                        return RoleSignal("END", prep=prep, verb=verb_lemma,
                                          confidence=0.82,
                                          reason=f"dep: prep '{prep}' → verb '{verb_lemma}'")
                # Otherwise use the verb
                role = self._lemma_to_role(verb_lemma)
                if role != "UNKNOWN":
                    return RoleSignal(role, verb=verb_lemma, prep=prep_text,
                                      confidence=0.78,
                                      reason=f"dep: prep+verb '{prep_text}+{verb_lemma}'")

            current = head

        return RoleSignal("UNKNOWN", confidence=0.2, reason="no role verb in dep tree")

    def _check_noun_head(self, token) -> Optional[RoleSignal]:
        """Check if the timex is a child of a role-signaling noun."""
        head = token.head
        if head.pos_ in ("NOUN", "PROPN"):
            lemma = head.lemma_.lower()
            if lemma in START_NOUNS:
                return RoleSignal("START", confidence=0.65,
                                  reason=f"noun head '{lemma}'")
            if lemma in END_NOUNS:
                return RoleSignal("END", confidence=0.65,
                                  reason=f"noun head '{lemma}'")
            if lemma in DURATION_NOUNS:
                return RoleSignal("DURATION", confidence=0.65,
                                  reason=f"noun head '{lemma}'")
        return None

    def _lemma_to_role(self, lemma: str) -> TemporalRole:
        """Map a verb lemma to a temporal role."""
        if lemma in START_VERB_LEMMAS:
            return "START"
        if lemma in END_VERB_LEMMAS:
            return "END"
        if lemma in DURATION_VERB_LEMMAS:
            return "DURATION"
        return "UNKNOWN"

    def clear_cache(self):
        """Clear the doc cache to free memory."""
        self._doc_cache.clear()


# ---------------------------------------------------------------------------
# Regex fallback classifier (original, kept for when spaCy is unavailable)
# ---------------------------------------------------------------------------

def _window_around(text: str, start: int, end: int, window: int = 80) -> str:
    return text[max(0, start - window): end + window].lower()


class PatternRoleClassifier(RoleClassifier):
    """
    Regex-based fallback classifier using verb proximity heuristics.
    Less accurate than SpacyRoleClassifier — use only as fallback.
    """

    # Use the same lemma sets but extend to cover common inflections
    # since we can't lemmatize without spaCy
    _START_WORDS = START_VERB_LEMMAS | {
        "began", "started", "commenced", "initiated", "launched", "opened",
        "inaugurated", "established", "founded", "incorporated", "admitted",
        "diagnosed", "enrolled", "deployed", "born", "entered", "invaded",
        "signed", "executed", "filed", "hired", "appointed", "elected",
        "announced", "awarded",
    }
    _END_WORDS = END_VERB_LEMMAS | {
        "ended", "finished", "terminated", "expired", "closed", "concluded",
        "dissolved", "abolished", "completed", "ceased", "stopped",
        "discharged", "died", "resigned", "retired", "abdicated",
        "surrendered", "repealed", "dismissed", "graduated", "defeated",
        "cured", "fell",
    }
    _DURATION_WORDS = DURATION_VERB_LEMMAS | {
        "lasted", "continued", "spanned", "endured", "persisted", "extended",
        "ran", "served", "reigned", "ruled", "survived",
    }

    def classify(self, text: str, timex_start: int, timex_end: int) -> RoleSignal:
        sent_start = max(text.rfind(".", 0, timex_start), text.rfind("\n", 0, timex_start)) + 1
        sent_end_idx = text.find(".", timex_end)
        sent_end = (sent_end_idx + 1) if sent_end_idx != -1 else len(text)

        before = text[sent_start:timex_start].lower()
        after = text[timex_end:sent_end].lower()
        prep_window = before[-30:].strip()

        # Check prepositions
        for prep in sorted(START_PREPS, key=len, reverse=True):
            if re.search(rf"\b{re.escape(prep)}\s*$", prep_window):
                return RoleSignal("START", prep=prep, confidence=0.8,
                                  reason=f"prep '{prep}'")
        for prep in sorted(END_PREPS, key=len, reverse=True):
            if re.search(rf"\b{re.escape(prep)}\s*$", prep_window):
                return RoleSignal("END", prep=prep, confidence=0.8,
                                  reason=f"prep '{prep}'")
        for prep in sorted(DURATION_PREPS, key=len, reverse=True):
            if re.search(rf"\b{re.escape(prep)}\s*$", prep_window):
                return RoleSignal("DURATION", prep=prep, confidence=0.7,
                                  reason=f"prep '{prep}'")
        for prep in sorted(CONTAINS_PREPS, key=len, reverse=True):
            if re.search(rf"\b{re.escape(prep)}\s*$", prep_window):
                return RoleSignal("CONTAINS", prep=prep, confidence=0.7,
                                  reason=f"prep '{prep}'")

        # Check verbs by proximity
        before_tokens = re.findall(r"[a-z]+", before)
        after_tokens = re.findall(r"[a-z]+", after)
        candidates = []

        for tok in before_tokens:
            idx = before.rfind(tok)
            dist = len(before) - idx if idx >= 0 else 999
            if tok in self._END_WORDS:
                candidates.append(("END", tok, dist, 0.75))
            elif tok in self._START_WORDS:
                candidates.append(("START", tok, dist, 0.75))
            elif tok in self._DURATION_WORDS:
                candidates.append(("DURATION", tok, dist, 0.70))

        for tok in after_tokens:
            idx = after.find(tok)
            dist = (idx if idx >= 0 else 999) + 10
            if tok in self._END_WORDS:
                candidates.append(("END", tok, dist, 0.70))
            elif tok in self._START_WORDS:
                candidates.append(("START", tok, dist, 0.70))
            elif tok in self._DURATION_WORDS:
                candidates.append(("DURATION", tok, dist, 0.65))

        if candidates:
            best = min(candidates, key=lambda c: c[2])
            return RoleSignal(best[0], verb=best[1], confidence=best[3],
                              reason=f"proximity verb '{best[1]}'")

        # Noun signals
        context = (before + " " + after).lower()
        for noun in sorted(START_NOUNS, key=len, reverse=True):
            if re.search(rf"\b{re.escape(noun)}\b", context):
                return RoleSignal("START", confidence=0.65, reason=f"noun '{noun}'")
        for noun in sorted(END_NOUNS, key=len, reverse=True):
            if re.search(rf"\b{re.escape(noun)}\b", context):
                return RoleSignal("END", confidence=0.65, reason=f"noun '{noun}'")
        for noun in sorted(DURATION_NOUNS, key=len, reverse=True):
            if re.search(rf"\b{re.escape(noun)}\b", context):
                return RoleSignal("DURATION", confidence=0.65, reason=f"noun '{noun}'")

        if re.search(r"\bon\s*$", prep_window):
            return RoleSignal("UNKNOWN", prep="on", confidence=0.3,
                              reason="ambiguous prep 'on'")

        return RoleSignal("UNKNOWN", confidence=0.2, reason="no signal found")
