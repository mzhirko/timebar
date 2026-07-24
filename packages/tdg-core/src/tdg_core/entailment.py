"""
General temporal entailment checker.

Given a RULE TDG (a statute/contract defining a temporal constraint) and an
INSTANCE TDG (a judgment/performance with concrete dates), decide whether the
instance satisfies the rule's time limit:  TIMELY, LATE, or INDETERMINATE.

Design principles (no hardcoded statute knowledge):
  - Rules are DISCOVERED from the rule TDG's additive dependencies. A statute
    that says "within 3 months of the effective date of termination" yields a
    dependency (anchor_fact + 3 months -> deadline). The engine reads the
    CALENDAR unit (months/years/days) from that dependency, not a day count, so
    "3 months" is evaluated with real calendar arithmetic, not a 90-day
    approximation.
  - The anchor (e.g. the termination date) and the action (e.g. the claim
    filing) are matched in the instance by entity/sentence similarity — the
    same general matcher used across the pipeline — never by a hardcoded
    entity name.
  - Optional ACAS-style "stop the clock" early-conciliation pause (UK
    ERA s.207B) is applied when the instance provides Day A / Day B, including
    the one-month floor. This is a general "limit extended by a paused period"
    mechanism, parameterised by the instance, not specific to one statute.

The s.111 cases are reproduced because the statute TEXT encodes the rule, not
because the rule is written into this file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict, field
from datetime import date, timedelta
from typing import Optional, Literal

from dateutil.relativedelta import relativedelta

from tdg_core.tdg import (
    TemporalDependencyGraph,
    TemporalFact,
    TemporalDependency,
)
from tdg_core.embeddings import EmbeddingSimilarity, normalise_entity
from tdg_core.cross_doc import _text_overlap, _entity_similarity


Verdict = Literal["TIMELY", "LATE", "INDETERMINATE"]


# ─── Calendar-aware offset ────────────────────────────────────────────────

@dataclass
class CalendarOffset:
    """A time offset expressed in calendar units, plus a 'minus one day' flag.

    UK limitation periods are "N months beginning with" the anchor, which in
    practice is "anchor + N months - 1 day". We keep months/years/days separate
    so the deadline is computed with real calendar arithmetic (relativedelta),
    not a 30-day-month approximation.
    """
    years: int = 0
    months: int = 0
    days: int = 0
    minus_one_day: bool = False
    # provenance of minus_one_day: "discovered" (read off the statute's own
    # connector, with the phrase in inclusivity_evidence) or "assumed" (no
    # readable connector; the UK convention was applied). Never claim a rule is
    # discovered when this says assumed.
    inclusivity_source: str = "assumed"
    inclusivity_evidence: Optional[str] = None

    def apply(self, anchor: date) -> date:
        d = anchor + relativedelta(years=self.years, months=self.months, days=self.days)
        if self.minus_one_day:
            d = d - timedelta(days=1)
        return d

    @property
    def is_zero(self) -> bool:
        return self.years == 0 and self.months == 0 and self.days == 0


_UNIT_RE = re.compile(
    r"(\d+)\s*(years?|y|months?|m|weeks?|w|days?|d)\b", re.IGNORECASE
)


def _offset_from_text(text: str) -> Optional[CalendarOffset]:
    """Parse a calendar offset from a text fragment, preserving the unit.

    Handles ISO-ish and natural forms: 'P3M', '+3m', '3 month(s)', '6y',
    '90d', '30 days', 'three months' (spelled handled upstream by extraction).
    """
    if not text:
        return None
    years = months = days = 0
    found = False
    for n, unit in _UNIT_RE.findall(text):
        n = int(n)
        u = unit.lower()
        if u.startswith("y"):
            years += n; found = True
        elif u.startswith("mo") or u == "m" or u.startswith("month"):
            months += n; found = True
        elif u.startswith("w"):
            days += n * 7; found = True
        elif u.startswith("d"):
            days += n; found = True
    return CalendarOffset(years=years, months=months, days=days) if found else None


def _offset_from_dependency(
    dep: TemporalDependency,
    facts: Optional[dict] = None,
) -> Optional[CalendarOffset]:
    """Recover a calendar offset for an additive dependency.

    Order of preference, all unit-preserving:
      1. the dependency's constraint_expr ('+3m', '3 months', ...) — BUT if
         the edge offers only a day count while an endpoint fact carries a
         month/year-denominated duration, the fact wins: LLM extraction
         routinely flattens '3 months' into '90 days' on the edge
         (unit-destroying arithmetic), and applying 90d instead of 3
         calendar months shifts statutory deadlines by up to 2 days;
      2. the duration carried by the from/to FACT (value 'P3M' or sentence
         'three months ...') — the common case, since the LLM often attaches
         the period to the duration fact and leaves the edge unitless;
      3. the fact's duration_days, or the dep's delta_days, as a raw day count
         (correct for day-denominated limits like '90 days').
    """
    facts = facts or {}

    def _fact_value_offset() -> Optional[CalendarOffset]:
        for fid in (dep.from_id, dep.to_id):
            f = facts.get(fid)
            if f is None:
                continue
            off = _offset_from_text(f.timex.value or "")
            if off is not None:
                return off
            off = _offset_from_text(f.sentence or "")
            if off is not None:
                return off
        return None

    # 1. period stated on the edge itself (with day-flattening guard)
    off = _offset_from_text(dep.constraint_expr or "")
    if off is not None:
        if off.days and not off.months and not off.years:
            fact_off = _fact_value_offset()
            if fact_off is not None and (fact_off.months or fact_off.years):
                return fact_off  # edge is a flattened day count; fact keeps the unit
        return off

    # 2. period carried by the endpoint facts (value first, then sentence)
    off = _fact_value_offset()
    if off is not None:
        return off

    # 3. raw day-count fallbacks
    for fid in (dep.from_id, dep.to_id):
        f = facts.get(fid)
        if f is not None and f.timex.duration_days:
            return CalendarOffset(days=f.timex.duration_days)
    if dep.delta_days is not None and dep.delta_days > 0:
        return CalendarOffset(days=dep.delta_days)
    return None


# ─── Data structures ──────────────────────────────────────────────────────

@dataclass
class TemporalRule:
    """A temporal constraint discovered from a rule TDG's dependency structure."""
    anchor_fact: TemporalFact
    target_fact: TemporalFact
    offset: CalendarOffset
    source_dep: TemporalDependency
    source_doc: str

    @property
    def description(self) -> str:
        anchor = normalise_entity(self.anchor_fact.entity)
        target = normalise_entity(self.target_fact.entity)
        parts = []
        if self.offset.years:
            parts.append(f"{self.offset.years}y")
        if self.offset.months:
            parts.append(f"{self.offset.months}m")
        if self.offset.days:
            parts.append(f"{self.offset.days}d")
        off = "+".join(parts) if parts else "0"
        tail = " - 1 day" if self.offset.minus_one_day else ""
        return f"{target} = {anchor} + {off}{tail}"


@dataclass
class EntailmentResult:
    rule_doc: str
    instance_doc: str
    rule_description: str
    anchor_date: Optional[str]
    deadline_computed: Optional[str]
    action_date: Optional[str]
    days_over: Optional[int]        # >0 late, <=0 timely
    verdict: Verdict
    explanation: str
    match_confidence: float
    acas_applied: bool = False
    # Structured derivation (Phase 1.6): every input the answer was built
    # from, including what was assumed vs discovered and what was passed
    # over. Rendered by tdg_core.trace; the trace IS the product (D3).
    trace: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ─── Rule discovery ───────────────────────────────────────────────────────

# ─── Re-anchoring ─────────────────────────────────────────────────────────
# A limitation period names its own anchor in its text:
#   "within N <unit> [beginning with|from|of|starting with] <EVENT>".
# Extraction sometimes attaches the period to a generic entity (e.g. a
# "complaint") instead of <EVENT>. This pass recovers <EVENT> from the
# period's own sentence and re-points the rule's anchor to the fact that
# best matches it. General to the phrasing of limitation clauses, not to any
# particular statute.

_PERIOD_RE = re.compile(
    r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+"
    r"(?:year|month|week|day)s?\b", re.IGNORECASE)

_ANCHOR_AFTER_RE = re.compile(
    r"\b(?:beginning with|starting with|from the date(?:\s+on\s+which)?|"
    r"from|of|after|following)\b\s+(.+)", re.IGNORECASE)

# ─── Inclusivity of the anchor day ────────────────────────────────────────
# The connector that names the anchor also fixes whether the anchor DAY is
# counted, and that decides the -1 day:
#
#   "three months BEGINNING WITH the effective date of termination"
#       -> the EDT is day 1 of the period -> EDT + 3 months - 1 day
#   "three months FROM the date of the act"
#       -> the period runs from the day AFTER -> date + 3 months
#
# This is a drafting convention of the statute, readable in its own text, so
# it is discovered rather than supplied. Previously `minus_one_day` arrived as
# a caller argument (in practice a flag in the gold file), which meant the most
# legally distinctive part of the rule was the one part a human set by hand.
# The phrases below are the same alternation _ANCHOR_AFTER_RE already matches;
# we simply stop discarding which one fired.
_INCLUSIVE_CONN = ("beginning with", "starting with", "commencing with",
                   "beginning on", "starting on")
_EXCLUSIVE_CONN = ("from the date on which", "from the date", "after the date",
                   "following the date", "from", "after", "following")


def anchor_day_inclusive(sentence: str) -> Optional[bool]:
    """Does the anchor day count as day 1? Read from the period's own clause.

    Returns True (inclusive -> minus one day), False (exclusive -> plain
    offset), or None when the clause names no connector we recognise, in which
    case the caller's default stands and the rule is marked undiscovered.
    """
    if not sentence:
        return None
    m = _PERIOD_RE.search(sentence)
    if not m:
        return None
    tail = sentence[m.end():].lower()
    hits = [(tail.find(c), c, True) for c in _INCLUSIVE_CONN if c in tail]
    hits += [(tail.find(c), c, False) for c in _EXCLUSIVE_CONN if c in tail]
    if not hits:
        return None
    # the connector NEAREST the period governs it; longest match wins ties so
    # that "from the date on which" is not read as a bare "from".
    hits.sort(key=lambda h: (h[0], -len(h[1])))
    return hits[0][2]


def _inclusivity_evidence(sentence: str) -> Optional[str]:
    """The phrase the inclusivity decision was read from, for the audit trail."""
    if not sentence:
        return None
    m = _PERIOD_RE.search(sentence)
    if not m:
        return None
    tail = sentence[m.end():]
    low = tail.lower()
    hits = [(low.find(c), c) for c in _INCLUSIVE_CONN + _EXCLUSIVE_CONN
            if c in low]
    if not hits:
        return None
    hits.sort(key=lambda h: (h[0], -len(h[1])))
    i, c = hits[0]
    return (m.group(0) + tail[:i + len(c)]).strip()


def _anchor_phrase_from_period(sentence: str) -> Optional[str]:
    """Pull the noun phrase naming the anchor event from a period clause."""
    if not sentence:
        return None
    m = _PERIOD_RE.search(sentence)
    if not m:
        return None
    a = _ANCHOR_AFTER_RE.search(sentence[m.end():])
    if not a:
        return None
    tail = a.group(1).strip().rstrip(".")
    tail = re.split(r"[,;]|\bor\b|\bunless\b|\bwithin\b|\bto which\b|\bon which\b",
                    tail)[0].strip()
    tail = re.sub(r"^(the|a|an)\s+", "", tail, flags=re.IGNORECASE).strip()
    return tail or None


def _reanchor_rule(
    rule: TemporalRule,
    facts: list[TemporalFact],
    embedder: Optional[EmbeddingSimilarity] = None,
    min_gain: float = 0.2,
) -> TemporalRule:
    """If the rule's period names an anchor event in its text, re-point the
    anchor to the fact that best matches that phrase (when clearly better than
    the current anchor). Returns the rule unchanged if no better anchor found."""
    # The period text can live on the dependency edge, the anchor fact, or the
    # target fact — check all three.
    phrase = None
    for txt in (rule.source_dep.constraint_expr,
                rule.anchor_fact.sentence, rule.anchor_fact.timex.value,
                rule.target_fact.sentence, rule.target_fact.timex.value):
        phrase = _anchor_phrase_from_period(txt or "")
        if phrase:
            break
    if not phrase:
        return rule

    cur = normalise_entity(rule.anchor_fact.entity)
    cur_score = _text_overlap(phrase, cur)
    if embedder is not None:
        cur_score = max(cur_score, embedder.similarity(phrase, cur))

    best = None
    best_score = cur_score + min_gain   # require a clear improvement
    for f in facts:
        ent = normalise_entity(f.entity)
        score = _text_overlap(phrase, ent)
        # also consider the fact's sentence (the EDT fact's sentence often
        # repeats "effective date of termination")
        score = max(score, _text_overlap(phrase, f.sentence or ""))
        if embedder is not None:
            score = max(score, embedder.similarity(phrase, ent))
        if score > best_score:
            best_score = score
            best = f

    if best is not None and best.id != rule.anchor_fact.id:
        return TemporalRule(
            anchor_fact=best, target_fact=rule.target_fact,
            offset=rule.offset, source_dep=rule.source_dep,
            source_doc=rule.source_doc,
        )

    # No statute fact matches the anchor phrase (common: the statute states the
    # period but the anchoring EVENT, e.g. the EDT, is only named in prose and
    # was not extracted as its own fact). The anchoring DATE lives in the
    # instance document anyway, so synthesize a phrase-anchor labelled with the
    # event. _match_fact in the instance will then find the case's EDT fact.
    if cur_score < 0.4:   # current anchor is a poor label (e.g. "complaint")
        from tdg_core.tdg import TimexSpan
        synthetic = TemporalFact(
            id=f"{rule.anchor_fact.id}__reanchored",
            entity=phrase, role=rule.anchor_fact.role,
            timex=TimexSpan(text=phrase, timex_type="DATE", value=None,
                            start_char=0, end_char=0, date_parsed=None,
                            duration_days=None),
            sentence=phrase,
        )
        return TemporalRule(
            anchor_fact=synthetic, target_fact=rule.target_fact,
            offset=rule.offset, source_dep=rule.source_dep,
            source_doc=rule.source_doc,
        )
    return rule


def find_rules(
    tdg: TemporalDependencyGraph,
    minus_one_day: Optional[bool] = None,
    reanchor: bool = True,
    embedder: Optional[EmbeddingSimilarity] = None,
) -> list[TemporalRule]:
    """Extract temporal rules from a rule TDG's additive dependencies.

    The N and the unit come from the statute's own extracted dependency. The
    -1 day now does too: `anchor_day_inclusive()` reads it off the connector in
    the period's own clause ("three months BEGINNING WITH the effective date"
    -> the anchor day counts -> minus one day; "three months FROM the date"
    -> it does not). So the whole rule is discovered, including the part that
    makes it legally distinctive.

    `minus_one_day` is now only a FALLBACK for clauses whose connector we
    cannot read (None -> assume the UK "beginning with" convention, which is
    what every limitation period in scope uses). Pass True/False to force the
    old behaviour of applying one value uniformly. Each rule records
    `inclusivity_source` so the audit can tell a discovered -1 day from an
    assumed one.
    """
    fact_map = {f.id: f for f in tdg.facts}
    rules: list[TemporalRule] = []
    # the period clause is the sentence of whichever fact carries the DURATION;
    # fall back to scanning every sentence in the graph.
    sentences = [f.sentence for f in tdg.facts if getattr(f, "sentence", None)]
    for dep in tdg.dependencies:
        if dep.constraint_type != "additive":
            continue
        offset = _offset_from_dependency(dep, fact_map)
        if offset is None or offset.is_zero:
            continue
        anchor = fact_map.get(dep.from_id)
        target = fact_map.get(dep.to_id)
        if anchor is None or target is None:
            continue

        # discover inclusivity: prefer the clause of the fact that states the
        # period, then the rule's own facts, then any sentence in the statute.
        inc, evidence = None, None
        cands = [getattr(f, "sentence", None) for f in (anchor, target)
                 if getattr(f, "sentence", None)] + sentences
        for snt in cands:
            inc = anchor_day_inclusive(snt)
            if inc is not None:
                evidence = _inclusivity_evidence(snt)
                break
        if inc is None:
            offset.minus_one_day = True if minus_one_day is None else minus_one_day
            offset.inclusivity_source = "assumed"
            offset.inclusivity_evidence = None
        else:
            offset.minus_one_day = inc
            offset.inclusivity_source = "discovered"
            offset.inclusivity_evidence = evidence
        rule = TemporalRule(
            anchor_fact=anchor, target_fact=target, offset=offset,
            source_dep=dep, source_doc=tdg.document_id,
        )
        if reanchor:
            rule = _reanchor_rule(rule, tdg.facts, embedder)
        rules.append(rule)
    return rules


# ─── Fact matching ────────────────────────────────────────────────────────

# ─── Legal vocabulary: loaded from DATA, never hardcoded here ─────────────
# The engine's arithmetic and rule discovery contain no statute knowledge.
# These alias/cue sets only widen fuzzy matching between a statute's wording
# and a judgment's wording. They ship as a JSON data file
# (tdg_core/data/legal_aliases.uk.json) and can be extended or replaced per
# rule pack (aliases.json) via load_alias_file().

_ANCHOR_ALIASES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
_ACTION_CUES: tuple[str, ...] = ()
_PROCEDURAL_NOISE: tuple[str, ...] = ()
_ALIAS_SOURCES: list[str] = []   # provenance of loaded vocabulary, for the trace


def register_aliases(concept, plain, end_role_only=None) -> None:
    """Add/replace one anchor concept's alias set (used by rule packs)."""
    _ANCHOR_ALIASES[concept.lower()] = (
        tuple(a.lower() for a in plain),
        tuple(a.lower() for a in (end_role_only or [])))


def load_alias_dict(data: dict, *, replace: bool = False,
                    source: str = "<dict>") -> None:
    global _ACTION_CUES, _PROCEDURAL_NOISE
    if replace:
        _ANCHOR_ALIASES.clear()
        _ACTION_CUES = ()
        _PROCEDURAL_NOISE = ()
        _ALIAS_SOURCES.clear()
    _ALIAS_SOURCES.append(source)
    for concept, sets in data.get("anchor_aliases", {}).items():
        register_aliases(concept, sets.get("plain", []), sets.get("end_role_only", []))
    _ACTION_CUES = tuple(dict.fromkeys(
        _ACTION_CUES + tuple(c.lower() for c in data.get("action_cues", []))))
    _PROCEDURAL_NOISE = tuple(dict.fromkeys(
        _PROCEDURAL_NOISE + tuple(c.lower() for c in data.get("procedural_noise", []))))


def load_alias_file(path, *, replace: bool = False) -> None:
    """Load a rule pack's aliases.json on top of (or instead of) the defaults."""
    import json as _json
    from pathlib import Path as _Path
    load_alias_dict(_json.loads(_Path(path).read_text()), replace=replace,
                    source=str(path))


def _load_default_aliases() -> None:
    """LANGUAGE-layer defaults only (generic English procedural cues).
    The engine ships with NO jurisdiction vocabulary; concept aliases
    arrive from a rule pack's aliases.json."""
    import json as _json
    from importlib import resources as _resources
    with _resources.files("tdg_core.data").joinpath("action_cues.en.json").open() as f:
        load_alias_dict(_json.load(f), source="tdg_core/data/action_cues.en.json")


_load_default_aliases()


def _alias_score(rule_phrase: str, cand: "TemporalFact") -> float:
    """Alias-aware anchor score: 1.0 for a verbatim concept label in the
    instance, 0.75 for a statutory alias (END-role enforced where the
    alias is a period concept). 0.0 when no alias applies."""
    rp = normalise_entity(rule_phrase).lower()
    ce = normalise_entity(cand.entity).lower()
    if rp and rp == ce:
        return 1.0
    for concept, (plain, end_only) in _ANCHOR_ALIASES.items():
        if concept not in rp:
            continue
        if any(a == ce or a in ce for a in plain):
            if any(n in ce for n in _PROCEDURAL_NOISE) and not any(a == ce for a in plain):
                continue
            return 0.75
        role = (getattr(cand, "role", "") or "").upper()
        if role == "END" and any(a == ce or a in ce for a in end_only):
            if any(n in ce for n in _PROCEDURAL_NOISE) and not any(a == ce for a in end_only):
                continue
            return 0.75
    return 0.0


def _match_fact(
    rule_fact: TemporalFact,
    candidates: list[TemporalFact],
    embedder: Optional[EmbeddingSimilarity] = None,
    min_score: float = 0.15,
) -> Optional[tuple[TemporalFact, float]]:
    """Best instance fact matching a rule fact, by entity + sentence overlap."""
    best = None
    best_score = 0.0
    for cand in candidates:
        entity_sim = _entity_similarity(rule_fact.entity, cand.entity, embedder)
        sentence_sim = _text_overlap(rule_fact.sentence or "", cand.sentence or "")
        score = entity_sim * 0.5 + sentence_sim * 0.5
        if normalise_entity(rule_fact.entity) == normalise_entity(cand.entity):
            score = max(score, entity_sim)
        score = max(score, _alias_score(rule_fact.entity, cand))
        if score > best_score and score >= min_score:
            best_score = score
            best = (cand, score)
    return best


def _select_action(
    rule: TemporalRule,
    post_anchor: list[TemporalFact],
    embedder: Optional[EmbeddingSimilarity],
) -> Optional[tuple[TemporalFact, float, str, list]]:
    """Choose the action fact whose date is compared against the deadline.

    The action is the event the limitation period governs (e.g. *presenting
    the complaint*), not merely the earliest thing that looks like an action.
    Each post-anchor fact is scored by how well it matches the rule's TARGET
    concept (entity + sentence), with a bonus for containing a generic action
    cue (claim/complaint/presented/...). The highest-scoring fact wins; ties
    break to the later date (the operative filing, not a preliminary step).
    Returns (fact, confidence, how).
    """
    # Build the concept text the action should resemble: the rule target's
    # entity + sentence (statute words like "complaint ... presented to the
    # tribunal"). This is the thing the deadline is about.
    target_blob = f"{normalise_entity(rule.target_fact.entity)} " \
                  f"{rule.target_fact.sentence or ''}".strip()

    def _cue_bonus(f: TemporalFact) -> float:
        ent = normalise_entity(f.entity).lower()
        return 0.15 if any(cue in ent for cue in _ACTION_CUES) else 0.0

    scored: list[tuple[float, TemporalFact]] = []
    for f in post_anchor:
        ent = normalise_entity(f.entity)
        s = _text_overlap(target_blob, ent)
        s = max(s, _text_overlap(target_blob, f.sentence or ""))
        if embedder is not None:
            s = max(s, embedder.similarity(target_blob, ent))
        s += _cue_bonus(f)
        scored.append((s, f))

    if not scored:
        return None

    # prefer cue-bearing candidates outright: the governed act is the
    # presentation of the claim, not a later procedural event that happens
    # to resemble statute wording.
    cue_scored = [(sc, f) for sc, f in scored if _cue_bonus(f) > 0]
    pool = cue_scored if cue_scored else scored
    best_score = max(sc for sc, _ in pool)
    floor = 0.0 if cue_scored else 0.25
    if best_score >= floor:
        # earliest within a BAND of the best, not only exact ties. Two
        # mentions of the governed act (the ET1 and a later re-filing of
        # the same claim) score within noise of each other, and the
        # limitation period governs the FIRST presentation; exact-tie
        # earliest let a hair of extra sentence overlap hand the verdict
        # to the re-filing (observed: ndow, 28/101 wrong verdicts with a
        # correct anchor and a correct deadline). The band is narrower
        # than the cue bonus (0.15), so an unrelated cue-bearing claim
        # (e.g. a personal-injury claim in county court) that scores a
        # full tier below the target-matching fact stays outside it
        # (observed: pepkolaj). Validated offline against the 427 saved
        # extractions of cf_anchor_pipeline_gemma: 0.688 -> 0.777
        # answered accuracy, no case degraded, band stable on [0.10,0.15].
        _BAND = 0.10
        winners = [f for sc, f in pool if sc >= best_score - _BAND]
        # ties -> EARLIEST post-anchor date: a limitation period governs the
        # FIRST presentation of the complaint; later filings/hearings are not
        # the governed act. (Replaces the latest-tie-break, whose fragility
        # on appellate documents was pre-registered 24/05 and observed 06/07.)
        f = min(winners, key=lambda x: x.timex.date_parsed)
        how = "target-match" if best_score >= 0.4 else "action-cue"
        return f, min(best_score, 1.0), how, scored

    # weak fallback: nothing resembles the target. The operative act for a
    # "present within N of X" rule is normally the LAST post-anchor event
    # (you file at the end of the process), so prefer latest, low confidence.
    f = max(post_anchor, key=lambda x: x.timex.date_parsed)
    return f, 0.2, "latest-fallback", scored


# ─── ACAS / early-conciliation pause (UK ERA s.207B) ──────────────────────

def _apply_conciliation(
    deadline: date,
    day_a: Optional[date],
    day_b: Optional[date],
) -> tuple[date, bool]:
    """Extend a deadline by a paused conciliation period.

    Implements the ERA s.207B mechanism generally:
      - the period from the day after Day A to Day B is not counted
        (deadline pushed out by that many days);
      - floor: if the (unextended) limit would expire between Day A and one
        month after Day B, it expires one month after Day B instead.
    If Day A is on/after the original deadline, conciliation gives no benefit.
    Returns (effective_deadline, applied?).
    """
    if day_a is None or day_b is None:
        return deadline, False
    if day_a >= deadline:
        # contacted ACAS after the limit already expired -> no freeze
        return deadline, False
    paused_days = (day_b - day_a).days  # days after Day A up to and incl Day B
    extended = deadline + timedelta(days=paused_days)
    one_month_after_b = day_b + relativedelta(months=1)
    if extended < one_month_after_b:
        extended = one_month_after_b
    return extended, True


# ─── Entailment checking ─────────────────────────────────────────────────

def check_entailment(
    rule_tdg: TemporalDependencyGraph,
    instance_tdg: TemporalDependencyGraph,
    embedder: Optional[EmbeddingSimilarity] = None,
    minus_one_day: Optional[bool] = None,
    acas_day_a: Optional[date] = None,
    acas_day_b: Optional[date] = None,
) -> list[EntailmentResult]:
    """Check whether instance_tdg satisfies each temporal rule in rule_tdg.

    minus_one_day defaults to None = discover it from the statute's wording
    (see find_rules). Pass True/False only to force one value uniformly.
    """
    rules = find_rules(rule_tdg, minus_one_day=minus_one_day, embedder=embedder)
    if not rules:
        return []
    dated = [f for f in instance_tdg.facts if f.timex.date_parsed]
    if not dated:
        return [EntailmentResult(
            rule_doc=rule_tdg.document_id, instance_doc=instance_tdg.document_id,
            rule_description=r.description, anchor_date=None,
            deadline_computed=None, action_date=None, days_over=None,
            verdict="INDETERMINATE",
            explanation="No dated facts in instance document",
            match_confidence=0.0,
            trace={
                "rule": _rule_trace(r),
                "request": (
                    f"No dated facts found. Supply at least the "
                    f"'{normalise_entity(r.anchor_fact.entity)}' date, "
                    f"or point me at the clause that states it."),
            },
        ) for r in rules]
    return [
        _check_single_rule(r, dated, instance_tdg, embedder, acas_day_a, acas_day_b)
        for r in rules
    ]


def _rule_trace(rule: TemporalRule) -> dict:
    """The statute side of the derivation: where every number came from."""
    off = rule.offset
    return {
        "source_doc": rule.source_doc,
        "description": rule.description,
        "statute_sentence": rule.target_fact.sentence or rule.anchor_fact.sentence or "",
        "anchor_concept": normalise_entity(rule.anchor_fact.entity),
        "target_concept": normalise_entity(rule.target_fact.entity),
        "offset": {"years": off.years, "months": off.months, "days": off.days,
                   "minus_one_day": off.minus_one_day},
        "inclusivity_source": off.inclusivity_source,      # discovered | assumed
        "inclusivity_evidence": off.inclusivity_evidence,  # the phrase, or None
        "constraint_expr": rule.source_dep.constraint_expr,
        "vocabulary_sources": list(_ALIAS_SOURCES),
    }


def _check_single_rule(
    rule: TemporalRule,
    dated_facts: list[TemporalFact],
    instance_tdg: TemporalDependencyGraph,
    embedder: Optional[EmbeddingSimilarity],
    day_a: Optional[date],
    day_b: Optional[date],
) -> EntailmentResult:
    trace: dict = {"rule": _rule_trace(rule)}

    def indet(msg: str, conf: float = 0.0,
              anchor: Optional[date] = None, deadline: Optional[date] = None,
              request: Optional[str] = None):
        if request:
            trace["request"] = request
        return EntailmentResult(
            rule_doc=rule.source_doc, instance_doc=instance_tdg.document_id,
            rule_description=rule.description,
            anchor_date=anchor.isoformat() if anchor else None,
            deadline_computed=deadline.isoformat() if deadline else None,
            action_date=None, days_over=None, verdict="INDETERMINATE",
            explanation=msg, match_confidence=conf, trace=dict(trace),
        )

    anchor_match = _match_fact(rule.anchor_fact, dated_facts, embedder)
    rp_l = normalise_entity(rule.anchor_fact.entity).lower()
    if any(concept in rp_l for concept in _ANCHOR_ALIASES):
        # the rule names a defined statutory concept: require verbatim or
        # alias-tier evidence; generic fuzzy similarity may not stand in.
        if anchor_match is not None:
            f, sc = anchor_match
            if (normalise_entity(f.entity).lower() != rp_l
                    and _alias_score(rule.anchor_fact.entity, f) < 0.75):
                anchor_match = None
    if anchor_match is None:
        _c = normalise_entity(rule.anchor_fact.entity)
        return indet(
            f"Could not match anchor '{_c}' in instance",
            request=(f"Supply a dated fact for '{_c}', or point me at the "
                     f"clause in the instance that states it."),
        )
    anchor_fact, anchor_conf = anchor_match

    # Conflict gate: if several near-best candidates carry DIFFERENT dates,
    # anchor selection is a live legal judgment (e.g. dismissal vs internal
    # appeal outcome) — abstain and surface the candidates rather than
    # silently choosing. A verbatim concept label in the document overrides
    # (the document itself names the statutory concept).
    rp = normalise_entity(rule.anchor_fact.entity).lower()
    verbatim = [f for f in dated_facts
                if normalise_entity(f.entity).lower() == rp]
    if not verbatim:
        near = {}
        for cand in dated_facts:
            sc = _alias_score(rule.anchor_fact.entity, cand)
            if sc >= 0.75 and cand.timex.date_parsed:
                near[cand.timex.date_parsed] = normalise_entity(cand.entity)
        if len(near) > 1:
            opts = "; ".join(f"{e} ({d.isoformat()})"
                             for d, e in sorted(near.items()))
            return indet(
                f"Anchor ambiguous — {len(near)} distinct-dated candidates "
                f"for '{normalise_entity(rule.anchor_fact.entity)}': {opts}. "
                f"Anchor selection is a legal judgment; abstaining.",
                conf=0.1,
                request=("State which candidate is the controlling "
                         f"'{normalise_entity(rule.anchor_fact.entity)}', "
                         "with its quote."),
            )
    anchor_date = anchor_fact.timex.date_parsed
    trace["anchor_match"] = {
        "fact_id": anchor_fact.id,
        "entity": normalise_entity(anchor_fact.entity),
        "date": anchor_date.isoformat(),
        "confidence": round(anchor_conf, 3),
        "quote": anchor_fact.sentence or "",
    }

    deadline_base = rule.offset.apply(anchor_date)
    deadline, acas_applied = _apply_conciliation(deadline_base, day_a, day_b)
    trace["arithmetic"] = {
        "anchor_date": anchor_date.isoformat(),
        "deadline_base": deadline_base.isoformat(),
        "deadline_effective": deadline.isoformat(),
    }
    if acas_applied:
        trace["conciliation"] = {
            "day_a": day_a.isoformat(), "day_b": day_b.isoformat(),
            "extension_days": (deadline - deadline_base).days,
        }

    post_anchor = [
        f for f in dated_facts
        if f.timex.date_parsed > anchor_date and f.id != anchor_fact.id
    ]
    if not post_anchor:
        return indet(
            f"Anchor matched ({normalise_entity(anchor_fact.entity)}, "
            f"{anchor_date.isoformat()}); deadline {deadline.isoformat()}; "
            f"no action fact after anchor",
            conf=anchor_conf, anchor=anchor_date, deadline=deadline,
            request=(f"Supply the dated fact for the governed act "
                     f"('{normalise_entity(rule.target_fact.entity)}') — "
                     f"it must fall after {anchor_date.isoformat()}."),
        )

    action_fact, action_conf, how, considered = _select_action(rule, post_anchor, embedder)
    action_date = action_fact.timex.date_parsed
    trace["action_selection"] = {
        "selected": {"fact_id": action_fact.id,
                     "entity": normalise_entity(action_fact.entity),
                     "date": action_date.isoformat(),
                     "score": round(action_conf, 3), "how": how,
                     "quote": action_fact.sentence or ""},
        "passed_over": [
            {"fact_id": f.id, "entity": normalise_entity(f.entity),
             "date": f.timex.date_parsed.isoformat(), "score": round(sc, 3)}
            for sc, f in sorted(considered, key=lambda t: -t[0])
            if f.id != action_fact.id
        ],
    }
    days_over = (action_date - deadline).days
    verdict: Verdict = "TIMELY" if days_over <= 0 else "LATE"
    match_conf = (anchor_conf + action_conf) / 2

    explanation = (
        f"anchor={normalise_entity(anchor_fact.entity)} ({anchor_date.isoformat()})"
        f" + [{rule.description.split('= ',1)[-1]}] = deadline {deadline.isoformat()}"
        f"{' (conciliation-extended)' if acas_applied else ''}. "
        f"action={normalise_entity(action_fact.entity)} ({action_date.isoformat()}, "
        f"via {how}) -> {abs(days_over)}d "
        f"{'after' if days_over > 0 else 'before/at'} deadline -> {verdict}"
    )
    trace["arithmetic"]["action_date"] = action_date.isoformat()
    trace["arithmetic"]["margin_days"] = days_over
    return EntailmentResult(
        rule_doc=rule.source_doc, instance_doc=instance_tdg.document_id,
        rule_description=rule.description,
        anchor_date=anchor_date.isoformat(),
        deadline_computed=deadline.isoformat(),
        action_date=action_date.isoformat(),
        days_over=days_over, verdict=verdict, explanation=explanation,
        match_confidence=match_conf, acas_applied=acas_applied, trace=trace,
    )
