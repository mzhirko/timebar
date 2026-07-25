"""
LLM-based TDG extraction pipeline.

Replaces HeidelTime + spaCy role classifier + entity linker with a single
LLM call that extracts events, dates, and relations in one shot.

Everything downstream (graph_builder, scenario_generator, tdg.py) stays the same.

Usage:
    # Hosted API (requires OPENAI_API_KEY)
    from llm_pipeline import LLMPipeline
    pipe = LLMPipeline(model="gpt-4o-mini")

    # Local Ollama, or any OpenAI-compatible endpoint (no real key needed)
    pipe = LLMPipeline(model="llama3", base_url="http://localhost:11434/v1")

    # There is no default model: pass one, or set TDG_LLM_MODEL.

    tdg = pipe.process(
        text="This Agreement is effective January 15, 2025...",
        document_id="contract_001",
        document_type="legal",
    )
    print(tdg.summary())

Requirements:
    pip install openai datasets
    For OpenAI also: pip install python-dotenv  +  .env with OPENAI_API_KEY=sk-...
    For Ollama: ollama pull gemma4:e4b  (no extra packages needed)
"""

from __future__ import annotations

import json
import os
import re
from datetime import date, datetime
from typing import Optional

from openai import OpenAI

from tdg_core.tdg import (
    TemporalDependencyGraph,
    TemporalDependency,
    TemporalFact,
    TimexSpan,
)
from tdg_core.graph_builder import GraphBuilder
from tdg_extractors.scenario_generator import generate_edit_scenarios


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a temporal information extraction system for legal documents.

Your job is to extract:
1. EVENTS — things that happen at a specific time or over a duration
2. RELATIONS — arithmetic or logical dependencies between events

━━━ RULES FOR EVENTS ━━━
- Only extract events grounded in the text. Do not invent events.

ENTITY NAMING:
- "entity": the legal concept this event is about. Use the concept name from the text, 1-4 words.
- GOOD entity names: "consultations", "notification deadline", "Agreement", "early conciliation", "filing deadline", "effective date of termination", "grievance period"
- BAD entity names:
  FAIL: "date" — too generic, says nothing about what the date is for
  FAIL: "period" — too generic
  FAIL: "termination" alone when the text says "effective date of termination" — use the full legal concept
  FAIL: An entire sentence or document title as the entity name
  FAIL: Article numbers in the entity — put those in article_ref instead
- Keep entity names STABLE across the document: if the same concept appears in multiple clauses, use the same entity name every time. "consultations" in Article 4 and "consultations" in Article 9 should both have entity="consultations".

- "article_ref": the article/paragraph reference for this specific clause (e.g. "Art.2.5", "Art.4.1", "Art.11.3"). Use null if the event is not tied to a specific article.

ROLE ASSIGNMENT:
- "role": one of START, END, DURATION, CONTAINS
- Roles describe the temporal position WITHIN the entity's lifecycle:
  - START: when the entity begins (signed, entered into force, commenced, filed, effective from)
  - END: when the entity concludes (expired, terminated, concluded, paid by, dismissed)
  - DURATION: how long the entity lasts OR a waiting/notice period (3 years, within 30 days, no later than 6 months)
  - CONTAINS: a point event that falls within an ongoing period

- PERSPECTIVE RULE: choose the role based on what ENDS or STARTS, not on the word "termination":
  - "effective date of termination" = END of employment (employment ends on this date)
  - "dismissal date" = END of employment (employment ends)
  - "complaint filed on August 24" = CONTAINS (a point event within the complaint process)
  - "entry into force on January 1" = START of the agreement
  - "notice period of 6 months" = DURATION of the notice requirement

- WARNING: NOT a DURATION: percentages ("up to 10%"), quantities ("5 607 million EUA"), counts ("90% of limits")
- WARNING: NOT a DURATION: vague phrases with no time unit ("immediately", "at any time", "upon request")
- A specific calendar date like '31 December 2006' is always a DATE, never a DURATION.

DATE vs DURATION — EU LEGAL DOCUMENT PATTERNS:
  EU legal documents use two distinct temporal forms. Do NOT confuse them:

  CALENDAR DATES — always output as YYYY-MM-DD (or null if the year is unknown):
    "31 March of the following year" → date: null (year unknown), role: END or CONTAINS
    "1 January 1995" → date: "1995-01-01"
    "28 February 1995" → date: "1995-02-28"
    "31 December 2006" → date: "2006-12-31"
    WARNING: "31 March" is a DATE (March 31st), NOT a duration of 31 days.
    WARNING: "28 February" is a DATE (Feb 28th), NOT P28D.
    WARNING: The number before a month name is ALWAYS a day-of-month, never a day count.

  DURATION PERIODS — always output as ISO 8601 (P__D, P__M, P__Y):
    "one month" → P1M
    "six months" → P6M
    "30 days" → P30D
    "120 days" → P120D
    "three years" → P3Y
    Duration patterns always use a time-unit word (days, months, years), never a month name.

DATE ACCURACY:
  When converting a written date to ISO format, verify the year matches the source:
    "19 July 1986" → "1986-07-19" (year is 1986, NOT 1979)
    "27 November 1992" → "1992-11-27" (year is 1992)
    "22 December 1994" → "1994-12-22"
  The day number and year number are different — do not swap them.

- "date": ISO format YYYY-MM-DD for point-in-time dates, ISO 8601 duration for periods (P30D, P1M, P5Y). Use null if the date cannot be resolved to a specific YYYY-MM-DD.
- "raw_text": exact phrase copied from the source

━━━ RULES FOR RELATIONS ━━━
Use these types PRECISELY:

  additive — one date is computed by adding a duration to another event.
    Trigger phrases: "within X days of", "X months after", "no later than X from",
                     "within X from the [event]", "before X days of", "X of their [event]"
    → ALWAYS use additive when you see a duration anchored to another event.
    → The anchor event may be in a DIFFERENT SENTENCE. Look for back-references:
      - "from the date of the request" → links to the request event
      - "of their commencement" → links to the commencement event
      - "from the entry into force" → links to the entry-into-force event
      - "after the notice deadline" → links to the notice deadline event
      - "the consultations referred to in paragraph N" → links to that event
    → Compute delta_days: 1 month = 30 days, 1 year = 365 days.
    → Write constraint as: "to_event = from_event + N days"

    Single-sentence example:
      "notify within one month after entry into force"
      → from_id: entry_into_force, to_id: notification_deadline
      → type: additive, delta_days: 30

    Multi-sentence chain example (this is common in legal procedures):
      Sentence A: "any request shall be notified in writing to the other Party"    → e1 (request)
      Sentence B: "consultations shall begin within one month from the date of
                   the request"                                                    → e2 (begin, P1M)
      Sentence C: "consultations shall arrive at a result within one month of
                   their commencement"                                             → e3 (result, P1M)
      Relations:
        e1 → e2, type: additive, delta_days: 30  ("from the date of the request" = e1)
        e2 → e3, type: additive, delta_days: 30  ("of their commencement" = e2)

  ordering — one event explicitly precedes another, with no computable offset.
    Use when the text states sequencing but gives NO duration:
      "following X, Y happens", "after X, Y shall...", "pending the results of X",
      "upon completion of X", "before the expiration of this Agreement"
    Do NOT use ordering just because two events appear near each other.
    Do NOT invent relations between events that have no textual link.
    Example: "following the consultations, the Community shall instigate procedures"
      → type: ordering, delta_days: null

  interval — one event occurs during a defined period.
    Example: "a hearing held during the trial period"
      → type: interval, delta_days: null

IMPORTANT:
- Prefer additive over ordering whenever a duration phrase connects two events.
- Do NOT use ordering when the text gives an explicit offset like "within 30 days".
- delta_days must be an integer (not null) for all additive relations.
- LOOK FOR CHAINS: legal procedures often define multi-step sequences across
  consecutive sentences. Connect them. A relation is valid even when the two
  events are in different sentences, as long as one sentence references the other.

━━━ RULES FOR PARTIES ━━━
- "parties": every named person, company or body this document is ABOUT.
  Include claimants, respondents, employers, employees, appellants.
- Use the fullest form the text gives, e.g. "Northgate Logistics Ltd", "Ms A. Okafor".
- EXCLUDE: courts and tribunals, judges, the drafting solicitors, statute
  names, and anyone mentioned only in passing as an authority or precedent.
- Return [] when the document names no parties. An empty list is correct and
  useful; do not guess.
- These identify which case the document belongs to, so accuracy matters more
  than completeness — a wrong name is worse than a missing one.

━━━ OUTPUT FORMAT ━━━
Return ONLY valid JSON. No explanation, no markdown, no code fences.

{
  "parties": ["Ms A. Okafor", "Northgate Logistics Ltd"],
  "events": [
    {
      "id": "e1",
      "description": "short description",
      "entity": "concept this event is about",
      "article_ref": "Art.X.Y or null",
      "role": "START|END|DURATION|CONTAINS",
      "date": "YYYY-MM-DD or P30D or P1M etc.",
      "raw_text": "exact phrase from source",
      "sentence": "full sentence this came from"
    }
  ],
  "relations": [
    {
      "from_id": "e1",
      "to_id": "e2",
      "type": "additive|ordering|interval",
      "delta_days": 30,
      "constraint": "e2 = e1 + 30 days"
    }
  ]
}"""

USER_PROMPT_TEMPLATE = """Extract temporal events and relations from this {document_type} document.

Text:
{text}
"""


# ---------------------------------------------------------------------------
# Parser helpers
# ---------------------------------------------------------------------------

def _parse_date(date_str: str) -> tuple[Optional[date], Optional[int], str]:
    """
    Parse a date or duration string from the LLM response.
    Returns (parsed_date, duration_days, timex_type)

    Handles:
    - ISO dates: 2005-09-08, 2005-09, 2005
    - Natural language: "19 July 1986", "1 January 1995", "31 October 1979"
    - ISO durations: P30D, P1M, P5Y, P2Y6M
    - Text durations: "six months", "30 days"

    Non-parseable strings ("immediate", "day of signature") return (None, None, "DATE")
    and are kept as descriptive values without a resolved date.
    """
    if not date_str or date_str.strip().lower() in ("null", "none", ""):
        return None, None, "DATE"

    # P0D is meaningless — treat as no date
    if date_str.strip() == "P0D":
        return None, None, "DATE"

    # ISO duration (P3Y, P30D, P2Y6M, etc.)
    if date_str.startswith("P"):
        days = _iso_duration_to_days(date_str)
        return None, days, "DURATION"

    # Plain duration strings the LLM might return, e.g. "30 days", "six months",
    # "at least three months". Match a digit OR a spelled-out number before a
    # time unit. The leading-digit-or-word requirement avoids false matches on
    # phrases like "day of signature".
    if _DURATION_RE.search(date_str):
        days = _text_duration_to_days(date_str)
        if days is not None:
            return None, days, "DURATION"

    # ISO date formats
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            d = datetime.strptime(date_str.strip(), fmt).date()
            return d, None, "DATE"
        except ValueError:
            continue

    # Natural language date formats: "19 July 1986", "1 January 1995", etc.
    for fmt in ("%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y", "%d/%m/%Y"):
        try:
            d = datetime.strptime(date_str.strip(), fmt).date()
            return d, None, "DATE"
        except ValueError:
            continue

    # ISO interval "YYYY-MM-DD/YYYY-MM-DD" — take the start date
    if "/" in date_str:
        start = date_str.split("/")[0].strip()
        try:
            d = datetime.strptime(start, "%Y-%m-%d").date()
            return d, None, "DATE"
        except ValueError:
            pass

    return None, None, "DATE"


def _iso_duration_to_days(value: str) -> Optional[int]:
    total = 0
    for m in re.finditer(r"(\d+(?:\.\d+)?)([YMWD])", value):
        n = float(m.group(1))
        unit = m.group(2)
        if unit == "Y":
            total += int(n * 365)
        elif unit == "M":
            total += int(n * 30)
        elif unit == "W":
            total += int(n * 7)
        elif unit == "D":
            total += int(n)
    return total if total > 0 else None


_NUM_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}


def _words_to_int(phrase: str) -> Optional[int]:
    """
    Convert a spelled-out number phrase to an int. Handles units, teens, tens,
    and 'hundred' (e.g. 'ninety' → 90, 'one hundred and twenty' → 120,
    'three' → 3). Returns None if no number words are present.
    """
    tokens = re.findall(r"[a-z]+", phrase.lower())
    total = 0
    current = 0
    found = False
    for t in tokens:
        if t in _NUM_WORDS:
            current += _NUM_WORDS[t]
            found = True
        elif t == "hundred":
            current = (current or 1) * 100
            found = True
        elif t == "and":
            continue
        else:
            # non-number token ends the run; keep accumulating across it
            continue
    return (total + current) if found else None


# digit, or a run of number-words, immediately before a time unit
_DURATION_RE = re.compile(
    r"(\d+|(?:(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
    r"nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
    r"hundred|and)[\s-]*)+)\s*(years?|months?|weeks?|days?)",
    re.IGNORECASE,
)


_MONTH_NAME_RE = "January|February|March|April|May|June|July|August|September|October|November|December"

_TEMPORAL_SIGNAL_RE = re.compile(
    r"\d{4}-\d{2}"                                          # ISO date fragment
    r"|\b(?:1[89]\d{2}|20\d{2})\b"                          # bare 4-digit year
    r"|\b\d{1,2}\s+(?:" + _MONTH_NAME_RE + r")\b"           # "31 March"
    r"|\b(?:" + _MONTH_NAME_RE + r")\b"                     # month name
    r"|\bP\d"                                               # ISO duration
    r"|\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
    r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)"
    r"[\s-]*(?:day|days|month|months|week|weeks|year|years)\b"
    # periodicity / recurrence (temporal even without a digit)
    r"|\b(?:annual|annually|periodic|periodically|quarterly|monthly|weekly|"
    r"daily|biannual|semi-annual|each year|every year|per year|each month|"
    r"first quarter|second quarter|third quarter|fourth quarter)\b"
    # relative-time triggers / date anchors (offset lives in a dependency)
    r"|\b(?:after|before|following|within|prior to|expir|upon|as of|as from|"
    r"on the date|date of|with effect from|from the date|no later than|"
    r"entry into force|signature|signing|deposit|receipt|anniversary|"
    r"takes? effect|effective)\b",
    re.IGNORECASE,
)


def _has_temporal_signal(text: str) -> bool:
    """
    True if the text carries temporal content: a date, month name, duration
    (digit or spelled), or a relative-time trigger. Used to drop clause
    fragments the LLM mis-tagged as temporal facts with null values, while
    keeping genuine relative references whose value is legitimately null.
    """
    return bool(text) and bool(_TEMPORAL_SIGNAL_RE.search(text))


def _text_duration_to_days(text: str) -> Optional[int]:
    total = 0
    found = False
    units = {"year": 365, "month": 30, "week": 7, "day": 1}
    for m in _DURATION_RE.finditer(text):
        num_token = m.group(1).strip()
        if num_token.isdigit():
            n = int(num_token)
        else:
            n = _words_to_int(num_token)
            if n is None:
                continue
        unit = m.group(2).lower().rstrip("s")
        total += n * units.get(unit, 0)
        found = True
    return total if found else None


def _strip_json_fences(text: str) -> str:
    """Remove markdown code fences that some local models add despite instructions."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Post-processing validation
# ---------------------------------------------------------------------------

_MONTH_NAMES = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\b",
    re.IGNORECASE,
)


def _fix_duration_month_confusion(
    date_str: str, raw_text: str, sentence: str
) -> str:
    """Catch durations that are actually calendar dates.

    Pattern: Gemma outputs "P31D" for "no later than 31 March" because it
    reads the day-of-month (31) as a day count. If the raw_text or sentence
    contains a month name, this is a calendar date reference, not a duration.

    Returns corrected date_str, or original if no fix needed.
    """
    if not date_str or not date_str.startswith("P"):
        return date_str
    context = f"{raw_text} {sentence}"
    if _MONTH_NAMES.search(context):
        # Try to extract an actual date from the context
        # Pattern: "31 March" or "28 February 1995"
        m = re.search(
            r"(\d{1,2})\s+(January|February|March|April|May|June|July|"
            r"August|September|October|November|December)(?:\s+(\d{4}))?",
            context, re.IGNORECASE,
        )
        if m:
            day, month_name, year = m.group(1), m.group(2), m.group(3)
            if year:
                try:
                    d = datetime.strptime(f"{day} {month_name} {year}", "%d %B %Y")
                    return d.strftime("%Y-%m-%d")
                except ValueError:
                    pass
            # No year → can't resolve to ISO date
            return "null"
    return date_str


def _fix_year_garbling(
    date_str: str, raw_text: str, sentence: str
) -> str:
    """Catch ISO dates where the year was garbled.

    Pattern: Gemma outputs "1979-07-19" for "19 July 1986" — the day (19)
    leaks into the year. If the year in the ISO date doesn't appear anywhere
    in the raw_text or sentence, try to find the correct year from context.

    Returns corrected date_str, or original if no fix needed.
    """
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", date_str.strip() if date_str else "")
    if not m:
        return date_str

    iso_year = m.group(1)
    context = f"{raw_text} {sentence}"

    # If the year from the ISO date appears in the source text, it's probably correct
    if iso_year in context:
        return date_str

    # Year doesn't appear in source — find all 4-digit years in context
    context_years = re.findall(r"\b(1[89]\d{2}|20[0-3]\d)\b", context)
    if len(context_years) == 1:
        # Exactly one year in context — use it
        corrected = f"{context_years[0]}-{m.group(2)}-{m.group(3)}"
        try:
            datetime.strptime(corrected, "%Y-%m-%d")  # validate
            return corrected
        except ValueError:
            pass
    elif len(context_years) > 1:
        # Multiple years — try to match based on day/month in raw_text
        # "19 July 1986" → day=19, month=July → find "1986" in context
        dm = re.search(
            r"(\d{1,2})\s+(?:January|February|March|April|May|June|July|"
            r"August|September|October|November|December)\s+(\d{4})",
            context, re.IGNORECASE,
        )
        if dm and dm.group(2) != iso_year:
            corrected = f"{dm.group(2)}-{m.group(2)}-{m.group(3)}"
            try:
                datetime.strptime(corrected, "%Y-%m-%d")
                return corrected
            except ValueError:
                pass

    return date_str


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class LLMPipeline:
    """
    Event-centric TDG pipeline using an OpenAI-compatible LLM for extraction.

    Works with:
    - OpenAI GPT models (default, requires OPENAI_API_KEY env var or .env file)
    - Local Ollama models via base_url (no API key needed)
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.0,
        base_url: Optional[str] = None,
        max_tokens: int = 8192,
    ):
        """
        Args:
            model:       Model name. For Ollama use e.g. "gemma4:e4b" or "gemma4:26b".
            temperature: Sampling temperature. Keep at 0.0 for deterministic extraction.
            base_url:    Ollama endpoint, e.g. "http://localhost:11434/v1".
                         If None, uses OpenAI API (requires OPENAI_API_KEY).
        """
        import os
        model = model or os.environ.get("TDG_LLM_MODEL") or os.environ.get("OPENAI_MODEL")
        base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        if not model:
            raise ValueError(
                "No LLM model configured. Pass --model on the command line "
                "(e.g. --model gemma3:4b for Ollama, --model gpt-4o-mini for "
                "OpenAI) or set TDG_LLM_MODEL. There is no default model.")
        if base_url:
            # Local Ollama — api_key value is required by the client but ignored
            self.client = OpenAI(api_key="ollama", base_url=base_url)
        else:
            # OpenAI — try loading .env, fall back to env var
            try:
                from dotenv import load_dotenv
                load_dotenv()
            except ImportError:
                pass  # python-dotenv not installed, rely on env var being set directly

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY not found. Either add it to a .env file "
                    "(requires python-dotenv) or set it as an environment variable. "
                    "For local Ollama usage pass base_url instead."
                )
            self.client = OpenAI(api_key=api_key)

        self.model = model
        self.temperature = temperature
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.graph_builder = GraphBuilder()

    def _call_llm(self, text: str, document_type: str) -> dict:
        """Call the LLM and return parsed JSON."""
        self.last_raw = ""
        user_msg = USER_PROMPT_TEMPLATE.format(
            document_type=document_type,
            text=text,
        )

        # OpenAI GPT-5.x rejects legacy `max_tokens` (wants max_completion_tokens);
        # Ollama's /v1 endpoint accepts only the legacy name. Try modern name for
        # direct-OpenAI, fall back on either error so both backends work.
        _tok_param = "max_tokens" if self.base_url else "max_completion_tokens"
        try:
            response = self._create(_tok_param, user_msg)
        except Exception as e:
            _alt = ("max_completion_tokens" if _tok_param == "max_tokens"
                    else "max_tokens")
            if "max_tokens" in str(e) or "max_completion_tokens" in str(e):
                response = self._create(_alt, user_msg)
            else:
                raise
        raw = response.choices[0].message.content
        # Archive the extractor's own words. An empty graph has three causes
        # (call failed / JSON unparseable / model genuinely found no events)
        # and they are NOT the same finding; without the raw text they are
        # indistinguishable after the run. Set before any parsing can throw.
        self.last_raw = raw or ""
        raw = _strip_json_fences(raw)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            # Small models occasionally emit syntax slips (e.g. a missing
            # comma) in long JSON outputs. Deterministic repair fallback:
            # strictly additive — only reached when strict parsing fails —
            # and flagged for provenance. A temp-0 retry would reproduce
            # the identical defect, so repair (not retry) is the fix.
            try:
                from json_repair import repair_json
            except ImportError:
                raise RuntimeError(
                    f"JSON parse failed ({e}) and json-repair not installed; "
                    f"pip install json-repair") from e
            parsed = json.loads(repair_json(raw))
            n_ev = len(parsed.get("events", []))
            print(f"  WARNING: extractor JSON repaired (strict parse: {e}); "
                  f"{n_ev} events recovered — flagged _json_repaired")
            parsed["_json_repaired"] = str(e)
            return parsed

    def _create(self, tok_param: str, user_msg: str):
        return self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            **{tok_param: self.max_tokens},
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )

    def _build_facts(self, events: list[dict]) -> list[TemporalFact]:
        """Convert LLM event dicts into TemporalFact objects."""
        facts = []
        skipped = sum(1 for ev in events if not isinstance(ev, dict))
        if skipped:
            print(f"  ! skipped {skipped} non-dict event entries "
                  f"(malformed model JSON)")
        events = [ev for ev in events if isinstance(ev, dict)]
        for i, ev in enumerate(events):
            date_str = ev.get("date", "")
            raw_text = ev.get("raw_text", date_str or "")
            sentence = ev.get("sentence", "")

            # Post-processing: fix common LLM extraction errors
            date_str = _fix_duration_month_confusion(date_str, raw_text, sentence)
            date_str = _fix_year_garbling(date_str, raw_text, sentence)

            parsed_date, duration_days, timex_type = _parse_date(date_str)

            role = ev.get("role", "UNKNOWN")
            if timex_type == "DURATION":
                role = "DURATION"

            # Recovery: the LLM sometimes labels a fact DURATION but leaves the
            # normalized value empty, putting the period only in raw_text /
            # sentence (e.g. value=null, raw_text="within 15 days of the
            # request", or "at least three months' notice"). Parse the duration
            # out of those so the fact carries a computable value instead of
            # null. Only applied to DURATION-role facts, so dates are untouched.
            if role == "DURATION" and duration_days is None:
                for source in (raw_text, sentence):
                    recovered = _text_duration_to_days(source or "")
                    if recovered is not None:
                        duration_days = recovered
                        timex_type = "DURATION"
                        if not date_str:
                            date_str = f"P{recovered}D"
                        break

            timex = TimexSpan(
                text=raw_text,
                timex_type=timex_type,
                value=date_str,
                start_char=0,
                end_char=len(raw_text),
                date_parsed=parsed_date,
                duration_days=duration_days,
            )

            # Qualify entity with article_ref if provided:
            # "consultations" + "Art.4.3" → "Art.4.3 consultations"
            base_entity = (ev.get("entity") or "UNKNOWN").strip()
            article_ref = (ev.get("article_ref") or "").strip()
            qualified_entity = (
                f"{article_ref} {base_entity}"
                if article_ref and article_ref.lower() != "null"
                else base_entity
            )

            # Precision flag (NOT a drop): mark facts that carry no temporal
            # content — no resolved value AND no temporal signal in the text
            # (e.g. "HAVE AGREED AS FOLLOWS", "The Contracting Parties
            # undertake"). The fact is KEPT in the graph for auditability;
            # downstream consumers may skip temporal_content=False facts to gain
            # precision. Flagging rather than dropping avoids irreversible,
            # regex-gated data loss — genuine relative references ("one month
            # after...") keep temporal_content=True via the relative triggers.
            has_value = bool(date_str) or duration_days is not None or parsed_date is not None
            temporal_content = has_value or _has_temporal_signal(f"{raw_text} {sentence}")

            fact = TemporalFact(
                id=ev.get("id", f"f{i+1}"),
                entity=qualified_entity,
                role=role,
                timex=timex,
                sentence=ev.get("sentence", ""),
                confidence=0.9,
                signal_verb=None,
                signal_prep=None,
                temporal_content=temporal_content,
            )

            facts.append(fact)
        return facts

    def _build_dependencies(
        self,
        relations: list[dict],
        facts: list[TemporalFact],
    ) -> list[TemporalDependency]:
        """Convert LLM relation dicts into TemporalDependency objects."""
        fact_ids = {f.id for f in facts}
        fact_sentences = {f.id: f.sentence for f in facts}

        # Offset trigger words that justify an additive relation
        _OFFSET_TRIGGERS = re.compile(
            r"\b(within|after|before|from|no later than|not later than|"
            r"following|upon|prior to|at least|at most|thereof)\b",
            re.IGNORECASE,
        )

        # edge_map: (from_id, to_id) -> TemporalDependency
        # Prevents exact duplicate edges from the LLM.
        edge_map: dict[tuple[str, str], TemporalDependency] = {}
        deps = []
        relations = [r for r in relations if isinstance(r, dict)]
        for rel in relations:
            from_id = rel.get("from_id")
            to_id = rel.get("to_id")
            if from_id not in fact_ids or to_id not in fact_ids:
                continue
            if from_id == to_id:
                continue
            pair = (from_id, to_id)

            constraint_type = rel.get("type", "ordering")
            if constraint_type not in ("additive", "ordering", "interval", "periodic"):
                constraint_type = "ordering"

            # Fix 2a: downgrade additive to ordering if neither source sentence
            # contains an explicit offset trigger word — likely a spurious relation
            if constraint_type == "additive":
                sent_from = fact_sentences.get(from_id, "")
                sent_to = fact_sentences.get(to_id, "")
                if not _OFFSET_TRIGGERS.search(sent_from) and \
                   not _OFFSET_TRIGGERS.search(sent_to):
                    constraint_type = "ordering"

            # Fix 2b: downgrade additive to ordering if the downstream fact has
            # a fixed calendar date — fixed dates don't ripple from anchor changes
            if constraint_type == "additive":
                to_fact = next((f for f in facts if f.id == to_id), None)
                if to_fact and to_fact.timex.date_parsed is not None:
                    constraint_type = "ordering"

            # Accept delta_days from the LLM when provided
            delta_days = rel.get("delta_days")
            if isinstance(delta_days, float):
                delta_days = int(delta_days)

            if pair in edge_map:
                # Same edge pair from the LLM — skip the duplicate
                edge_map[pair].corroboration_count += 1
            else:
                dep = TemporalDependency(
                    from_id=from_id,
                    to_id=to_id,
                    constraint_type=constraint_type,
                    constraint_expr=rel.get("constraint", f"{from_id} → {to_id}"),
                    delta_days=delta_days,
                    confidence=0.85,
                    verified=False,
                    corroboration_count=1,
                )
                edge_map[pair] = dep
                deps.append(dep)
        return deps

    def process(
        self,
        text: str,
        document_id: str = "doc_001",
        document_type: str = "legal",
        generate_scenarios: bool = True,
    ) -> TemporalDependencyGraph:
        """Full pipeline: text → TemporalDependencyGraph."""
        print(f"  Calling {self.model}...")
        # last_status distinguishes the three ways this method can return an
        # empty graph. The harness reads it instead of guessing with a ping.
        # Behaviour is unchanged: this only records WHY.
        self.last_status = "ok"
        self.last_raw = ""
        try:
            llm_output = self._call_llm(text, document_type)
        except json.JSONDecodeError as e:
            print(f"  JSON parse failed: {e}")
            print(f"  Raw output (first 500 chars): {getattr(e, 'doc', '')[:500]!r}")
            self.last_status = f"extract-json-error: {e}"
            return TemporalDependencyGraph(
                document_id=document_id,
                document_type=document_type,
                source_text=text,
            )
        except Exception as e:
            print(f"  LLM call failed: {e}")
            self.last_status = f"extract-call-error: {e}"
            return TemporalDependencyGraph(
                document_id=document_id,
                document_type=document_type,
                source_text=text,
            )

        events = llm_output.get("events", [])
        relations = llm_output.get("relations", [])

        if not events:
            print("  No events extracted.")
            self.last_status = "extract-zero-events"
            return TemporalDependencyGraph(
                document_id=document_id,
                document_type=document_type,
                source_text=text,
            )

        print(f"  Extracted {len(events)} events, {len(relations)} relations.")

        facts = self._build_facts(events)
        llm_deps = self._build_dependencies(relations, facts)

        tdg = self.graph_builder.build(
            facts=facts,
            document_id=document_id,
            document_type=document_type,
            source_text=text,
        )

        # Named parties, used downstream to tell one matter from another.
        # Kept exactly as extracted: normalising names is the linker's job,
        # and rewriting them here would lose what the document actually said.
        raw_parties = llm_output.get("parties") or []
        if isinstance(raw_parties, str):
            raw_parties = [raw_parties]
        tdg.parties = [str(p).strip() for p in raw_parties if str(p).strip()]

        # Merge LLM-detected relations with graph builder output
        existing = {(d.from_id, d.to_id) for d in tdg.dependencies}
        for dep in llm_deps:
            if (dep.from_id, dep.to_id) not in existing:
                tdg.dependencies.append(dep)

        if generate_scenarios:
            tdg.edit_scenarios = generate_edit_scenarios(tdg)

        return tdg
    