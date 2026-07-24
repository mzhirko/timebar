"""
Temporal expression extraction.

Backends:
  1. HeidelTimeExtractor  — py-heideltime wrapper (default, recommended)
  2. RegexExtractor       — pure Python fallback (no Java needed)

HeidelTime handles:
  - Explicit dates: "January 15, 2025"
  - Relative expressions: "within 30 days", "two weeks after"
  - Underspecified dates: "in 1891", "last March"
  - Vague durations: "several months"

Install:
    pip install py-heideltime
    # Requires Java (OpenJDK 11+)
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from datetime import date
from typing import Optional

from dateutil import parser as dateutil_parser

from tdg_core.tdg import TimexSpan


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class TimexExtractor(ABC):
    """Interface for temporal expression extractors."""

    @abstractmethod
    def extract(self, text: str, reference_date: Optional[date] = None) -> list[TimexSpan]:
        """Extract temporal expressions from text."""
        ...


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def try_parse_date(text: str) -> Optional[date]:
    """Attempt to parse a date string into a Python date object."""
    try:
        return dateutil_parser.parse(text, dayfirst=False, fuzzy=True).date()
    except (ValueError, OverflowError, TypeError):
        return None


_UNIT_TO_DAYS = {
    "day": 1, "days": 1,
    "week": 7, "weeks": 7,
    "month": 30, "months": 30,
    "year": 365, "years": 365,
    "business day": 1, "business days": 1,
    "calendar day": 1, "calendar days": 1,
    "working day": 1, "working days": 1,
}


def parse_duration(text: str) -> Optional[int]:
    """Parse a duration string like '5 years 8 months' into approximate days."""
    total = 0
    found = False
    for m in re.finditer(r"(\d+)\s*(years?|months?|weeks?|days?)", text, re.IGNORECASE):
        n = int(m.group(1))
        unit = m.group(2).lower()
        factor = _UNIT_TO_DAYS.get(unit, 0)
        total += n * factor
        found = True
    return total if found else None


def _iso_value_to_duration_days(value: str) -> Optional[int]:
    """
    Convert ISO 8601 duration string (e.g. P1Y6M) to approximate days.
    HeidelTime outputs durations in this format.
    """
    if not value.startswith("P"):
        return None
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
    # Handle time component (T...)
    for m in re.finditer(r"(\d+(?:\.\d+)?)([HMS])", value):
        pass  # ignore sub-day durations for now
    return total if total > 0 else None


# ---------------------------------------------------------------------------
# HeidelTime extractor (recommended)
# ---------------------------------------------------------------------------

class HeidelTimeExtractor(TimexExtractor):
    """
    Wraps py-heideltime for robust temporal expression extraction.

    Handles relative, underspecified, and vague temporal expressions
    that regex cannot reliably capture.

    Usage:
        extractor = HeidelTimeExtractor(document_type="news")
        spans = extractor.extract(text, reference_date=date(2024, 3, 1))

    document_type options: "news", "narrative", "scientific", "colloquial"
    """

    def __init__(
        self,
        document_type: str = "narrative",
        language: str = "english",
        heideltime_path: Optional[str] = None,
    ):
        """
        Args:
            document_type: HeidelTime document type. Use "narrative" for
                historical/biographical, "news" for corporate/legal, 
                "scientific" for medical.
            language: Language of input text.
            heideltime_path: Optional path to HeidelTime installation.
                py-heideltime auto-detects if None.
        """
        self.document_type = document_type
        self.language = language
        self.heideltime_path = heideltime_path
        self._fallback = RegexExtractor()

    def extract(self, text: str, reference_date: Optional[date] = None) -> list[TimexSpan]:
        try:
            from py_heideltime import heideltime

            dct = reference_date.isoformat() if reference_date else date.today().isoformat()

            # py-heideltime 1.0.x uses 'dct' for document creation time
            # Try both argument names for compatibility across versions
            try:
                kwargs = dict(
                    text=text,
                    language=self.language,
                    document_type=self.document_type,
                    dct=dct,
                )
                if self.heideltime_path:
                    kwargs["heideltime_path"] = self.heideltime_path
                result = heideltime(**kwargs)
            except TypeError:
                # Fallback for versions using 'document_creation_time'
                kwargs = dict(
                    text=text,
                    language=self.language,
                    document_type=self.document_type,
                    document_creation_time=dct,
                )
                if self.heideltime_path:
                    kwargs["heideltime_path"] = self.heideltime_path
                result = heideltime(**kwargs)
            return self._parse_result(result, text)

        except Exception as e:
            print(f"[HeidelTimeExtractor] Failed: {e}. Falling back to regex.")
            return self._fallback.extract(text, reference_date)

    def _parse_result(self, result, original_text: str) -> list[TimexSpan]:
        """
        Parse py-heideltime output into TimexSpan objects.

        py-heideltime returns a list of dicts:
            [{"text": "January 15, 2025", "type": "DATE", "value": "2025-01-15", ...}]
        or a TimeML XML string depending on version — handle both.
        """
        spans = []
        seen: set[tuple[int, int]] = set()

        # Handle list-of-dicts output
        if isinstance(result, list):
            for item in result:
                raw = item.get("text", "")
                ttype = item.get("type", "DATE")
                value = item.get("value", "")
                spans.append(self._make_span(raw, ttype, value, original_text, seen))

        # Handle TimeML XML string output
        elif isinstance(result, str) and "<TimeML" in result:
            import xml.etree.ElementTree as ET
            try:
                root = ET.fromstring(result)
                for timex in root.iter("TIMEX3"):
                    raw = timex.text or ""
                    ttype = timex.get("type", "DATE")
                    value = timex.get("value", "")
                    spans.append(self._make_span(raw, ttype, value, original_text, seen))
            except ET.ParseError as e:
                print(f"[HeidelTimeExtractor] XML parse error: {e}")

        return [s for s in spans if s is not None]

    def _make_span(
        self,
        raw: str,
        ttype: str,
        value: str,
        original_text: str,
        seen: set[tuple[int, int]],
    ) -> Optional[TimexSpan]:
        if not raw:
            return None

        # Find position in original text
        idx = original_text.find(raw)
        start = idx if idx >= 0 else 0
        end = start + len(raw)

        key = (start, end)
        if key in seen:
            return None
        seen.add(key)

        parsed_date = None
        duration_days = None

        if ttype == "DATE":
            parsed_date = try_parse_date(value) if value else try_parse_date(raw)
            if not parsed_date and re.fullmatch(r"\d{4}", value.strip()):
                try:
                    parsed_date = date(int(value.strip()), 1, 1)
                except ValueError:
                    pass

        elif ttype == "DURATION":
            # value is ISO 8601 duration (P1Y6M, P30D, etc.)
            duration_days = _iso_value_to_duration_days(value)
            if duration_days is None:
                duration_days = parse_duration(raw)

        elif ttype == "TIME":
            parsed_date = try_parse_date(value) if value else None

        return TimexSpan(
            text=raw.strip(),
            timex_type=ttype,
            value=value or raw.strip(),
            start_char=start,
            end_char=end,
            date_parsed=parsed_date,
            duration_days=duration_days,
        )


# ---------------------------------------------------------------------------
# Regex fallback extractor
# ---------------------------------------------------------------------------

_DATE_PATTERNS = [
    (r"\b(\d{4}-\d{2}-\d{2})\b", "DATE"),
    (r"\b((?:January|February|March|April|May|June|July|August|September|"
     r"October|November|December)\s+\d{1,2},?\s+\d{4})\b", "DATE"),
    (r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
     r"\.?\s+\d{1,2},?\s+\d{4})\b", "DATE"),
    (r"\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
     r"September|October|November|December)\s+\d{4})\b", "DATE"),
    (r"\((\d{4})\)", "DATE"),
]

_DURATION_PATTERNS = [
    (r"(\d+\s+years?(?:\s*,\s*|\s+and\s+)\d+\s+months?(?:(?:\s*,\s*|\s+and\s+)\d+\s+days?)?)", "DURATION"),
    (r"(\d+\s+months?(?:\s*,\s*|\s+and\s+)\d+\s+days?)", "DURATION"),
]

_SIMPLE_DURATION_PATTERN = re.compile(
    r"(\d+)\s+((?:business\s+|calendar\s+|working\s+)?(?:days?|months?|years?|weeks?))",
    re.IGNORECASE,
)

_RANGE_PATTERN = re.compile(
    r"\b(?:from|between)\s+(\d{4})\s+(?:to|and|until|through|[-–—])\s+(\d{4})\b",
    re.IGNORECASE,
)


class RegexExtractor(TimexExtractor):
    """
    Regex + dateutil fallback. Used when HeidelTime is unavailable.
    Handles explicit dates and durations only.
    """

    def extract(self, text: str, reference_date: Optional[date] = None) -> list[TimexSpan]:
        spans: list[TimexSpan] = []
        seen: set[tuple[int, int]] = set()

        def _add(raw: str, start: int, end: int, ttype: str):
            key = (start, end)
            if key in seen:
                return
            seen.add(key)

            parsed = try_parse_date(raw) if ttype == "DATE" else None
            dur_days = parse_duration(raw) if ttype == "DURATION" else None
            value = None
            if parsed:
                value = parsed.isoformat()
            elif ttype == "DURATION":
                value = raw.strip()
            elif ttype == "DATE" and re.fullmatch(r"\d{4}", raw.strip()):
                value = raw.strip()
                try:
                    parsed = date(int(raw.strip()), 1, 1)
                except ValueError:
                    pass

            spans.append(TimexSpan(
                text=raw.strip(),
                timex_type=ttype,
                value=value,
                start_char=start,
                end_char=end,
                date_parsed=parsed,
                duration_days=dur_days,
            ))

        for m in _RANGE_PATTERN.finditer(text):
            _add(m.group(1), m.start(1), m.end(1), "DATE")
            _add(m.group(2), m.start(2), m.end(2), "DATE")

        for pat, ttype in _DURATION_PATTERNS:
            for m in re.finditer(pat, text, re.IGNORECASE):
                _add(m.group(0), m.start(), m.end(), ttype)

        for m in _SIMPLE_DURATION_PATTERN.finditer(text):
            overlaps = any(
                s <= m.start() < e or s < m.end() <= e
                for s, e in seen
            )
            if not overlaps:
                _add(m.group(0), m.start(), m.end(), "DURATION")

        for pat, ttype in _DATE_PATTERNS:
            for m in re.finditer(pat, text, re.IGNORECASE):
                raw = m.group(1) if m.lastindex else m.group(0)
                _add(raw, m.start(1) if m.lastindex else m.start(),
                     m.end(1) if m.lastindex else m.end(), ttype)

        spans.sort(key=lambda s: s.start_char)
        return spans
