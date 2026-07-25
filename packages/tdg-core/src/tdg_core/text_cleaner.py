"""
Legal text cleaner — extracts clean paragraphs from noisy corpus documents.

Handles:
  - Two-column PDF artifacts (interleaved whitespace)
  - Header/footer metadata (OJ references, case numbers, page numbers)
  - Mid-sentence line wraps from PDF extraction
  - Short fragment lines with no temporal content
  - Citation-only lines (OJ C 229, 2.9.1995)

Returns a list of clean paragraphs suitable for TDG pipeline input.

Order matters: hard-wrapped lines are joined into logical paragraphs
*before* anything is classified as junk. Classifying first meant the tail
line of every wrapped paragraph ("effect from 12 July 2025.", 25 chars)
was shorter than the header/footer threshold and silently deleted —
destroying evidence no downstream extractor could then recover. Length is
only ever judged on a whole paragraph, and never overrides the temporal
veto: a span carrying a date, duration or year is kept regardless of how
short it is. This follows the rule stated for TemporalFact.temporal_content —
flag, never drop, because regex-gated deletion is irreversible.
"""

from __future__ import annotations

import re

_MONTHS = (r"January|February|March|April|May|June|July|August|September|"
           r"October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|"
           r"Oct|Nov|Dec")

# Anything that might carry temporal meaning. Deliberately broad: this is a
# veto on deletion, so a false positive costs one retained line while a false
# negative destroys a date. Shared with best_paragraph's scoring.
_TEMPORAL_RE = re.compile(
    r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b"              # 12/07/2025, 12.07.25
    r"|\b\d{4}-\d{2}-\d{2}\b"                            # ISO 2025-07-12
    rf"|\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:{_MONTHS})\b"   # 12 July
    rf"|\b(?:{_MONTHS})\s+\d{{1,2}}\b"                   # July 12
    rf"|\b(?:{_MONTHS})\s+\d{{4}}\b"                     # July 2025
    r"|\b(?:19|20)\d{2}\b"                               # bare year
    r"|\b\d+\s+(?:day|week|month|year|hour)s?\b"         # durations
    r"|\b(?:yesterday|today|tomorrow|forthwith|immediately)\b",
    re.IGNORECASE,
)


def has_temporal_content(text: str) -> bool:
    """True if the span carries anything date-like. Used as a deletion veto."""
    return bool(_TEMPORAL_RE.search(text))


_MONTH_NAMES = [m.lower() for m in _MONTHS.split("|")]


def mentions_date(text: str, when) -> bool:
    """Does this text actually state the given date?

    Used to tell a quote that evidences a fact from one that merely travels
    with it. Accepts the written forms a document plausibly uses — "12 July
    2025", "July 12, 2025", "2025-07-12", "12/07/2025" — and requires day,
    month and year to agree, so a nearby sentence naming a different date in
    the same month does not count.
    """
    if not text or when is None:
        return False
    low = text.lower()
    if when.isoformat() in low:
        return True

    year = str(when.year)
    day, month = str(when.day), str(when.month)

    # Written form: the day and the month by name, with the year present.
    if year in low:
        month_name = _MONTH_NAMES[when.month - 1][:3]
        if re.search(rf"\b0?{day}\b", low) and month_name in low:
            return True

    # Numeric forms, both orders, either separator, two- or four-digit year.
    numeric = (rf"\b(?:0?{day}[./-]0?{month}|0?{month}[./-]0?{day})"
               rf"[./-](?:{year}|{year[2:]})\b")
    return bool(re.search(numeric, low))


# Lines that are almost certainly headers, footers, or citations — not body text
_JUNK_PATTERNS = [
    re.compile(r"^\s*OJ\s+[A-Z]\s+\d+", re.IGNORECASE),        # OJ C 229, OJ L 45
    re.compile(r"^\s*\(\d+\)\s*$"),                              # lone footnote markers: (1)
    re.compile(r"^\s*\d+\s*/\s*\d+\s*$"),                       # page refs: C 71/2
    re.compile(r"^\s*[IVX]+\s*$"),                               # roman numerals alone
    re.compile(r"^\s*[-–—]+\s*$"),                               # separator lines
    re.compile(r"^\s*\*+\s*$"),                                  # asterisk lines
    re.compile(r"official journal", re.IGNORECASE),              # header
    re.compile(r"european communities", re.IGNORECASE),          # header
    re.compile(r"^\s*\(language of the case", re.IGNORECASE),    # metadata
    re.compile(r"^\s*judgment of the court", re.IGNORECASE),     # title
    re.compile(r"^\s*in case\s+[A-Z]-\d+", re.IGNORECASE),      # case number lines
]

# Paragraphs shorter than this are skipped — applied to a joined paragraph,
# never to a single wrapped line, and never against the temporal veto.
_MIN_PARAGRAPH_LENGTH = 80

# Paragraphs longer than this are split at sentence boundaries
_MAX_PARAGRAPH_LENGTH = 1500


def _is_junk_line(line: str) -> bool:
    """True if this line is layout noise (header/footer/citation/column split).

    Length is deliberately not a criterion here. A short line is far more
    often the tail of a hard wrap than a header, and deleting it loses body
    text. Shortness is judged later, on the assembled paragraph.
    """
    stripped = line.strip()
    if has_temporal_content(stripped):
        return False
    for pat in _JUNK_PATTERNS:
        if pat.search(stripped):
            return True
    # Lines with excessive whitespace in the middle = two-column artifact
    if re.search(r"\s{6,}", stripped):
        return True
    return False


def _normalize_line(line: str) -> str:
    """Collapse internal whitespace, strip edges."""
    return re.sub(r"\s+", " ", line).strip()


def _split_at_sentences(text: str) -> list[str]:
    """Split a long paragraph at sentence boundaries."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [p.strip() for p in parts if p.strip()]


def clean_with_report(text: str) -> tuple[list[str], list[str]]:
    """Clean the text and also return every span that was discarded.

    The report exists so deletion is auditable rather than silent — a tool
    whose output is meant to be checked against source cannot quietly drop
    part of that source.
    """
    dropped: list[str] = []

    # Rejoin words split across a line break ("Ware-\nhouse" -> "Warehouse").
    # Must precede the dash rule below, which would otherwise turn the wrap
    # hyphen into a sentence break and cut the word in half.
    text = re.sub(r"(\w)[-‐]\s*\n\s*(\w)", r"\1\2", text)

    # Replace em-dash bullet points (EU court doc artifact: "— declare the Commission...")
    # These are actually separate clauses — split them into sentences
    text = re.sub(r'\s*[—–\-]{1,2}\s+', '. ', text)

    # Blank lines delimit paragraphs; within a block, a newline is a hard wrap.
    # Joining before classifying is what keeps wrapped tails alive.
    paragraphs: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        kept: list[str] = []
        for line in block.split("\n"):
            if not line.strip():
                continue
            if _is_junk_line(line):
                dropped.append(_normalize_line(line))
                continue
            normalized = _normalize_line(line)
            if normalized:
                kept.append(normalized)
        if not kept:
            continue

        current: list[str] = []
        for line in kept:
            is_new_clause = bool(re.match(r"^\d+[\.\)]\s+[A-Z]", line) or
                                 re.match(r"^[a-z]\)\s+[A-Z]", line))
            if is_new_clause and current:
                paragraphs.append(" ".join(current))
                current = [line]
            else:
                current.append(line)
        if current:
            paragraphs.append(" ".join(current))

    # Final pass: drop short paragraphs (unless temporal), split overly long ones
    result = []
    for para in paragraphs:
        para = para.strip()
        if len(para) < _MIN_PARAGRAPH_LENGTH and not has_temporal_content(para):
            dropped.append(para)
            continue
        if len(para) > _MAX_PARAGRAPH_LENGTH:
            sentences = _split_at_sentences(para)
            chunk = []
            chunk_len = 0
            for sent in sentences:
                if chunk_len + len(sent) > _MAX_PARAGRAPH_LENGTH and chunk:
                    result.append(" ".join(chunk))
                    chunk = [sent]
                    chunk_len = len(sent)
                else:
                    chunk.append(sent)
                    chunk_len += len(sent)
            if chunk:
                result.append(" ".join(chunk))
        else:
            result.append(para)

    return result, dropped


def clean_to_paragraphs(text: str) -> list[str]:
    """
    Clean a raw legal document text and return a list of clean paragraphs.

    Each paragraph is a coherent block of body text, suitable for passing
    directly to the TDG pipeline. Use clean_with_report when you also need
    to see what was discarded.
    """
    paragraphs, _ = clean_with_report(text)
    return paragraphs


def best_paragraph(text: str) -> str:
    """
    Return the single best paragraph for TDG processing —
    the one most likely to contain explicit temporal facts.

    Scores paragraphs by: number of date-like patterns + temporal keywords.
    """
    paragraphs = clean_to_paragraphs(text)
    if not paragraphs:
        # Last resort: just normalize the whole text
        return re.sub(r"\s+", " ", text).strip()[:2000]

    # Shared with the deletion veto, so scoring and retention agree on what
    # "temporal" means. The previous local pattern matched "July 12, 2025" but
    # not the "12 July 2025" form used throughout UK practice.
    date_pattern = _TEMPORAL_RE
    temporal_keywords = re.compile(
        r"\b(?:filed|concluded|signed|terminated|commenced|began|ended|"
        r"entered|effective|issued|decided|ordered|dismissed|granted|"
        r"submitted|notified|registered|appointed|adopted)\b",
        re.IGNORECASE
    )

    def score(para: str) -> int:
        return (len(date_pattern.findall(para)) * 2 +
                len(temporal_keywords.findall(para)))

    return max(paragraphs, key=score)
