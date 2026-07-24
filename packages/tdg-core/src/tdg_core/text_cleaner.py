"""
Legal text cleaner — extracts clean paragraphs from noisy corpus documents.

Handles:
  - Two-column PDF artifacts (interleaved whitespace)
  - Header/footer metadata (OJ references, case numbers, page numbers)
  - Mid-sentence line wraps from PDF extraction
  - Short fragment lines with no temporal content
  - Citation-only lines (OJ C 229, 2.9.1995)

Returns a list of clean paragraphs suitable for TDG pipeline input.
"""

from __future__ import annotations

import re


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

# Lines that are too short to contain useful temporal info
_MIN_LINE_LENGTH = 40

# Paragraphs shorter than this are skipped
_MIN_PARAGRAPH_LENGTH = 80

# Paragraphs longer than this are split at sentence boundaries
_MAX_PARAGRAPH_LENGTH = 1500


def _is_junk_line(line: str) -> bool:
    """Return True if this line is metadata/header/footer, not body text."""
    stripped = line.strip()
    if len(stripped) < _MIN_LINE_LENGTH:
        return True
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


def clean_to_paragraphs(text: str) -> list[str]:
    """
    Clean a raw legal document text and return a list of clean paragraphs.

    Each paragraph is a coherent block of body text, suitable for passing
    directly to the TDG pipeline.
    """
    # Replace em-dash bullet points (EU court doc artifact: "— declare the Commission...")
    # These are actually separate clauses — split them into sentences
    text = re.sub(r'\s*[—–\-]{1,2}\s+', '. ', text)

    # Split into lines
    lines = text.split("\n")

    # First pass: filter junk lines, normalize remaining
    clean_lines = []
    for line in lines:
        if _is_junk_line(line):
            continue
        normalized = _normalize_line(line)
        if normalized:
            clean_lines.append(normalized)

    if not clean_lines:
        return []

    # Second pass: join lines into paragraphs
    paragraphs = []
    current = []

    for line in clean_lines:
        is_new_clause = bool(re.match(r"^\d+[\.\)]\s+[A-Z]", line) or
                             re.match(r"^[a-z]\)\s+[A-Z]", line))
        prev_ends_sentence = current and re.search(r"[.!?]\s*$", current[-1])

        if is_new_clause and current:
            paragraphs.append(" ".join(current))
            current = [line]
        elif prev_ends_sentence and is_new_clause:
            paragraphs.append(" ".join(current))
            current = [line]
        else:
            current.append(line)

    if current:
        paragraphs.append(" ".join(current))

    # Third pass: filter short paragraphs, split overly long ones
    result = []
    for para in paragraphs:
        para = para.strip()
        if len(para) < _MIN_PARAGRAPH_LENGTH:
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

    return result


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

    date_pattern = re.compile(
        r"\b\d{1,2}[\./]\d{1,2}[\./]\d{2,4}\b"          # DD.MM.YYYY
        r"|\b(?:January|February|March|April|May|June|"
        r"July|August|September|October|November|December)"
        r"\s+\d{1,2},?\s+\d{4}\b"                        # Month DD, YYYY
        r"|\b\d{4}\b"                                     # bare year
        r"|\b\d+\s+(?:days?|months?|years?|weeks?)\b",   # durations
        re.IGNORECASE
    )
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
