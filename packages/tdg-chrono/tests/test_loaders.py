"""Intake defaults to raw text; the aggressive cleaner is opt-in.

Routing every document through clean_to_paragraphs was the wrong default.
The cleaner is tuned for messy two-column PDF corpora; on ordinary
correspondence it deleted body text, and it is not the configuration the
pipeline was evaluated on. These tests pin the default to a
content-preserving path.
"""

from __future__ import annotations

import re

from tdg_chrono.loaders import (
    normalize_whitespace,
    process_text,
    process_with_report,
)

LETTER = """PRIVATE AND CONFIDENTIAL

Northgate Logistics Ltd

3 June 2025

Dear Ms Okafor,

Following the hearing, the company has decided to terminate your contract of
employment. Your employment terminates with
effect from 12 July 2025.

Yours sincerely,
"""

DATE_RE = re.compile(
    r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{4}\b")


def test_default_intake_keeps_every_date():
    out = process_text(LETTER)
    in_source = {m.group(0) for m in DATE_RE.finditer(LETTER)}
    assert in_source == {m.group(0) for m in DATE_RE.finditer(out)}


def test_default_intake_keeps_boilerplate_too():
    """The default deletes nothing at all — not even the letterhead."""
    out = process_text(LETTER)
    assert "PRIVATE AND CONFIDENTIAL" in out
    assert "Yours sincerely" in out


def test_default_intake_reports_nothing_dropped():
    _, dropped = process_with_report(LETTER)
    assert dropped == []


def test_clean_is_opt_in_and_reports_what_it_removed():
    out, dropped = process_with_report(LETTER, clean=True)
    assert "PRIVATE AND CONFIDENTIAL" not in out
    assert dropped, "cleaner must declare what it discarded"
    # even opted in, it may not destroy a date
    assert "12 July 2025" in out
    assert not [d for d in dropped if DATE_RE.search(d)]


def test_normalize_rejoins_hyphenated_line_break():
    assert "warehouse" in normalize_whitespace("ware-\nhouse supervisor")


def test_normalize_collapses_spaces_but_keeps_paragraphs():
    out = normalize_whitespace("one    two\n\n\n\nthree")
    assert "one two" in out
    assert "\n\n" in out          # paragraph break preserved
    assert "\n\n\n" not in out    # runs capped


def test_normalize_preserves_all_words():
    text = "The claim was presented on 1 October 2025.\nIt was served later."
    out = normalize_whitespace(text)
    for word in text.split():
        assert word.strip() in out
