"""The cleaner must never destroy a date.

Regression cover for a bug where lines shorter than 40 characters were
classified as headers/footers and deleted. On hard-wrapped text — which is
what PDF extraction produces — that ate the last line of most paragraphs.
In a two-paragraph dismissal letter it silently removed 35% of the document
including "effect from 12 July 2025.", the date that fixes the limitation
clock. No downstream extractor could recover it.
"""

from __future__ import annotations

import re

from tdg_core.text_cleaner import (
    clean_to_paragraphs,
    clean_with_report,
    has_temporal_content,
)

WRAPPED_LETTER = """PRIVATE AND CONFIDENTIAL

Northgate Logistics Ltd
14 Camden Row, Manchester M1 4TB

3 June 2025

Dear Ms Okafor,

TERMINATION OF EMPLOYMENT

I write to confirm the outcome of the disciplinary hearing held on 28 May 2025.

You commenced employment with the company on 3 March 2019 as a Warehouse
Supervisor.

Following the hearing, the company has decided to terminate your contract of
employment on grounds of gross misconduct. Your employment terminates with
effect from 12 July 2025.

Yours sincerely,

R. Whitaker
"""

DATE_RE = re.compile(
    r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{4}\b")


def test_no_date_is_lost_from_hard_wrapped_text():
    cleaned = "\n\n".join(clean_to_paragraphs(WRAPPED_LETTER))
    in_source = {m.group(0) for m in DATE_RE.finditer(WRAPPED_LETTER)}
    survived = {m.group(0) for m in DATE_RE.finditer(cleaned)}
    assert in_source, "fixture should contain dates"
    assert in_source == survived, f"cleaner destroyed {sorted(in_source - survived)}"


def test_wrapped_tail_line_survives():
    """The exact span the original bug deleted."""
    cleaned = "\n\n".join(clean_to_paragraphs(WRAPPED_LETTER))
    assert "12 July 2025" in cleaned
    # and it is joined to its own sentence, not orphaned
    assert "effect from 12 July 2025" in cleaned


def test_short_line_carrying_a_date_is_kept():
    """A date on its own line is 11 characters — far under any threshold."""
    cleaned = "\n\n".join(clean_to_paragraphs("3 June 2025"))
    assert "3 June 2025" in cleaned


def test_boilerplate_without_dates_is_still_removed():
    paragraphs = clean_to_paragraphs(WRAPPED_LETTER)
    joined = " ".join(paragraphs)
    assert "PRIVATE AND CONFIDENTIAL" not in joined
    assert "Yours sincerely" not in joined


def test_removals_are_reported_not_silent():
    _, dropped = clean_with_report(WRAPPED_LETTER)
    assert "PRIVATE AND CONFIDENTIAL" in dropped
    # nothing carrying a date may appear in the dropped list
    assert not [d for d in dropped if DATE_RE.search(d)]


def test_hyphenated_word_split_across_lines_is_rejoined():
    cleaned = "\n\n".join(clean_to_paragraphs(
        "The claimant was employed as a ware-\nhouse supervisor until 12 July 2025."))
    assert "warehouse" in cleaned
    assert "12 July 2025" in cleaned


def test_temporal_veto_recognises_uk_date_order():
    assert has_temporal_content("12 July 2025")
    assert has_temporal_content("July 12, 2025")
    assert has_temporal_content("12/07/2025")
    assert has_temporal_content("2025-07-12")
    assert has_temporal_content("within 28 days")
    assert not has_temporal_content("Head of Human Resources")


def test_two_column_artifact_still_dropped():
    _, dropped = clean_with_report("Left column text here      right column text")
    assert dropped


def test_mentions_date_accepts_the_forms_documents_use():
    from datetime import date

    from tdg_core.text_cleaner import mentions_date

    d = date(2025, 7, 12)
    for text in ("terminates on 12 July 2025",
                 "July 12, 2025",
                 "effective 2025-07-12",
                 "dated 12/07/2025",
                 "dated 12.07.25"):
        assert mentions_date(text, d), text


def test_mentions_date_rejects_a_different_day_in_the_same_month():
    from datetime import date

    from tdg_core.text_cleaner import mentions_date

    assert not mentions_date("the hearing on 28 July 2025", date(2025, 7, 12))
    assert not mentions_date("no dates at all", date(2025, 7, 12))
