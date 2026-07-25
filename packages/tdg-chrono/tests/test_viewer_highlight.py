"""The document reader marks quoted passages in the original text.

It writes raw HTML into the page, so two things matter beyond looking
right: the document must come back unchanged with the markup stripped, and
anything in the source that looks like markup must not survive as markup.
"""

from __future__ import annotations

import html
import pathlib
import re

SRC = pathlib.Path(__file__).parents[1] / "src" / "tdg_chrono" / "app.py"


def _load_highlight():
    """Import the helper without executing the Streamlit app at module load."""
    text = SRC.read_text()
    start = text.index("def _highlight(")
    end = text.index("def _documents_tab(")
    ns: dict = {}
    exec(compile(text[start:end], "app_highlight", "exec"), ns)
    return ns["_highlight"]


highlight = _load_highlight()

LETTER = (
    "Dear Ms Okafor,\n\n"
    "You commenced employment with the company on 3 March 2019 as a Warehouse\n"
    "Supervisor.\n\n"
    "Your employment terminates with effect from 12 July 2025.\n"
)


def _marked(out: str) -> list[str]:
    return [" ".join(m.split())
            for m in re.findall(r"<mark[^>]*>(.*?)</mark>", out, re.S)]


def test_the_document_survives_unchanged():
    out = highlight(LETTER, [("Your employment terminates with effect from "
                              "12 July 2025.", "#c62828", "t")])
    assert html.unescape(re.sub(r"<[^>]+>", "", out)) == LETTER


def test_a_quote_broken_across_lines_is_still_found():
    """Hard-wrapped source is the normal case, not the exception."""
    out = highlight(LETTER, [("You commenced employment with the company on "
                              "3 March 2019 as a Warehouse Supervisor.",
                              "#2f7d32", "t")])
    assert len(_marked(out)) == 1
    assert _marked(out)[0].endswith("Warehouse Supervisor.")


def test_a_highlight_stops_at_the_end_of_its_quote():
    """An over-running span would attribute the wrong words to a date."""
    out = highlight(LETTER, [("Your employment terminates with effect from "
                              "12 July 2025.", "#c62828", "t")])
    assert _marked(out) == ["Your employment terminates with effect from 12 July 2025."]


def test_markup_in_the_source_is_escaped():
    evil = 'The date <script>alert("x")</script> was 12 July 2025 as stated here.'
    out = highlight(evil, [("was 12 July 2025 as stated here.", "#2f7d32", "t")])
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


def test_a_quote_that_is_not_in_the_text_is_skipped():
    out = highlight(LETTER, [("A disciplinary hearing took place.", "#2f7d32", "t")])
    assert _marked(out) == []
    assert html.unescape(re.sub(r"<[^>]+>", "", out)) == LETTER


def test_overlapping_quotes_do_not_nest():
    out = highlight(LETTER, [
        ("Your employment terminates with effect from 12 July 2025.", "#c62828", "a"),
        ("employment terminates with effect from 12 July", "#2f7d32", "b")])
    assert len(_marked(out)) == 1


def test_a_short_quote_is_ignored_rather_than_matched_loosely():
    out = highlight(LETTER, [("2019", "#2f7d32", "t")])
    assert _marked(out) == []


def test_the_title_carries_the_fact_it_came_from():
    out = highlight(LETTER, [("Your employment terminates with effect from "
                              "12 July 2025.", "#c62828",
                              "employment termination = 2025-07-12 (disputed)")])
    assert 'title="employment termination = 2025-07-12 (disputed)"' in out


# ── finding the original document behind an extracted one ───────────────

def _load_document_text():
    text = SRC.read_text()
    start = text.index("def _document_text(")
    end = text.index("def _highlight(")
    ns: dict = {}
    exec(compile("from pathlib import Path\n" + text[start:end], "app_doc", "exec"), ns)
    return ns["_document_text"]


document_text = _load_document_text()


class _Tdg:
    def __init__(self, source_text=""):
        self.source_text = source_text


def _workspace_with(tmp_path, filenames):
    (tmp_path / "documents").mkdir(parents=True)
    for name in filenames:
        (tmp_path / "documents" / name).write_text(
            f"Contents of {name}, stating a date of 12 July 2025 plainly.")
    return tmp_path


def test_the_extracted_text_is_preferred_when_present(tmp_path):
    ws = _workspace_with(tmp_path, ["et1_claim.txt"])
    body = "A long enough source text carried by the extracted file itself."
    text, origin = document_text(ws, "et1", _Tdg(body))
    assert text == body and "extracted" in origin


def test_a_differently_named_file_is_still_found(tmp_path):
    """The graph may be called "et1" while the file is "et1_claim.txt"."""
    ws = _workspace_with(tmp_path, ["et1_claim.txt"])
    text, origin = document_text(ws, "et1", _Tdg())
    assert "12 July 2025" in text
    assert origin == "et1_claim.txt"


def test_an_exact_match_wins_over_a_near_one(tmp_path):
    ws = _workspace_with(tmp_path, ["et1.txt", "et1_claim.txt"])
    _, origin = document_text(ws, "et1", _Tdg())
    assert origin == "et1.txt"


def test_an_ambiguous_name_is_not_guessed_at(tmp_path):
    """Two plausible files and no exact match: show nothing rather than the
    wrong document, since the reader would have no way to tell."""
    ws = _workspace_with(tmp_path, ["et1_claim.txt", "et1_response.txt"])
    text, origin = document_text(ws, "et1", _Tdg())
    assert text == "" and origin == ""


def test_missing_everywhere_reports_nothing_found(tmp_path):
    ws = _workspace_with(tmp_path, ["dismissal_letter.txt"])
    text, origin = document_text(ws, "grounds_of_resistance", _Tdg())
    assert text == "" and origin == ""
