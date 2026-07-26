"""What the tool says when the user gets it wrong.

Four ordinary mistakes used to end in a Python traceback: naming a document
that is not in the bundle, naming a fact that is not in it, mistyping a
date, and pointing at a file that does not exist. A traceback tells the
reader nothing about which of those they did, so each of these asserts that
the message names the mistake and, where possible, the way out.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from tdg_chrono.cli import _entry, main

BUNDLE = Path(__file__).parents[3] / "examples" / "showcase" / "tdgs"
SAMPLE = Path(__file__).parents[3] / "examples" / "sample-bundle"


def run(argv, capsys, monkeypatch):
    monkeypatch.setattr("sys.argv", ["tdg-chrono", *argv])
    code = _entry()
    out = capsys.readouterr()
    return code, out.out + out.err


# ── naming things that are not there ────────────────────────────────────

def test_unknown_document_lists_the_ones_that_exist(capsys, monkeypatch):
    code, text = run(["whatif", str(BUNDLE), "--set", "ghost:f1=2025-01-01"],
                     capsys, monkeypatch)
    assert code != 0
    assert "Traceback" not in text
    assert "no document 'ghost'" in text
    assert "et1_claim" in text, "the message should say what is available"


def test_unknown_fact_lists_the_facts_in_that_document(capsys, monkeypatch):
    code, text = run(["whatif", str(BUNDLE), "--set", "et1_claim:zz=2025-01-01"],
                     capsys, monkeypatch)
    assert code != 0
    assert "Traceback" not in text
    assert "no fact 'zz'" in text
    assert "service of the claim" in text


def test_a_missing_file_says_which_one(capsys, monkeypatch, tmp_path):
    code, text = run(["deadline", str(BUNDLE), "--rule",
                      str(tmp_path / "nope.json")], capsys, monkeypatch)
    assert code != 0
    assert "Traceback" not in text
    assert "no such file" in text and "nope.json" in text


# ── malformed input ─────────────────────────────────────────────────────

def test_a_mistyped_date_says_the_expected_format(capsys, monkeypatch):
    code, text = run(["whatif", str(BUNDLE), "--set", "et1_claim:f4=12/07/2025"],
                     capsys, monkeypatch)
    assert code != 0
    assert "Traceback" not in text
    assert "YYYY-MM-DD" in text


def test_a_fact_reference_without_a_colon_shows_the_shape(capsys, monkeypatch):
    code, text = run(["whatif", str(BUNDLE), "--set", "justafact=2025-01-01"],
                     capsys, monkeypatch)
    assert code != 0
    assert "DOCUMENT:FACT" in text


def test_broken_json_does_not_silently_vanish(capsys, monkeypatch, tmp_path):
    """A file that cannot be read must not leave a quietly shorter timeline."""
    src = tmp_path / "tdgs"
    src.mkdir()
    for f in SAMPLE.glob("*.json"):
        shutil.copy(f, src / f.name)
    (src / "broken.json").write_text("{ this is not json")
    code, text = run(["build", str(src), "-o", str(tmp_path / "out"),
                      "--from-tdgs", "--formats", "csv"], capsys, monkeypatch)
    assert "could not be read" in text
    assert "broken.json" in text
    assert "missing from this timeline" in text


# ── empty and near-empty bundles ────────────────────────────────────────

def test_an_empty_folder_says_what_it_looked_for(capsys, monkeypatch, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    code, text = run(["build", str(empty), "-o", str(tmp_path / "o"),
                      "--from-tdgs"], capsys, monkeypatch)
    assert code == 1
    assert "no usable documents" in text
    assert "TDG *.json" in text


def test_raw_documents_without_from_tdgs_suggests_the_flag(capsys, monkeypatch,
                                                           tmp_path):
    """The commonest first-run mistake: the right folder, the wrong mode."""
    folder = tmp_path / "docs"
    folder.mkdir()
    for f in SAMPLE.glob("*.json"):
        shutil.copy(f, folder / f.name)
    code, text = run(["build", str(folder), "-o", str(tmp_path / "o")],
                     capsys, monkeypatch)
    assert code == 1
    assert "--from-tdgs" in text


def test_a_folder_that_does_not_exist_is_refused_before_work_starts(
        capsys, monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        main(["build", str(tmp_path / "nowhere"), "-o", str(tmp_path / "o"),
              "--from-tdgs"])


# ── bundles that are valid but hold nothing to say ──────────────────────

def test_a_single_document_bundle_works(capsys, monkeypatch, tmp_path):
    one = tmp_path / "one"
    one.mkdir()
    shutil.copy(SAMPLE / "et1.json", one / "et1.json")
    code, text = run(["build", str(one), "-o", str(tmp_path / "o"),
                      "--from-tdgs", "--formats", "csv"], capsys, monkeypatch)
    assert code == 0
    assert "1 documents" in text


def test_a_document_with_no_dates_at_all(capsys, monkeypatch, tmp_path):
    folder = tmp_path / "nodates"
    folder.mkdir()
    (folder / "d.json").write_text(json.dumps({
        "schema_version": "1.0", "document_id": "d", "document_type": "legal",
        "source_text": "The respondent denies each and every allegation made.",
        "facts": [{"id": "f1", "entity": "denial", "role": "UNKNOWN",
                   "value": None, "raw_text": "",
                   "sentence": "The respondent denies each and every allegation made.",
                   "temporal_content": False, "start_char": 0, "end_char": 52}],
        "dependencies": []}))
    code, text = run(["build", str(folder), "-o", str(tmp_path / "o"),
                      "--from-tdgs", "--formats", "csv"], capsys, monkeypatch)
    assert code == 0, "a bundle with no dates is a valid answer, not a failure"
    assert "unplaced" in text


def test_context_on_a_question_matching_nothing(capsys, monkeypatch):
    code, text = run(["context", str(BUNDLE), "--about", "bankruptcy"],
                     capsys, monkeypatch)
    assert code == 0
    assert "no facts match this question" in text


def test_stale_on_a_date_nothing_depends_on(capsys, monkeypatch):
    code, text = run(["stale", str(BUNDLE), "--changed",
                      "grounds_of_resistance:f1"], capsys, monkeypatch)
    assert code == 0
    assert "Traceback" not in text


def test_whatif_on_a_date_nothing_depends_on_explains_itself(capsys, monkeypatch):
    code, text = run(["whatif", str(BUNDLE), "--set",
                      "dismissal_letter:f2=2025-09-05"], capsys, monkeypatch)
    assert code == 0
    assert "nothing is defined relative to" in text


def test_a_readme_is_not_treated_as_a_case_document(tmp_path, capsys, monkeypatch):
    """A bundle folder usually documents itself. That documentation is not
    evidence, and extracting dates from it produces facts about the folder."""
    from tdg_chrono.cli import _extract_bundle

    folder = tmp_path / "bundle"
    folder.mkdir()
    (folder / "README.md").write_text("This bundle was assembled on 1 May 2025.")
    (folder / "letter.txt").write_text("Employment ended on 12 July 2025.")

    # No model configured, so extraction cannot run; what matters is which
    # files it decided to hand over, reported before any model is needed.
    with pytest.raises(SystemExit):
        _extract_bundle(folder, "llm")
    text = capsys.readouterr().out
    assert "skipping README.md" in text
    assert "letter.txt" not in text.split("skipping")[0]


def test_a_folder_of_only_readme_reports_no_documents(tmp_path, capsys, monkeypatch):
    folder = tmp_path / "bundle"
    folder.mkdir()
    (folder / "README.md").write_text("Notes about this folder.")
    code, text = run(["build", str(folder), "-o", str(tmp_path / "o")],
                     capsys, monkeypatch)
    assert code == 1
    assert "no usable documents" in text
