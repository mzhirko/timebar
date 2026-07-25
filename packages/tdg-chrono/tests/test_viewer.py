"""Viewer tests (Phase 1.4) — run the Streamlit app headlessly with
st.testing.v1.AppTest against a real workspace, click through the core
flows, and assert the app never crashes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from test_chronology import bundle  # noqa: E402,F401

APP = Path(__file__).parent.parent / "src" / "tdg_chrono" / "app.py"


@pytest.fixture()
def workspace(tmp_path, bundle, monkeypatch):
    ws = tmp_path / "case"
    (ws / "tdgs").mkdir(parents=True)
    for doc_id, tdg in bundle.items():
        (ws / "tdgs" / f"{doc_id}.json").write_text(tdg.to_json())
    monkeypatch.setenv("TIMEBAR_WORKSPACE", str(ws))
    return ws


def _run(timeout=30):
    at = AppTest.from_file(str(APP), default_timeout=timeout)
    at.run()
    return at


def test_app_boots_without_errors(workspace):
    at = _run()
    assert not at.exception
    assert "Timebar" in at.title[0].value


def test_timeline_renders_with_dispute(workspace):
    """The disputed row must be visible in the data, not just in a heading.

    This previously passed on a column *header* that happened to contain the
    word "disputed"; renaming the column broke it while the table still
    showed the dispute perfectly well. Assert on the values instead.
    """
    at = _run()
    assert not at.exception
    assert at.dataframe, "timeline table should render"
    body = json.dumps(at.dataframe[0].value.to_dict()
                      if hasattr(at.dataframe[0].value, "to_dict")
                      else str(at.dataframe[0].value))
    assert "DISPUTED" in body.upper(), "the disputed status should be shown"
    # and both conflicting dates, so the reader can see what is in dispute
    assert "2025-07-12" in body and "2025-07-14" in body


def test_timeline_draws_a_chart(workspace):
    """The timeline should be drawn as a timeline, not only tabulated."""
    at = _run()
    assert not at.exception
    assert at.get("arrow_vega_lite_chart") or at.get("vega_lite_chart"), (
        "expected an Altair/Vega timeline chart to render")


def test_inspect_event_shows_quotes_and_buttons(workspace):
    at = _run()
    sel = at.selectbox[0]
    target = next(o for o in sel.options if "termination" in o)
    sel.select(target).run()
    assert not at.exception
    labels = [b.label for b in at.button]
    assert any("Confirm" in l for l in labels)
    assert any("Remove" in l for l in labels)
    assert any("corrected date" in l for l in labels)


def test_remove_row_writes_reversible_correction(workspace):
    at = _run()
    sel = at.selectbox[0]
    target = next(o for o in sel.options if "presentation of the claim" in o)
    sel.select(target).run()
    remove = next(b for b in at.button if "Remove" in b.label)
    remove.click().run()
    assert not at.exception
    corr = json.loads((workspace / "corrections.json").read_text())
    assert corr["corrections"][0]["op"] == "reject"
    # and the source TDGs are untouched
    et1 = json.loads((workspace / "tdgs" / "et1.json").read_text())
    assert any(f["id"] == "f3" for f in et1["facts"])


def test_glossary_defines_the_concepts(workspace):
    at = _run()
    text = " ".join(m.value for m in at.markdown)
    for term in ("Disputed", "Derived", "Rule pack", "Correction", "Anchor"):
        assert term in text


# ── the map, the comparison, and the filters ────────────────────────────

def test_map_shows_documents_as_nodes(workspace):
    """The map must show which document said what, not one document's sums.

    It previously drew only within-document arithmetic arrows, so a bundle
    of three documents rendered five boxes and one line, and the
    cross-document structure was collapsed away entirely.
    """
    at = _run()
    assert not at.exception
    dot = "\n".join(str(g.proto.spec) for g in at.get("graphviz_chart"))
    assert dot, "expected a graph to render"
    assert "Your documents" in dot, "documents should appear as their own nodes"
    for doc in ("dismissal_letter", "et1"):
        assert f'"doc::{doc}"' in dot, f"{doc} should be a node"
    assert "->" in dot, "documents should connect to the events they evidence"


def test_map_marks_a_disagreement_in_red(workspace):
    at = _run()
    dot = "\n".join(str(g.proto.spec) for g in at.get("graphviz_chart"))
    assert "#c62828" in dot, "a disputed event's edges should be red"
    assert "2025-07-12" in dot and "2025-07-14" in dot, (
        "each document's own date belongs on its edge")


def test_comparison_tab_lines_two_documents_up(workspace):
    at = _run()
    assert not at.exception
    frames = [f.value for f in at.dataframe]
    joined = " ".join(str(f) for f in frames)
    assert "Comparison" in joined or at.metric, (
        "the compare tab should render its table or its counters")


def test_filters_are_present_and_labelled(workspace):
    at = _run()
    assert not at.exception
    labels = [w.label for w in at.text_input] + [w.label for w in at.multiselect]
    assert any("Search" in (l or "") for l in labels), "expected a search box"
    assert any("kinds of row" in (l or "") for l in labels), (
        "expected a status filter in plain words")
