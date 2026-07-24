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
    at = _run()
    assert not at.exception
    assert at.dataframe, "timeline table should render"
    body = json.dumps(at.dataframe[0].value.to_dict()
                      if hasattr(at.dataframe[0].value, "to_dict")
                      else str(at.dataframe[0].value))
    assert "disputed" in body


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
