"""Timebar viewer — a Streamlit app for non-technical users (Phase 1.4).

Run with:  tdg-chrono view [workspace-folder]

Design rules:
  - every screen opens with a plain-language explanation of what it shows;
  - every button says what it does in normal words;
  - every theoretical concept has a glossary entry and inline help;
  - documents live in a user-chosen workspace folder on disk
    (workspace/documents, workspace/tdgs, workspace/out, corrections.json)
    and can be uploaded and downloaded from the browser;
  - all edits go through the corrections file: reversible, source files
    never modified.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import streamlit as st

from tdg_core.io import build_tdg
from tdg_chrono import capabilities
from tdg_chrono.chronology import build_chronology
from tdg_chrono.corrections import (Correction, append_correction,
                                    apply_corrections, load_corrections,
                                    mark_confirmed)
from tdg_chrono.exports import to_csv, to_xlsx, to_json

GLOSSARY = {
    "Fact": "One dated statement found in a document — e.g. “employment "
            "terminated on 12 July 2025”. Every fact keeps the exact sentence "
            "it came from, so you can always check it.",
    "Timeline (chronology)": "All facts from all documents, merged into one "
            "list of real-world events sorted by date. When several documents "
            "mention the same event, they share one row.",
    "Disputed": "Two documents describe the same event with different dates. "
            "The tool shows BOTH dates and BOTH quotes — it never picks a "
            "winner for you.",
    "Single source": "Only one document mentions this event.",
    "Agreed": "At least two documents state the same date for this event.",
    "Derived": "No document states this date directly — it was computed, "
            "e.g. “28 days after service”. The row shows the calculation.",
    "Unplaced": "A fact the tool could not put on the timeline (no readable "
            "date). It is listed rather than hidden, so nothing is lost.",
    "Dependency (arrow in the graph)": "One date is defined relative to "
            "another — “payment due 30 days after the effective date”. If the "
            "first date moves, the second moves with it.",
    "Rule pack": "A statute's time limit expressed as data (a small folder "
            "of files), e.g. “a claim must be presented within three months "
            "beginning with the effective date of termination”. The tool "
            "reads the rule from the statute's own wording.",
    "Anchor": "The event a legal time limit counts from (e.g. the "
            "termination date).",
    "Deadline check": "The tool finds the anchor in your documents, applies "
            "the rule pack's period with real calendar arithmetic, and shows "
            "every step of the working. It gives a derivation, not a verdict "
            "— the legal conclusion is yours.",
    "Correction": "Your fix to something the tool got wrong: change a date, "
            "remove a row, confirm a row, merge or split events. Corrections "
            "are stored in a separate file, re-applied on every rebuild, and "
            "fully reversible. Your documents are never modified.",
}

STATUS_ICON = {"agreed": "agreed", "disputed": "DISPUTED",
               "single_source": "single source", "derived": "derived"}


# ─── workspace ───────────────────────────────────────────────────────────

def _workspace() -> Path:
    import os
    default = st.session_state.get(
        "ws_path", os.environ.get("TIMEBAR_WORKSPACE", "./timebar-workspace"))
    st.sidebar.header("1 · Your case folder")
    st.sidebar.caption("Everything is stored here on your computer — "
                       "documents, results and your corrections. Nothing "
                       "leaves this machine unless you configure an "
                       "extractor to do so.")
    path = st.sidebar.text_input(
        "Folder for this case", value=default,
        help="Type a folder path. It will be created if it doesn't exist. "
             "Use a different folder per case.")
    st.session_state["ws_path"] = path
    ws = Path(path).expanduser()
    for sub in ("documents", "tdgs", "out"):
        (ws / sub).mkdir(parents=True, exist_ok=True)
    return ws


def _uploads(ws: Path) -> None:
    st.sidebar.header("2 · Add documents")
    files = st.sidebar.file_uploader(
        "Drop files here",
        type=["json", "txt", "md", "pdf", "docx"],
        accept_multiple_files=True,
        help="JSON files with extracted facts (TDG format) go straight onto "
             "the timeline. Raw documents (PDF/DOCX/TXT) are saved to the "
             "case folder; reading them needs an extractor to be installed.")
    for f in files or []:
        target = ws / ("tdgs" if f.name.endswith(".json") else "documents")
        (target / f.name).write_bytes(f.getbuffer())
    docs = sorted(p.name for p in (ws / "documents").iterdir())
    tdgs = sorted(p.name for p in (ws / "tdgs").iterdir())
    if tdgs:
        st.sidebar.write("**On the timeline (extracted):**")
        for n in tdgs:
            st.sidebar.write(f"· {n}")
    if docs:
        st.sidebar.write("**Raw documents (not yet extracted):**")
        for n in docs:
            st.sidebar.write(f"· {n}")


def _downloads(ws: Path, chron) -> None:
    st.sidebar.header("3 · Take your results")
    if chron is not None:
        x = to_xlsx(chron, ws / "out" / "chronology.xlsx")
        c = to_csv(chron, ws / "out" / "chronology.csv")
        j = to_json(chron, ws / "out" / "chronology.json")
        st.sidebar.download_button("Timeline as Excel (.xlsx)",
                                   x.read_bytes(), "chronology.xlsx",
                                   help="One row per event, quotes attached, "
                                        "disputed rows highlighted. Ready to "
                                        "review or drop into a bundle.")
        st.sidebar.download_button("Timeline as CSV", c.read_bytes(),
                                   "chronology.csv")
        st.sidebar.download_button("Timeline as JSON", j.read_bytes(),
                                   "chronology.json")
    corr = ws / "corrections.json"
    if corr.exists():
        st.sidebar.download_button("My corrections (backup)",
                                   corr.read_bytes(), "corrections.json",
                                   help="Your fixes, in one file. Keep it "
                                        "with the case; restoring it "
                                        "restores your edits.")


# ─── data ────────────────────────────────────────────────────────────────

def _load_bundle(ws: Path):
    tdgs, failures = {}, []
    for p in sorted((ws / "tdgs").glob("*.json")):
        try:
            t = build_tdg(json.loads(p.read_text()))
            if t.document_id in ("unknown", "", "cli_input") or t.document_id in tdgs:
                t.document_id = p.stem
            tdgs[t.document_id] = t
        except Exception as e:  # noqa: BLE001
            failures.append(f"{p.name}: {e}")
    return tdgs, failures


def _build(ws: Path):
    tdgs, failures = _load_bundle(ws)
    if not tdgs:
        return None, None, failures
    corrections = load_corrections(ws / "corrections.json")
    outcome = apply_corrections(tdgs, corrections)
    chron = build_chronology(outcome.tdgs, overrides=outcome.overrides)
    mark_confirmed(chron, outcome.accepted)
    if outcome.rejected:
        chron.meta["rejected"] = outcome.rejected
    return chron, outcome, failures


def _correct(ws: Path, **kwargs):
    append_correction(ws / "corrections.json", Correction(**kwargs))
    st.rerun()


# ─── tabs ────────────────────────────────────────────────────────────────

def _timeline_tab(ws: Path, chron, outcome):
    with st.expander("What am I looking at?"):
        st.write(
            "Each row is **one real-world event**, built from every document "
            "that mentions it. agreed = documents state the same date · "
            "disputed = documents disagree (both versions shown) · "
            "single source = one document only · derived = computed "
            "from another date. Click a row to see the exact quotes. "
            "Anything without a readable date is listed under *Unplaced* — "
            "never deleted.")
    if chron is None:
        st.info("No extracted documents yet. Add TDG JSON files in the "
                "sidebar (or run the extractor on your raw documents), then "
                "the timeline appears here.")
        return

    rows = [{
        "Date": e.date.isoformat() if e.date else "—",
        "Event": e.label,
        "Status": STATUS_ICON.get(e.status, e.status),
        "Documents": ", ".join(sorted({s.doc_id for s in e.sources})),
        "Both dates (if disputed)": " vs ".join(v.value for v in e.disputed_values),
        "Confidence": f"{e.confidence:.2f}",
    } for e in chron.events]
    st.dataframe(rows, use_container_width=True, hide_index=True)

    labels = {f"{e.date or '—'} · {e.label}": e for e in chron.events}
    pick = st.selectbox("Inspect an event (see the quotes it came from)",
                        ["—"] + list(labels))
    if pick != "—":
        e = labels[pick]
        for s in e.sources:
            st.markdown(f"> **[{s.doc_id}]** “{s.quote}”")
        if e.derivation:
            st.caption(f"How this date was computed: {e.derivation}")
        _correction_buttons(ws, e)

    if chron.unplaced:
        with st.expander(f"Unplaced items ({len(chron.unplaced)}) — facts "
                         "without a readable date (kept, not deleted)"):
            for u in chron.unplaced:
                st.write(f"**{u.event.label}** — {u.reason}")
                for s in u.event.sources:
                    st.markdown(f"> [{s.doc_id}] “{s.quote}”")
    if outcome and outcome.rejected:
        with st.expander(f"Rows you removed ({len(outcome.rejected)}) — "
                         "reversible"):
            for r in outcome.rejected:
                st.write(f"**{r['entity']}** [{r['doc_id']}/{r['fact_id']}] "
                         f"— “{r['quote']}”")
                if st.button("↩Put this row back",
                             key=f"undo{r['doc_id']}{r['fact_id']}"):
                    _correct(ws, op="accept", doc_id=r["doc_id"],
                             fact_id=r["fact_id"],
                             note="restored from viewer")


def _correction_buttons(ws: Path, e):
    st.write("**Fix this row** — changes are saved to your corrections file, "
             "applied on every rebuild, and reversible. Your documents are "
             "never modified.")
    src = e.sources[0]
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Confirm this row is correct", key=f"ok{e.event_id}",
                     help="Marks the row human-checked (confidence 1.0)."):
            _correct(ws, op="accept", doc_id=src.doc_id, fact_id=src.fact_id)
    with c2:
        if st.button("Remove this row (reversible)", key=f"rm{e.event_id}",
                     help="Takes the row off the timeline. It stays listed "
                          "under 'Rows you removed' and can be restored."):
            _correct(ws, op="reject", doc_id=src.doc_id, fact_id=src.fact_id,
                     note="removed in viewer")
    with c3:
        new = st.date_input("Correct date", value=e.date or date.today(),
                            key=f"dt{e.event_id}")
        if st.button("Save corrected date", key=f"ed{e.event_id}",
                     help="Use when the extracted date is wrong, or to "
                          "resolve a dispute by choosing the controlling "
                          "date. The original value is kept in the "
                          "corrections file."):
            _correct(ws, op="edit_date", doc_id=src.doc_id,
                     fact_id=src.fact_id, new_date=new.isoformat())
    if e.status == "disputed":
        st.warning("This event is **disputed** — the documents disagree. "
                   "If you know which date is controlling, set it with "
                   "“Save corrected date”. If these are actually two "
                   "different events, split them below.")
        if st.button("These are two different events — split them",
                     key=f"sp{e.event_id}"):
            _correct(ws, op="split", doc_id=e.sources[-1].doc_id,
                     fact_id=e.sources[-1].fact_id)


def _graph_tab(chron, tdgs):
    with st.expander("How to read this picture"):
        st.write(
            "Each box is an event from your timeline, placed under the "
            "document(s) that mention it. **An arrow means: this date is "
            "defined relative to that date** — e.g. “+ 28 days”. If the "
            "date at the arrow's tail moves, the date at its head moves "
            "too (try it in the *What-if* tab). A red box is a disputed "
            "event.")
    if chron is None:
        st.info("Build the timeline first (add documents in the sidebar).")
        return
    key_to_event = {(s.doc_id, s.fact_id): e
                    for e in chron.events for s in e.sources}
    lines = ["digraph G {", 'rankdir=LR; node [shape=box, fontsize=10];']
    for e in chron.events:
        color = ('"#ffdddd"' if e.status == "disputed"
                 else '"#eeeeff"' if e.status == "derived" else '"#eeffee"')
        label = f"{e.label}\\n{e.date or 'no date'}"
        lines.append(f'"{e.event_id}" [label="{label}", style=filled, '
                     f'fillcolor={color}];')
    for doc_id, t in tdgs.items():
        for dep in t.dependencies:
            a = key_to_event.get((doc_id, dep.from_id))
            b = key_to_event.get((doc_id, dep.to_id))
            if a and b and a is not b:
                expr = dep.constraint_expr or (
                    f"+{dep.delta_days}d" if dep.delta_days else "")
                lines.append(f'"{a.event_id}" -> "{b.event_id}" '
                             f'[label="{expr}", fontsize=9];')
    lines.append("}")
    st.graphviz_chart("\n".join(lines))
    st.caption("agreed / single source · disputed · derived. "
               "Events with no arrows have no stated relationship to other "
               "dates — that's normal.")


def _deadline_tab(ws: Path, tdgs):
    with st.expander("What this does"):
        st.write(
            "Checks your documents against a statute's time limit and shows "
            "**every step of the working**: which sentence of the statute "
            "the period came from, which event was used as the anchor, "
            "whether the first day counts (read from the statute's own "
            "wording), the calendar arithmetic, and which candidate events "
            "were considered and passed over. It gives a **derivation, not "
            "a verdict** — the legal conclusion is yours. If something "
            "needed is missing from the documents, it tells you exactly "
            "what to supply instead of guessing.")
    default = str(Path("rulepacks/uk/era-1996-s111").resolve())
    pack = st.text_input("Rule pack folder (the statute's time limit, as data)",
                         value=default,
                         help="A rule pack is a small folder: the statute's "
                              "clause, its vocabulary, and test cases. See "
                              "the Glossary.")
    if st.button("Check the deadline and show the working"):
        if not tdgs:
            st.info("Add documents first.")
            return
        from tdg_core.entailment import check_entailment, load_alias_file
        from tdg_core.trace import render_text
        rule_path = Path(pack) / "statute.tdg.json"
        if not rule_path.exists():
            st.error(f"No statute.tdg.json in {pack}")
            return
        aliases = Path(pack) / "aliases.json"
        if aliases.exists():
            load_alias_file(aliases)
        rule = build_tdg(json.loads(rule_path.read_text()))
        instance = capabilities.merged_instance(tdgs)
        results = check_entailment(rule, instance)
        for r in results:
            if r.verdict == "INDETERMINATE":
                st.warning("Could not compute an answer — here is why, and "
                           "what to supply:")
            st.code(render_text(r))


def _whatif_tab(tdgs):
    with st.expander("What this does"):
        st.write(
            "Answers “if this date moves, what else moves?”. Pick an event, "
            "give it a new date, and every date defined relative to it is "
            "recomputed along the arrows from the *Connections* picture. "
            "This is a preview — your documents and timeline are not "
            "changed.")
    if not tdgs:
        st.info("Add documents first.")
        return
    options = {f"[{d}] {f.entity} ({f.timex.date_parsed})": (d, f.id)
               for d, t in tdgs.items() for f in t.facts
               if f.timex.date_parsed}
    pick = st.selectbox("Which date moves?", ["—"] + list(options))
    if pick == "—":
        return
    new = st.date_input("Its new date", value=date.today())
    if st.button("Show what changes"):
        out = capabilities.whatif(tdgs, options[pick], new)
        st.write(f"Shift: **{out['shift_days']:+d} days**")
        st.table([{"Event": c["entity"], "Was": c["was"], "Becomes": c["now"],
                   "Why": c["via"]} for c in out["changes"]])
        st.caption(out["unchanged_note"])


def _glossary_tab():
    st.write("Plain-language definitions of everything this tool shows. "
             "The same explanations appear as boxes on each screen.")
    for term, text in GLOSSARY.items():
        st.markdown(f"**{term}** — {text}")


# ─── main ────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="Timebar", layout="wide")
    st.title("Timebar")
    st.write("**Your case documents in — a checkable timeline out.** Every "
             "date on the timeline links to the exact sentence it came "
             "from. Disagreements between documents are shown, never "
             "silently resolved. *Not legal advice.*")

    ws = _workspace()
    _uploads(ws)

    tdgs, _ = _load_bundle(ws)
    chron, outcome, failures = _build(ws)
    for f in failures:
        st.warning(f"Skipped a file it couldn't read: {f}")

    _downloads(ws, chron)

    t1, t2, t3, t4, t5 = st.tabs(
        ["Timeline", "Connections", "Deadline check",
         "What-if", "Glossary"])
    with t1:
        _timeline_tab(ws, chron, outcome)
    with t2:
        _graph_tab(chron, tdgs)
    with t3:
        _deadline_tab(ws, tdgs)
    with t4:
        _whatif_tab(tdgs)
    with t5:
        _glossary_tab()


main()
