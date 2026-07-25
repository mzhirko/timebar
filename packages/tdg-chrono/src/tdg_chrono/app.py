"""Timebar viewer: a Streamlit app for non-technical users (Phase 1.4).

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
from datetime import date, timedelta
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from tdg_core.io import build_tdg
from tdg_chrono import capabilities
from tdg_chrono.chronology import build_chronology
from tdg_chrono.corrections import (Correction, append_correction,
                                    apply_corrections, load_corrections,
                                    mark_confirmed)
from tdg_chrono.exports import to_csv, to_xlsx, to_json

GLOSSARY = {
    "Fact": "One dated statement found in a document, such as \"employment "
            "terminated on 12 July 2025\". Every fact keeps the exact "
            "sentence it came from, so you can always check it.",
    "Timeline (chronology)": "All facts from all documents, merged into one "
            "list of real-world events sorted by date. When several documents "
            "mention the same event, they share one row.",
    "Disputed": "Two documents describe the same event with different dates. "
            "The tool shows both dates and both quotes. It never picks a "
            "winner for you.",
    "Single source": "Only one document mentions this event.",
    "Agreed": "At least two documents state the same date for this event.",
    "Derived": "No document states this date directly. It was computed, for "
            "example \"28 days after service\". The row shows the working.",
    "Unplaced": "A fact the tool could not put on the timeline, because it "
            "has no readable date. It is listed rather than hidden, so "
            "nothing is lost.",
    "Dependency (arrow in the graph)": "One date is defined relative to "
            "another, such as \"payment due 30 days after the effective "
            "date\". If the first date moves, the second moves with it.",
    "Rule pack": "A statute's time limit expressed as data, in a small "
            "folder of files. For example, \"a claim must be presented "
            "within three months beginning with the effective date of "
            "termination\". The tool reads the rule from the statute's own "
            "wording.",
    "Anchor": "The event a legal time limit counts from, such as the "
            "termination date.",
    "Deadline check": "The tool finds the anchor in your documents, applies "
            "the rule pack's period with real calendar arithmetic, and shows "
            "every step of the working. It gives a derivation, not a "
            "verdict. The legal conclusion is yours.",
    "Correction": "Your fix to something the tool got wrong: change a date, "
            "remove a row, confirm a row, merge or split events. Corrections "
            "are stored in a separate file, re-applied on every rebuild, and "
            "fully reversible. Your documents are never modified.",
}

STATUS_ICON = {"agreed": "agreed", "disputed": "DISPUTED",
               "single_source": "single source", "derived": "derived"}

# Shown in a dropdown when nothing is chosen yet.
NOTHING_SELECTED = "(nothing selected)"
NO_DATE = "no date"

# One colour per status, used identically on the timeline, the map and the
# comparison, so a colour means the same thing wherever it appears.
STATUS_COLOUR = {
    "agreed": "#2f7d32",         # documents match
    "disputed": "#c62828",       # documents conflict
    "single_source": "#5c6bc0",  # only one document says it
    "derived": "#8e5db0",        # calculated, not stated
}
STATUS_HELP = {
    "agreed": "two or more documents give the same date",
    "disputed": "documents give different dates",
    "single_source": "only one document mentions it",
    "derived": "worked out from another date",
}


# ─── shared data shaping ─────────────────────────────────────────────────

def _event_rows(chron) -> pd.DataFrame:
    """One row per event, for the table and the filters."""
    return pd.DataFrame([{
        "Date": e.date,
        "Event": e.label,
        "Status": e.status,
        "Documents": ", ".join(sorted({s.doc_id for s in e.sources})),
        "Values": " vs ".join(v.value for v in e.disputed_values) or (
            e.sources[0].value or ""),
        "Confidence": round(e.confidence, 2),
        "_id": e.event_id,
    } for e in chron.events])


def _claim_rows(chron) -> pd.DataFrame:
    """One row per document's claim about an event.

    A disputed event produces one row per version, which is what lets the
    timeline draw the disagreement as a visible gap rather than a single
    point hiding two dates.
    """
    rows = []
    for e in chron.events:
        for s in e.sources:
            when = s.date_parsed or e.date
            if when is None:
                continue
            rows.append({
                "Date": when, "Event": e.label, "Status": e.status,
                "Document": s.doc_id, "Quote": (s.quote or "")[:160],
                "_id": e.event_id,
            })
    return pd.DataFrame(rows)


def _apply_filters(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Plain-language filters. Every control says what it does."""
    if df.empty:
        return df
    dated = df[df["Date"].notna()]
    c1, c2 = st.columns([2, 3])
    with c1:
        text = st.text_input("Search events and documents", key=f"q{key}",
                             placeholder="e.g. termination")
    with c2:
        statuses = st.multiselect(
            "Show which kinds of row", options=list(STATUS_COLOUR),
            default=list(STATUS_COLOUR), key=f"s{key}",
            format_func=lambda s: f"{s.replace('_', ' ')} ({STATUS_HELP[s]})")

    if not dated.empty:
        lo, hi = dated["Date"].min(), dated["Date"].max()
        if lo < hi:
            start, end = st.slider(
                "Only show events between these dates",
                min_value=lo, max_value=hi, value=(lo, hi),
                key=f"d{key}")
            df = df[df["Date"].isna() | ((df["Date"] >= start) & (df["Date"] <= end))]

    if statuses:
        df = df[df["Status"].isin(statuses)]
    if text:
        needle = text.lower()
        hay = df.astype(str).apply(lambda r: " ".join(r).lower(), axis=1)
        df = df[hay.str.contains(needle, regex=False)]
    return df


# ─── workspace ───────────────────────────────────────────────────────────

def _workspace() -> Path:
    import os
    default = st.session_state.get(
        "ws_path", os.environ.get("TIMEBAR_WORKSPACE", "./timebar-workspace"))
    st.sidebar.header("1. Your case folder")
    st.sidebar.caption("Everything is stored here on your computer: "
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
    st.sidebar.header("2. Add documents")
    files = st.sidebar.file_uploader(
        "Drop files here",
        type=["json", "txt", "md", "pdf", "docx"],
        accept_multiple_files=True,
        help="JSON files with extracted facts (TDG format) go straight onto "
             "the timeline. Raw documents (PDF, DOCX, TXT) are saved to the "
             "case folder. Reading those needs an extractor installed.")
    for f in files or []:
        target = ws / ("tdgs" if f.name.endswith(".json") else "documents")
        (target / f.name).write_bytes(f.getbuffer())
    docs = sorted(p.name for p in (ws / "documents").iterdir())
    tdgs = sorted(p.name for p in (ws / "tdgs").iterdir())
    if tdgs:
        st.sidebar.write("**On the timeline (extracted):**")
        for n in tdgs:
            st.sidebar.write(f"- {n}")
    if docs:
        st.sidebar.write("**Raw documents (not yet extracted):**")
        for n in docs:
            st.sidebar.write(f"- {n}")


def _downloads(ws: Path, chron) -> None:
    st.sidebar.header("3. Take your results")
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
            "that mentions it.")
        st.markdown(
            "- **agreed**: two or more documents state the same date\n"
            "- **disputed**: documents disagree, and both versions are shown\n"
            "- **single source**: only one document mentions it\n"
            "- **derived**: computed from another date, with the working shown"
        )
        st.write(
            "On the chart, a red line joining two dots is a disagreement: "
            "each dot is one document's version, and the line spans the gap "
            "between them. Hover any dot to see which document said it. "
            "Anything without a readable date is listed under Unplaced, "
            "never deleted.")
    if chron is None:
        st.info("No extracted documents yet. Add TDG JSON files in the "
                "sidebar, or run the extractor on your raw documents, and "
                "the timeline will appear here.")
        return

    events = _event_rows(chron)
    st.caption(f"{len(chron.events)} events from "
               f"{len(chron.meta.get('documents', []))} documents. "
               "Narrow them down below; the chart and the table both follow "
               "your choices.")
    shown = _apply_filters(events, "tl")
    if shown.empty:
        st.info("Nothing matches those filters. Widen them to see events again.")
        return

    _timeline_chart(_claim_rows(chron), set(shown["_id"]))

    st.write("**The same events as a table.** Click a column heading to sort.")
    table = shown.drop(columns=["_id"]).copy()
    table["Date"] = table["Date"].map(lambda d: d.isoformat() if d else NO_DATE)
    table["Status"] = table["Status"].map(lambda s: STATUS_ICON.get(s, s))
    st.dataframe(table, width="stretch", hide_index=True)

    visible = [e for e in chron.events if e.event_id in set(shown["_id"])]
    labels = {f"{e.date or NO_DATE}: {e.label}": e for e in visible}
    pick = st.selectbox("Inspect an event and see the quotes it came from",
                        [NOTHING_SELECTED] + list(labels))
    if pick != NOTHING_SELECTED:
        e = labels[pick]
        for s in e.sources:
            st.markdown(f"> **[{s.doc_id}]** \"{s.quote}\"")
        if e.derivation:
            st.caption(f"How this date was computed: {e.derivation}")
        _correction_buttons(ws, e)

    if chron.unplaced:
        with st.expander(f"Unplaced items ({len(chron.unplaced)}): facts "
                         "without a readable date, kept rather than deleted"):
            for u in chron.unplaced:
                st.write(f"**{u.event.label}**: {u.reason}")
                for s in u.event.sources:
                    st.markdown(f"> [{s.doc_id}] \"{s.quote}\"")
    if outcome and outcome.rejected:
        with st.expander(f"Rows you removed ({len(outcome.rejected)}), "
                         "all reversible"):
            for r in outcome.rejected:
                st.write(f"**{r['entity']}** [{r['doc_id']}/{r['fact_id']}]: "
                         f"\"{r['quote']}\"")
                if st.button("Put this row back",
                             key=f"undo{r['doc_id']}{r['fact_id']}"):
                    _correct(ws, op="accept", doc_id=r["doc_id"],
                             fact_id=r["fact_id"],
                             note="restored from viewer")


def _timeline_chart(claims: pd.DataFrame, keep: set) -> None:
    """A timeline drawn as a timeline: time across, one event per row.

    Each dot is one document's claim about an event. Where documents
    disagree, the two dots sit apart and a line spans the gap, so a
    disagreement is something you see rather than something you read.
    """
    claims = claims[claims["_id"].isin(keep)]
    if claims.empty:
        st.info("No dated events to plot for this selection.")
        return

    order = (claims.groupby("Event")["Date"].min().sort_values().index.tolist())
    height = max(180, 34 * len(order))
    colour = alt.Color(
        "Status:N",
        scale=alt.Scale(domain=list(STATUS_COLOUR), range=list(STATUS_COLOUR.values())),
        legend=alt.Legend(title="How well the documents agree"))

    # The span between the earliest and latest version of one event.
    spans = (claims.groupby(["Event", "Status"], as_index=False)
             .agg(Start=("Date", "min"), End=("Date", "max")))
    spans = spans[spans["Start"] != spans["End"]]

    base = alt.Chart(claims).encode(
        y=alt.Y("Event:N", sort=order, title=None,
                axis=alt.Axis(labelLimit=280)))
    dots = base.mark_circle(size=150, opacity=0.95).encode(
        x=alt.X("Date:T", title="Date"),
        color=colour,
        tooltip=["Event:N", "Date:T", "Document:N", "Status:N", "Quote:N"])
    layers = [dots]
    if not spans.empty:
        layers.insert(0, alt.Chart(spans).mark_rule(size=3, opacity=0.55).encode(
            y=alt.Y("Event:N", sort=order, title=None),
            x="Start:T", x2="End:T", color=colour))
    st.altair_chart(alt.layer(*layers).properties(height=height)
                    .interactive(), width="stretch")


def _correction_buttons(ws: Path, e):
    st.write("**Fix this row.** Changes are saved to your corrections file, "
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
        st.warning("This event is **disputed**: the documents give different "
                   "dates. If you know which date is controlling, set it "
                   "with \"Save corrected date\" above. If these are "
                   "actually two different events, split them below.")
        if st.button("These are two different events, split them",
                     key=f"sp{e.event_id}"):
            _correct(ws, op="split", doc_id=e.sources[-1].doc_id,
                     fact_id=e.sources[-1].fact_id)


def _graph_tab(chron, tdgs):
    with st.expander("How to read this picture"):
        st.write(
            "This is a map of **which document said what**. Rounded boxes on "
            "the left are your documents. Boxes on the right are the events "
            "on your timeline. A line means that document is where the event "
            "came from, and the label on the line is the date that document "
            "gives.")
        st.write(
            "Two lines arriving at one event with **different dates** is a "
            "disagreement, drawn in red. An event with a single line rests "
            "on one document alone. Dashed arrows between events mean one "
            "date is calculated from another, so moving the first moves the "
            "second.")
    if chron is None:
        st.info("Build the timeline first by adding documents in the sidebar.")
        return

    show_unplaced = st.checkbox(
        "Also show facts with no readable date", value=False,
        help="These are never deleted, only left off the timeline. Turn this "
             "on to see what they are and which document they came from.")

    events = list(chron.events) + (
        [u.event for u in chron.unplaced] if show_unplaced else [])
    if not events:
        st.info("Nothing to draw yet.")
        return

    docs = sorted({s.doc_id for e in events for s in e.sources})
    lines = ["digraph G {", "rankdir=LR;", "graph [splines=spline, nodesep=0.3];",
             'node [fontname="Helvetica", fontsize=10];',
             'edge [fontname="Helvetica", fontsize=9, color="#777777"];']

    lines.append('subgraph cluster_docs { label="Your documents"; '
                 'style=dashed; color="#bbbbbb"; fontsize=11;')
    for d in docs:
        lines.append(f'"doc::{d}" [label="{_esc(d)}", shape=box, style="rounded,filled", '
                     f'fillcolor="#eef2f7", color="#8fa4bf"];')
    lines.append("}")

    placed = {e.event_id for e in chron.events}
    for e in events:
        undated = e.event_id not in placed
        fill = "#f2f2f2" if undated else {
            "disputed": "#fdecec", "derived": "#f3ecfa",
            "agreed": "#eaf5ea"}.get(e.status, "#eef0fb")
        edge = "#bbbbbb" if undated else STATUS_COLOUR.get(e.status, "#5c6bc0")
        when = e.date.isoformat() if e.date else "no date"
        lines.append(f'"{e.event_id}" [label="{_esc(e.label)}\n{when}", '
                     f'shape=box, style="rounded,filled", fillcolor="{fill}", '
                     f'color="{edge}"];')

    for e in events:
        disputed = e.status == "disputed"
        for s in e.sources:
            value = s.date_parsed.isoformat() if s.date_parsed else (s.value or "")
            attrs = [f'label="{_esc(value)}"']
            if disputed:
                attrs.append('color="#c62828"')
                attrs.append('fontcolor="#c62828"')
            lines.append(f'"doc::{s.doc_id}" -> "{e.event_id}" [{", ".join(attrs)}];')

    # Calculated-from arrows, drawn differently so they read as a different
    # kind of relationship than "this document says so".
    key_to_event = {(s.doc_id, s.fact_id): e for e in events for s in e.sources}
    for doc_id, t in tdgs.items():
        for dep in t.dependencies:
            a = key_to_event.get((doc_id, dep.from_id))
            b = key_to_event.get((doc_id, dep.to_id))
            if a and b and a is not b:
                expr = dep.constraint_expr or (
                    f"+{dep.delta_days} days" if dep.delta_days else "calculated")
                lines.append(f'"{a.event_id}" -> "{b.event_id}" '
                             f'[label="{_esc(expr)}", style=dashed, '
                             f'color="#8e5db0", fontcolor="#8e5db0", constraint=false];')
    lines.append("}")
    st.graphviz_chart("\n".join(lines))

    n_disputed = sum(1 for e in chron.events if e.status == "disputed")
    bits = [f"{len(docs)} documents", f"{len(events)} events"]
    if n_disputed:
        bits.append(f"**{n_disputed} disagreement(s), in red**")
    st.caption("Showing " + ", ".join(bits) +
               ". An event with one incoming line rests on a single document.")


def _esc(text: str) -> str:
    """Make a label safe to drop inside a DOT string."""
    return str(text).replace("\\", " ").replace('"', "'").replace("\n", " ")


def _document_text(ws: Path, doc_id: str, tdg) -> tuple[str, str]:
    """The full text of a document, and where it was found.

    Three places, in order. The TDG may carry the source text; a bundle that
    shipped without it (the schema allows offsets plus a hash instead) may
    still have the original file in the case folder; and if neither exists
    there is nothing to show but the quotes themselves.
    """
    text = (tdg.source_text or "").strip() if tdg is not None else ""
    if len(text) > 40:
        return text, "the extracted file"

    docs_dir = ws / "documents"
    if docs_dir.is_dir():
        files = [p for p in sorted(docs_dir.iterdir()) if p.is_file()]
        # A document's id and its filename often differ: an extractor may name
        # the graph "et1" from a file called "et1_claim.txt". Take an exact
        # match first, then a unique one where either name contains the other,
        # and never guess when more than one file could be meant.
        exact = [p for p in files if p.stem == doc_id]
        loose = [p for p in files
                 if p.stem.startswith(doc_id) or doc_id.startswith(p.stem)]
        chosen = exact or (loose if len(loose) == 1 else [])
        for candidate in chosen:
            try:
                from tdg_chrono.loaders import load_document_raw
                return load_document_raw(candidate), candidate.name
            except Exception:  # noqa: BLE001 — unreadable file is not fatal
                continue
    return "", ""


def _highlight(text: str, marks: list[tuple[str, str, str]]) -> str:
    """Wrap each quoted passage in colour, without trusting char offsets.

    Offsets survive some extractors and not others, and a stale offset would
    highlight the wrong words, which is worse than highlighting nothing. The
    quote itself is matched in the text instead.
    """
    import html as _h

    # Collapse runs of whitespace, keeping each kept character's position in
    # the original. A quote is matched against the collapsed text, then both
    # of its ends are mapped straight back, so a sentence broken across lines
    # still highlights and never spills into the sentence after it.
    flat, origin = [], []
    prev_space = False
    for i, ch in enumerate(text):
        if ch.isspace():
            if prev_space:
                continue
            flat.append(" ")
            prev_space = True
        else:
            flat.append(ch)
            prev_space = False
        origin.append(i)
    flat = "".join(flat)

    spans = []
    for quote, colour, title in marks:
        needle = " ".join((quote or "").split())
        if len(needle) < 12:
            continue
        at = flat.find(needle)
        if at < 0:
            continue
        spans.append((origin[at], origin[at + len(needle) - 1] + 1,
                      colour, title))

    spans.sort()
    out, cursor = [], 0
    for start, end, colour, title in spans:
        if start < cursor:
            continue
        out.append(_h.escape(text[cursor:start]))
        out.append(f'<mark title="{_h.escape(title)}" style="background:{colour}33;'
                   f'border-bottom:2px solid {colour};padding:1px 0;">'
                   f'{_h.escape(text[start:end])}</mark>')
        cursor = end
    out.append(_h.escape(text[cursor:]))
    return "".join(out)


def _documents_tab(ws: Path, tdgs, chron):
    """Read a document, with the dates the tool took from it marked."""
    with st.expander("What this does"):
        st.write(
            "Shows a document as it was written, with every passage the tool "
            "took a date from highlighted. The colour tells you what became "
            "of that date on the timeline, so you can check the tool against "
            "the page it read.")
    if not tdgs:
        st.info("No documents yet. Add some in the sidebar.")
        return

    doc_id = st.selectbox("Which document do you want to read?", sorted(tdgs))
    tdg = tdgs.get(doc_id)
    text, origin = _document_text(ws, doc_id, tdg)

    status_of = {}
    if chron is not None:
        for e in chron.events:
            for s in e.sources:
                if s.doc_id == doc_id:
                    status_of[s.fact_id] = e.status
        for u in chron.unplaced:
            for s in u.event.sources:
                if s.doc_id == doc_id:
                    status_of.setdefault(s.fact_id, "unplaced")

    facts = list(tdg.facts) if tdg is not None else []
    st.caption(f"{len(facts)} dated passage(s) found in this document"
               + (f", read from {origin}." if origin else "."))

    legend = " &nbsp; ".join(
        f'<span style="border-bottom:3px solid {c};">{s.replace("_", " ")}</span>'
        for s, c in STATUS_COLOUR.items())
    st.markdown("Highlight colours: " + legend + ' &nbsp; '
                '<span style="border-bottom:3px solid #999;">unplaced</span>',
                unsafe_allow_html=True)

    if not text:
        st.warning(
            "The full text of this document is not available. The extracted "
            "file did not keep it, and no original was found in the case "
            "folder. The passages the tool quoted are listed below instead.")
        for f in facts:
            st.markdown(f"> \"{f.sentence or f.timex.text}\"")
        return

    # Which field holds the real quote differs by extractor: one puts the
    # sentence in `sentence` and the bare value in `raw_text`, another does
    # the reverse and leaves a section heading behind. supporting_quote picks
    # whichever text actually states the fact's own date.
    from tdg_core.linking import supporting_quote

    marks = []
    for f in facts:
        status = status_of.get(f.id, "unplaced")
        colour = STATUS_COLOUR.get(status, "#999999")
        value = f.timex.value or f.timex.text or ""
        quote, _ = supporting_quote(f)
        marks.append((quote or f.sentence or f.timex.text or "",
                      colour, f"{f.entity} = {value} ({status})"))

    st.markdown(
        f'<div style="white-space:pre-wrap; font-family:Georgia,serif; '
        f'line-height:1.6; border:1px solid #ddd; border-radius:6px; '
        f'padding:1rem; max-height:32rem; overflow-y:auto;">'
        f'{_highlight(text, marks)}</div>',
        unsafe_allow_html=True)

    unmatched = [m for m in marks if len(" ".join(m[0].split())) < 12]
    if unmatched:
        st.caption(f"{len(unmatched)} passage(s) were too short to locate in "
                   "the text and are not highlighted.")

    st.download_button("Download this document", text.encode("utf-8"),
                       f"{doc_id}.txt",
                       help="The text exactly as the tool read it.")


def _compare_tab(chron):
    """Two documents, side by side, aligned on the events they share."""
    with st.expander("What this does"):
        st.write(
            "Puts two documents next to each other and lines up the events "
            "they both mention, so you can see at a glance where they agree, "
            "where they give different dates, and what each one mentions "
            "that the other leaves out.")
    if chron is None:
        st.info("Build the timeline first by adding documents in the sidebar.")
        return

    docs = sorted({s.doc_id for e in chron.events for s in e.sources})
    if len(docs) < 2:
        st.info("Add a second document to compare. With only one document "
                "there is nothing to line it up against.")
        return

    c1, c2 = st.columns(2)
    left = c1.selectbox("Left-hand document", docs, index=0)
    right = c2.selectbox("Right-hand document", docs,
                         index=1 if len(docs) > 1 else 0)
    if left == right:
        st.info("Pick two different documents.")
        return

    rows = []
    for e in chron.events:
        by_doc = {}
        for s in e.sources:
            by_doc.setdefault(s.doc_id, s)
        a, b = by_doc.get(left), by_doc.get(right)
        if not a and not b:
            continue
        av = (a.date_parsed.isoformat() if a and a.date_parsed
              else (a.value if a else ""))
        bv = (b.date_parsed.isoformat() if b and b.date_parsed
              else (b.value if b else ""))
        if a and b:
            verdict = "same date" if av == bv else "DIFFERENT DATES"
        else:
            verdict = f"only in {left}" if a else f"only in {right}"
        rows.append({"Event": e.label, left: av or "-", right: bv or "-",
                     "Comparison": verdict})

    frame = pd.DataFrame(rows)
    counts = frame["Comparison"].value_counts().to_dict()
    m1, m2, m3 = st.columns(3)
    m1.metric("Both documents agree", counts.get("same date", 0))
    m2.metric("Dates conflict", counts.get("DIFFERENT DATES", 0))
    m3.metric("Only one mentions it",
              sum(v for k, v in counts.items() if k.startswith("only in")))

    only_shared = st.checkbox(
        "Hide events that only one of them mentions", value=False,
        help="Leaves just the events both documents talk about, which is "
             "where agreement and disagreement can be judged.")
    if only_shared:
        frame = frame[~frame["Comparison"].str.startswith("only in")]

    st.dataframe(frame, width="stretch", hide_index=True)
    if counts.get("DIFFERENT DATES"):
        st.warning(f"{counts['DIFFERENT DATES']} event(s) where these two "
                   "documents give different dates. Open them in the Timeline "
                   "tab to read the quotes and decide which is right.")


def _deadline_tab(ws: Path, tdgs):
    with st.expander("What this does"):
        st.write(
            "Checks your documents against a statute's time limit and shows "
            "**every step of the working**: which sentence of the statute "
            "the period came from, which event was used as the anchor, "
            "whether the first day counts (read from the statute's own "
            "wording), the calendar arithmetic, and which candidate events "
            "were considered and passed over. It gives a **derivation, not "
            "a verdict**. The legal conclusion is yours. If something it "
            "needs is missing from the documents, it tells you exactly what "
            "to supply instead of guessing.")
    # Only offer a path that exists. The bundled packs sit in the source tree,
    # which an installed copy of the tool does not have, so a hardcoded
    # relative path pointed at nothing whenever the viewer ran from anywhere
    # but the repository root.
    candidates = [
        Path.cwd() / "rulepacks",
        Path(__file__).resolve().parents[4] / "rulepacks",
    ]
    found = next((c / "uk" / "era-1996-s111" for c in candidates
                  if (c / "uk" / "era-1996-s111").is_dir()), None)
    pack = st.text_input("Rule pack folder (the statute's time limit, as data)",
                         value=str(found) if found else "",
                         placeholder="path to a rule pack folder",
                         help="A rule pack is a small folder holding the "
                              "statute's clause, its vocabulary and test "
                              "cases. See the Glossary.")
    if not found:
        st.caption("No bundled rule packs found next to this install. Enter "
                   "the path to a pack, or clone the repository to use the "
                   "ones it ships with.")
    if st.button("Check the deadline and show the working"):
        if not tdgs:
            st.info("Add documents first.")
            return
        from tdg_core.entailment import check_entailment, use_rulepack_vocabulary
        from tdg_core.trace import render_text
        rule_path = Path(pack) / "statute.tdg.json"
        if not rule_path.exists():
            st.error(f"No statute.tdg.json in {pack}")
            return
        aliases = Path(pack) / "aliases.json"
        if aliases.exists():
            use_rulepack_vocabulary(aliases)
        rule = build_tdg(json.loads(rule_path.read_text()))
        instance = capabilities.merged_instance(tdgs)
        results = check_entailment(rule, instance)
        for r in results:
            if r.verdict == "INDETERMINATE":
                st.warning("Could not compute an answer. Here is why, and "
                           "what to supply:")
            st.code(render_text(r))


def _whatif_tab(tdgs):
    with st.expander("What this does"):
        st.write(
            "Answers the question \"if this date moves, what else moves?\". "
            "Pick an event, give it a new date, and every date defined "
            "relative to it is recomputed along the arrows from the "
            "Connections picture. This is a preview only. Your documents "
            "and timeline are not changed.")
    if not tdgs:
        st.info("Add documents first.")
        return
    options = {f"[{d}] {f.entity} ({f.timex.date_parsed})": (d, f.id)
               for d, t in tdgs.items() for f in t.facts
               if f.timex.date_parsed}
    pick = st.selectbox("Which date moves?", [NOTHING_SELECTED] + list(options))
    if pick == NOTHING_SELECTED:
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
        st.markdown(f"**{term}**")
        st.markdown(text)


# ─── main ────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="Timebar", layout="wide")
    st.title("Timebar")
    st.write("**Case documents in, a checkable timeline out.** Every date "
             "links to the exact sentence it came from, and disagreements "
             "between documents are shown rather than silently resolved.")
    st.caption("Assistive output. Check every row against its quoted source "
               "before relying on it. Not legal advice.")

    ws = _workspace()
    _uploads(ws)

    tdgs, _ = _load_bundle(ws)
    chron, outcome, failures = _build(ws)
    for f in failures:
        st.warning(f"Skipped a file it couldn't read: {f}")

    _downloads(ws, chron)

    t1, t2, t3, t4, t5, t6, t7 = st.tabs(
        ["Timeline", "Read a document", "Which document said what",
         "Compare two documents", "Deadline check", "What-if", "Glossary"])
    with t1:
        _timeline_tab(ws, chron, outcome)
    with t2:
        _documents_tab(ws, tdgs, chron)
    with t3:
        _graph_tab(chron, tdgs)
    with t4:
        _compare_tab(chron)
    with t5:
        _deadline_tab(ws, tdgs)
    with t6:
        _whatif_tab(tdgs)
    with t7:
        _glossary_tab()


main()
