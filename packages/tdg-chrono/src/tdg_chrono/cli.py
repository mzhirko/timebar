"""One command (Phase 1.2): ``tdg-chrono build ./bundle/ -o ./out/``.

Documents in, chronology out. A bundle run must not die on one bad
file: per-document failures are collected and summarised, and the
chronology is built from whatever succeeded.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from tdg_core.io import build_tdg
from tdg_core.tdg import TemporalDependencyGraph

from tdg_chrono import __version__
from tdg_chrono.chronology import MergeOverrides, build_chronology
from tdg_chrono.corrections import (Correction, append_correction,
                                    apply_corrections, export_gold,
                                    load_corrections, mark_confirmed)
from tdg_chrono.exports import EXPORTERS
from tdg_chrono.loaders import load_document_text
from tdg_chrono import capabilities

DEFAULT_FORMATS = ["xlsx", "csv", "json"]


def _load_tdg_bundle(folder: Path) -> tuple[dict[str, TemporalDependencyGraph], list[str]]:
    tdgs, failures = {}, []
    for p in sorted(folder.glob("*.json")):
        try:
            tdg = build_tdg(json.loads(p.read_text()))
            if tdg.document_id in ("unknown", "", "cli_input") or tdg.document_id in tdgs:
                tdg.document_id = p.stem
            tdgs[tdg.document_id] = tdg
        except Exception as e:  # noqa: BLE001 — must not die on one bad file
            failures.append(f"{p.name}: {e}")
    return tdgs, failures


def _extract_bundle(folder: Path, extractor_name: str, **kwargs
                    ) -> tuple[dict[str, TemporalDependencyGraph], list[str]]:
    from tdg_core.extractor import load_extractor
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    try:
        extractor = load_extractor(extractor_name, **kwargs)
    except TypeError as e:
        raise SystemExit(
            f"extractor {extractor_name!r} does not accept these options "
            f"({', '.join(kwargs)}): {e}") from e
    except (ImportError, ValueError) as e:
        raise SystemExit(str(e)) from e
    tdgs, failures = {}, []
    docs = [p for p in sorted(folder.iterdir())
            if p.suffix.lower() in (".txt", ".md", ".pdf", ".docx")]
    if not docs:
        failures.append(f"no .txt/.md/.pdf/.docx documents found in {folder}")
    for i, p in enumerate(docs, 1):
        print(f"  [{i}/{len(docs)}] {p.name} ...", flush=True)
        try:
            text = load_document_text(p)
            tdgs[p.stem] = extractor.extract(text, document_id=p.stem,
                                             document_type="legal")
        except Exception as e:  # noqa: BLE001
            failures.append(f"{p.name}: {e}")
    return tdgs, failures


def _load_overrides(path: Path | None) -> MergeOverrides:
    if not path or not path.exists():
        return MergeOverrides()
    data = json.loads(path.read_text())
    return MergeOverrides(
        force_merge=[[tuple(k) for k in group]
                     for group in data.get("force_merge", [])],
        split=[tuple(k) for k in data.get("split", [])],
    )


def _cmd_view(args) -> int:
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print("The viewer needs the [viewer] extra:\n"
              "    pip install 'tdg-chrono[viewer]'", file=sys.stderr)
        return 1
    import subprocess
    app = Path(__file__).parent / "app.py"
    ws = Path(args.workspace).expanduser().resolve()
    ws.mkdir(parents=True, exist_ok=True)
    print(f"Opening the viewer for case folder: {ws}")
    return subprocess.call(
        [sys.executable, "-m", "streamlit", "run", str(app),
         "--server.headless", "false", "--browser.gatherUsageStats", "false"],
        cwd=str(ws.parent) if ws.parent.exists() else None,
        env={**__import__("os").environ, "TIMEBAR_WORKSPACE": str(ws)})


def _parse_key(s: str) -> tuple[str, str]:
    if ":" not in s:
        raise SystemExit(f"fact key must be doc:fact, got {s!r}")
    d, f = s.split(":", 1)
    return d, f


def _cmd_capability(args) -> int:
    from datetime import date as _date

    tdgs, failures = _load_tdg_bundle(args.input)
    for f in failures:
        print(f"  FAIL: skipped {f}", file=sys.stderr)
    if not tdgs:
        print("no TDG JSON found", file=sys.stderr)
        return 1

    if args.cmd == "interval":
        if args.between:
            out = capabilities.interval_between(
                tdgs, _parse_key(args.between[0]), _parse_key(args.between[1]))
        elif args.entity and args.doc and args.on:
            out = capabilities.interval_contains(
                tdgs, args.doc, args.entity, _date.fromisoformat(args.on))
        else:
            print("use --between A B, or --entity/--doc/--on", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(out, indent=2))
        else:
            print(out.get("answer") or out["relation"])
            print(f"  {out['derivation']}")
            for e in out.get("evidence", []):
                print(f"  [{e['role']}] \"{e['quote']}\"")
        return 0

    if args.cmd == "contradictions":
        out = capabilities.contradiction_report(tdgs)
        if args.json:
            print(json.dumps(out, indent=2))
        else:
            print(f"{out['count']} contradiction(s) across {len(out['documents'])} documents")
            for c in out["contradictions"]:
                print(f"\n  {c['value_a']}  vs  {c['value_b']}"
                      f"  (Δ{c['delta_days']}d, conf {c['confidence']})")
                print(f"    [{c['a']['doc']}] \"{c['a']['quote']}\"")
                print(f"    [{c['b']['doc']}] \"{c['b']['quote']}\"")
        return 0

    if args.cmd == "whatif":
        keypart, datepart = args.set.split("=", 1)
        out = capabilities.whatif(tdgs, _parse_key(keypart),
                                  _date.fromisoformat(datepart))
        if args.json:
            print(json.dumps(out, indent=2))
        else:
            print(f"shift: {out['shift_days']:+d} days")
            for ch in out["changes"]:
                print(f"  {ch['entity']}: {ch['was']} → {ch['now']}   via {ch['via']}")
            print(f"  ({out['unchanged_note']})")
        return 0

    if args.cmd == "deadline":
        from tdg_core.entailment import check_entailment, load_alias_file
        from tdg_core.trace import render_text, render_line

        rule_path = Path(args.rule)
        pack_aliases = rule_path.parent / "aliases.json"
        if pack_aliases.exists():
            load_alias_file(pack_aliases)
            print(f"loaded pack vocabulary: {pack_aliases}", file=sys.stderr)
        rule_tdg = build_tdg(json.loads(rule_path.read_text()))
        instance = capabilities.merged_instance(tdgs)
        results = check_entailment(
            rule_tdg, instance,
            acas_day_a=_date.fromisoformat(args.acas_a) if args.acas_a else None,
            acas_day_b=_date.fromisoformat(args.acas_b) if args.acas_b else None)
        if not results:
            print("no rule discoverable in the rule document", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps([r.to_dict() for r in results], indent=2))
        else:
            for i, r in enumerate(results):
                if i:
                    print("\n" + "─" * 60 + "\n")
                print(render_text(r) if args.explain else render_line(r))
        return 0 if all(r.verdict != "INDETERMINATE" for r in results) else 2
    return 1


def _cmd_correct(args) -> int:
    if args.ccmd == "list":
        for i, c in enumerate(load_corrections(args.file)):
            print(f"[{i}] {json.dumps(c.to_dict())}")
        return 0
    if args.ccmd == "export-gold":
        print(json.dumps(export_gold(load_corrections(args.file)), indent=2))
        return 0
    op = args.op.replace("-", "_")
    if op == "merge":
        keys = [k.split(":", 1) for k in args.key]
        if len(keys) < 2:
            print("merge needs at least two --key doc:fact", file=sys.stderr)
            return 1
        corr = Correction(op="merge", keys=keys, note=args.note)
    else:
        if not (args.doc and args.fact):
            print(f"{args.op} needs --doc and --fact", file=sys.stderr)
            return 1
        corr = Correction(op=op, doc_id=args.doc, fact_id=args.fact,
                          new_date=args.date, new_label=args.label,
                          note=args.note)
        if op == "edit_date" and not args.date:
            print("edit-date needs --date", file=sys.stderr)
            return 1
        if op == "edit_label" and not args.label:
            print("edit-label needs --label", file=sys.stderr)
            return 1
    append_correction(args.file, corr)
    print(f"recorded: {json.dumps(corr.to_dict())}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tdg-chrono",
        description="Build a chronology from a folder of legal documents or TDG JSON files.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="bundle folder in, chronology out")
    b.add_argument("input", type=Path, help="folder of documents (or of TDG JSON with --from-tdgs)")
    b.add_argument("-o", "--output", type=Path, default=Path("./out"))
    b.add_argument("--from-tdgs", action="store_true",
                   help="input folder contains already-extracted TDG *.json (no extractor, fully offline)")
    b.add_argument("--extractor", default="llm",
                   help="registered extractor name (default: llm; requires the matching extra)")
    b.add_argument("--model", default=None,
                   help="LLM extractor model, e.g. gemma3:4b (Ollama) or "
                        "gpt-4o-mini (OpenAI). Env fallback: TDG_LLM_MODEL. "
                        "Required for --extractor llm; no default.")
    b.add_argument("--base-url", default=None,
                   help="OpenAI-compatible endpoint, e.g. http://localhost:11434/v1 "
                        "for Ollama. Env fallback: OPENAI_BASE_URL.")
    b.add_argument("--allow-empty", action="store_true",
                   help="exit 0 even when extraction finds no dated facts "
                        "(default: exit 3 so scripts and CI notice)")
    b.add_argument("--formats", default=",".join(DEFAULT_FORMATS),
                   help=f"comma list of {sorted(EXPORTERS)} (default: {','.join(DEFAULT_FORMATS)})")
    b.add_argument("--corrections", type=Path, default=None,
                   help="corrections.json (accept/reject/edit/merge/split), "
                        "re-applied on every run; see tdg-chrono correct")
    b.add_argument("--overrides", type=Path, default=None,
                   help=argparse.SUPPRESS)  # legacy force_merge/split file
    b.add_argument("--merge-threshold", type=float, default=0.45)

    def _tdg_input(sp):
        sp.add_argument("input", type=Path, help="folder of TDG *.json")

    iv = sub.add_parser("interval", help="Allen relation between two facts, "
                        "or 'was X active on DATE'")
    _tdg_input(iv)
    iv.add_argument("--between", nargs=2, metavar="doc:fact",
                    help="relation between two facts")
    iv.add_argument("--entity", help="entity whose START/END define the interval")
    iv.add_argument("--doc", help="document holding the entity's facts")
    iv.add_argument("--on", metavar="DATE", help="date to test (ISO)")
    iv.add_argument("--json", action="store_true")

    cn = sub.add_parser("contradictions",
                        help="bundle-level conflict report with both quotes")
    _tdg_input(cn)
    cn.add_argument("--json", action="store_true")

    wf = sub.add_parser("whatif", help="move one date, recompute everything downstream")
    _tdg_input(wf)
    wf.add_argument("--set", required=True, metavar="doc:fact=DATE",
                    help="e.g. contract:f1=2025-08-01")
    wf.add_argument("--json", action="store_true")

    dl = sub.add_parser("deadline",
                        help="check the bundle against a rule pack's time limit (the funnel)")
    _tdg_input(dl)
    dl.add_argument("--rule", required=True,
                    help="statute.tdg.json (a sibling aliases.json is auto-loaded)")
    dl.add_argument("--explain", action="store_true")
    dl.add_argument("--json", action="store_true")
    dl.add_argument("--acas-a", metavar="DATE")
    dl.add_argument("--acas-b", metavar="DATE")

    vw = sub.add_parser("view", help="open the point-and-click viewer in your browser")
    vw.add_argument("workspace", nargs="?", default="./timebar-workspace",
                    help="case folder (created if missing)")

    c = sub.add_parser("correct", help="record a correction (re-applied on every build)")
    c.add_argument("file", type=Path, help="corrections.json (created if missing)")
    csub = c.add_subparsers(dest="ccmd", required=True)
    add = csub.add_parser("add")
    add.add_argument("--op", required=True,
                     choices=["accept", "reject", "edit-date", "edit-label", "merge", "split"])
    add.add_argument("--doc"); add.add_argument("--fact")
    add.add_argument("--date", help="for edit-date (ISO)")
    add.add_argument("--label", help="for edit-label")
    add.add_argument("--key", action="append", default=[],
                     help="for merge: doc:fact (repeatable)")
    add.add_argument("--note", default="")
    csub.add_parser("list")
    csub.add_parser("export-gold")

    args = parser.parse_args(argv)

    if args.cmd == "view":
        return _cmd_view(args)
    if args.cmd == "correct":
        return _cmd_correct(args)
    if args.cmd in ("interval", "contradictions", "whatif", "deadline"):
        return _cmd_capability(args)

    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    unknown = [f for f in formats if f not in EXPORTERS]
    if unknown:
        parser.error(f"unknown format(s) {unknown}; choose from {sorted(EXPORTERS)}")

    if not args.input.is_dir():
        parser.error(f"{args.input} is not a directory")

    if args.from_tdgs:
        tdgs, failures = _load_tdg_bundle(args.input)
        extractor_label = "none (pre-extracted TDGs)"
    else:
        tdgs, failures = _extract_bundle(args.input, args.extractor,
                                         model=args.model, base_url=args.base_url)
        extractor_label = args.extractor + (f" ({args.model})" if args.model else "")
        total_facts = sum(len(t.facts) for t in tdgs.values())
        if tdgs and total_facts == 0 and not args.allow_empty:
            print(f"\n{len(tdgs)} document(s) processed but ZERO dated facts "
                  "extracted. This usually means the extractor/model is "
                  "misconfigured, not that the documents contain no dates. "
                  "Not writing an empty chronology. "
                  "(Use --allow-empty to override.)", file=sys.stderr)
            for f in failures:
                print(f"  FAIL: {f}", file=sys.stderr)
            return 3

    if not tdgs:
        print("No documents could be processed:", file=sys.stderr)
        for f in failures:
            print(f"  FAIL: {f}", file=sys.stderr)
        return 1

    corrections = load_corrections(args.corrections) if args.corrections else []
    outcome = apply_corrections(tdgs, corrections)
    if args.corrections and any(c.was for c in corrections):
        # write back the originals captured at apply time — the gold harvest
        from tdg_chrono.corrections import save_corrections
        save_corrections(args.corrections, corrections)
    legacy = _load_overrides(args.overrides)
    outcome.overrides.force_merge.extend(legacy.force_merge)
    outcome.overrides.split.extend(legacy.split)

    chron = build_chronology(
        outcome.tdgs,
        overrides=outcome.overrides,
        merge_threshold=args.merge_threshold,
        meta={
            "extractor": extractor_label,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tool_version": __version__,
        },
    )

    mark_confirmed(chron, outcome.accepted)
    if outcome.rejected:
        chron.meta["rejected"] = outcome.rejected
        chron.meta["counts"]["rejected"] = len(outcome.rejected)
    if outcome.edits:
        chron.meta["edits_applied"] = outcome.edits

    args.output.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        written.append(EXPORTERS[fmt](chron, args.output / f"chronology.{fmt}"))

    c = chron.meta["counts"]
    print(f"\n{len(tdgs)} documents → {c['events']} events "
          f"({c['disputed']} disputed, {c['unplaced']} unplaced)")
    for p in written:
        print(f"  wrote {p}")
    if failures:
        print(f"\n{len(failures)} document(s) failed and were skipped:")
        for f in failures:
            print(f"  FAIL: {f}")
    return 0


def _entry() -> int:
    try:
        return main()
    except BrokenPipeError:
        import os
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0


if __name__ == "__main__":
    sys.exit(_entry())
