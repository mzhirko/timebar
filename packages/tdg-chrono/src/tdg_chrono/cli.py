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
from tdg_chrono.loaders import load_document_raw, process_with_report
from tdg_chrono import capabilities

DEFAULT_FORMATS = ["xlsx", "csv", "json"]


def _make_embedder(model: str | None, base_url: str | None):
    """Build an entity-name embedder, or None when none is configured.

    Any OpenAI-compatible endpoint works, hosted or local, and it need not
    be the same provider as the extraction or answering model. Falls back to
    lexical scoring if the service is unreachable rather than failing.
    """
    from tdg_chrono.models import resolve

    cfg = resolve("embed", model=model, base_url=base_url)
    if not cfg.configured:
        return None
    from tdg_core.embeddings import EmbeddingSimilarity
    kwargs = {"model": cfg.model, "api_key": cfg.effective_key}
    if cfg.base_url:
        kwargs["base_url"] = cfg.base_url
    return EmbeddingSimilarity(**kwargs)


def _load_tdg_bundle(folder: Path, matter_field: str = "matter"
                     ) -> tuple[dict[str, TemporalDependencyGraph], list[str]]:
    tdgs, failures = {}, []
    for p in sorted(folder.glob("*.json")):
        try:
            tdg = build_tdg(json.loads(p.read_text()), matter_field=matter_field)
            if tdg.document_id in ("unknown", "", "cli_input") or tdg.document_id in tdgs:
                tdg.document_id = p.stem
            tdgs[tdg.document_id] = tdg
        except Exception as e:  # noqa: BLE001 — must not die on one bad file
            failures.append(f"{p.name}: {e}")
    return tdgs, failures


def _extract_bundle(folder: Path, extractor_name: str, *, clean: bool = False,
                    **kwargs
                    ) -> tuple[dict[str, TemporalDependencyGraph], list[str],
                               dict[str, list]]:
    from tdg_core.extractor import load_extractor
    from tdg_chrono.recall import audit_recall

    tdgs, failures, missed = {}, [], {}
    candidates = [p for p in sorted(folder.iterdir())
                  if p.suffix.lower() in (".txt", ".md", ".pdf", ".docx")]
    # A bundle folder often carries a README describing the case. It is not
    # part of the case, and extracting dates from it produces facts about the
    # folder rather than the matter. Skipped rather than silently included,
    # and announced, so a document genuinely named README is not lost quietly.
    docs = [p for p in candidates if not p.stem.upper().startswith("README")]
    for skipped in [p for p in candidates if p not in docs]:
        print(f"  skipping {skipped.name} (looks like folder documentation, "
              "not a case document)", flush=True)
    if not docs:
        # Say so before asking for a model. Demanding --model first sends the
        # reader to fix the wrong thing when their real problem is that the
        # folder holds already-extracted JSON and wants --from-tdgs.
        json_files = [p for p in folder.glob("*.json")]
        hint = (" That folder holds "
                f"{len(json_files)} .json file(s); if those are already-"
                "extracted TDGs, add --from-tdgs." if json_files else "")
        failures.append(
            f"no .txt/.md/.pdf/.docx documents found in {folder}.{hint}")
        return tdgs, failures, missed

    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    try:
        extractor = load_extractor(extractor_name, **kwargs)
    except TypeError as e:
        raise SystemExit(
            f"error: extractor {extractor_name!r} does not accept these "
            f"options ({', '.join(kwargs)}): {e}") from e
    except (ImportError, ValueError) as e:
        raise SystemExit(f"error: {e}") from e
    for i, p in enumerate(docs, 1):
        print(f"  [{i}/{len(docs)}] {p.name} ...", flush=True)
        try:
            raw = load_document_raw(p)
            text, dropped = process_with_report(raw, clean=clean)
            if dropped:
                print(f"      --clean removed {len(dropped)} span(s): "
                      + "; ".join(d[:60] for d in dropped[:5])
                      + (" ..." if len(dropped) > 5 else ""), flush=True)
            tdg = extractor.extract(text, document_id=p.stem,
                                    document_type="legal")
            tdgs[p.stem] = tdg
            # Audit against the raw text, so preprocessing losses surface too.
            missed[p.stem] = audit_recall(raw, tdg)
        except Exception as e:  # noqa: BLE001
            failures.append(f"{p.name}: {e}")
    return tdgs, failures, missed


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


def _tolling_date(args, which: str):
    """Read the tolled-period bound, accepting the superseded ACAS names.

    --acas-a/--acas-b named one UK statute's version of a mechanism the
    engine now implements generically. They still work so existing scripts
    do not break, but they are hidden from help.
    """
    from datetime import date as _d
    new = getattr(args, "tolled_from" if which == "start" else "tolled_to", None)
    old = getattr(args, "acas_a" if which == "start" else "acas_b", None)
    value = new or old
    return _d.fromisoformat(value) if value else None


def _parse_key(s: str) -> tuple[str, str]:
    if ":" not in s:
        raise SystemExit(
            f"error: {s!r} is not a fact reference.\n"
            "       Use DOCUMENT:FACT, for example et1_claim:f4.\n"
            "       The document is the JSON file's name without .json, and "
            "the fact id is the 'id' field on the fact.")
    d, f = s.split(":", 1)
    return d, f


def _parse_date(value: str, what: str):
    """Read an ISO date, or explain precisely what was wrong with it."""
    from datetime import date as _d
    try:
        return _d.fromisoformat(value)
    except ValueError:
        raise SystemExit(
            f"error: {value!r} is not a date ({what}).\n"
            "       Dates are written YYYY-MM-DD, for example 2025-07-12.") from None


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
                tdgs, args.doc, args.entity, _parse_date(args.on, "--on"))
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

    if args.cmd == "ask":
        from tdg_chrono.chronology import build_chronology
        from tdg_chrono import rag

        from tdg_chrono.models import resolve

        answerer = resolve("answer", model=args.model, base_url=args.base_url)
        embedder = _make_embedder(args.embed_model, args.embed_base_url)
        chron = build_chronology(tdgs, embedder=embedder)
        out = rag.ask(chron, tdgs, args.question,
                      model=answerer.model, base_url=answerer.base_url,
                      limit=args.passages, embedder=embedder)
        if args.json:
            print(json.dumps(out, indent=2))
        elif args.show_prompt:
            print(out["prompt"])
        else:
            print(out["answer"])
            print("\n" + "─" * 60)
            if out["unsupported_dates"]:
                print("WARNING  the answer states date(s) that were never "
                      "established from these documents:", file=sys.stderr)
                for d in out["unsupported_dates"]:
                    print(f"           {d}", file=sys.stderr)
                print("         The model either did its own arithmetic or "
                      "slipped. Treat those dates as unverified.",
                      file=sys.stderr)
            print(f"grounded in {len(out['facts_used']['facts'])} established "
                  f"date(s) and {len(out['passages_used'])} passage(s). "
                  "Check the answer against them.")
        return 0

    if args.cmd == "context":
        from datetime import date as _d
        from tdg_chrono.chronology import build_chronology
        from tdg_chrono import rag

        chron = build_chronology(tdgs)
        ctx = rag.select(
            chron, about=args.about or "",
            since=_parse_date(args.since, "--since") if args.since else None,
            until=_parse_date(args.until, "--until") if args.until else None,
            on=_parse_date(args.on, "--on") if args.on else None,
            limit=args.limit)
        if args.json:
            print(json.dumps(rag.to_dict(ctx), indent=2))
        else:
            print(rag.render(ctx, source=str(args.input)))
        return 0

    if args.cmd == "stale":
        out = capabilities.stale_report(tdgs, _parse_key(args.changed))
        if args.json:
            print(json.dumps(out, indent=2))
        else:
            c = out["changed"]
            print(f"If {c['entity']} ({c['value']}) changes, "
                  f"{out['count']} fact(s) stop being trustworthy:")
            if not out["stale"]:
                print("  nothing else depends on it, in this document or any "
                      "other in the bundle.")
            for i in out["stale"]:
                where = "same document" if i["same_document"] else "another document"
                print(f"  [{i['doc_id']}] {i['entity']} = {i['value']}  ({where})")
                print(f"      because: {i['reason']}")
                if i["quote"]:
                    print(f"      \"{i['quote'][:100]}\"")
        return 0

    if args.cmd == "whatif":
        keypart, datepart = args.set.split("=", 1)
        moved = _parse_key(keypart)
        out = capabilities.whatif(tdgs, moved, _parse_date(datepart, "the date in --set"))
        if not args.json and len(out.get("changes", [])) <= 1:
            print(f"shift: {out['shift_days']:+d} days")
            print(f"  nothing is defined relative to {keypart}, so no other "
                  "date moves. This is the answer, not a failure: the "
                  "documents state no rule connecting it to anything else.")
            return 0
        if args.json:
            print(json.dumps(out, indent=2))
        else:
            print(f"shift: {out['shift_days']:+d} days")
            for ch in out["changes"]:
                print(f"  {ch['entity']}: {ch['was']} → {ch['now']}   via {ch['via']}")
            print(f"  ({out['unchanged_note']})")
        return 0

    if args.cmd == "deadline":
        from tdg_core.entailment import check_entailment, use_rulepack_vocabulary
        from tdg_core.trace import render_text, render_line

        rule_path = Path(args.rule)
        pack_aliases = rule_path.parent / "aliases.json"
        if pack_aliases.exists():
            use_rulepack_vocabulary(pack_aliases)
            print(f"loaded pack vocabulary: {pack_aliases}", file=sys.stderr)
        rule_tdg = build_tdg(json.loads(rule_path.read_text()))
        instance = capabilities.merged_instance(tdgs)
        results = check_entailment(
            rule_tdg, instance,
            tolled_from=_tolling_date(args, "start"),
            tolled_to=_tolling_date(args, "end"))
        if not results:
            from tdg_core.provenance import check_relations
            rejected = [c for c in check_relations(rule_tdg) if not c.supported]
            if rejected:
                print("error: this rule pack states a period its own statute "
                      "text does not contain, so no rule was used.",
                      file=sys.stderr)
                for c in rejected:
                    print(f"       {c.summary}", file=sys.stderr)
                print("       Fix the pack, or run 'tdg rulepack validate' on "
                      "it to see the full report.", file=sys.stderr)
            else:
                print("error: no time limit could be read from that rule "
                      "document.\n"
                      "       A rule pack needs an additive dependency "
                      "carrying a period, for example\n"
                      "       'effective date of termination' + 3 months -> "
                      "'presentation of the complaint'.\n"
                      "       Run 'tdg rulepack validate <pack folder>' for a "
                      "full diagnosis.", file=sys.stderr)
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
    b.add_argument("--clean", action="store_true",
                   help="run the aggressive paragraph cleaner over the "
                        "documents first (strips headers, footers and "
                        "two-column artifacts). Off by default: it is tuned "
                        "for messy PDF corpora and discards body text on "
                        "ordinary correspondence. Anything it removes is "
                        "listed in the run output.")
    b.add_argument("--formats", default=",".join(DEFAULT_FORMATS),
                   help=f"comma list of {sorted(EXPORTERS)} (default: {','.join(DEFAULT_FORMATS)})")
    b.add_argument("--corrections", type=Path, default=None,
                   help="corrections.json (accept/reject/edit/merge/split), "
                        "re-applied on every run; see tdg-chrono correct")
    b.add_argument("--overrides", type=Path, default=None,
                   help=argparse.SUPPRESS)  # legacy force_merge/split file
    b.add_argument("--merge-threshold", type=float, default=0.45)
    b.add_argument("--linking", choices=["composed", "gated"], default="composed",
                   help="how facts are matched across documents. 'composed' "
                        "weighs entity name, quote overlap and temporal "
                        "proximity together, scaling the bar by how related "
                        "the two documents are. 'gated' is the older "
                        "behaviour, where name and quote overlap must each "
                        "clear a fixed floor, so one weak signal, usually "
                        "the quote, vetoes an otherwise sound match.")
    b.add_argument("--embed-model", default=None,
                   help="embedding model for entity-name similarity, e.g. "
                        "nomic-embed-text. Lets paraphrases link that no "
                        "lexical measure can reach ('termination' vs "
                        "'dismissal'). Off unless set. Env: TDG_EMBED_MODEL")
    b.add_argument("--embed-base-url", default=None,
                   help="OpenAI-compatible embeddings endpoint, e.g. "
                        "http://localhost:11434/v1 for Ollama. Any provider "
                        "works. Env: TDG_EMBED_BASE_URL")
    b.add_argument("--matter-field", default="matter",
                   help="which key in each TDG carries the matter identifier "
                        "(default: matter). Documents declaring different "
                        "matters are never linked; when the key is absent "
                        "linking proceeds and the run says so.")

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
    dl.add_argument("--tolled-from", metavar="DATE",
                    help="start of a period the statute does not count "
                         "against its limit (ISO date). What qualifies is "
                         "declared by the rule pack.")
    dl.add_argument("--tolled-to", metavar="DATE",
                    help="end of that period (ISO date)")
    dl.add_argument("--acas-a", metavar="DATE", help=argparse.SUPPRESS)
    dl.add_argument("--acas-b", metavar="DATE", help=argparse.SUPPRESS)

    ask = sub.add_parser(
        "ask",
        help="ask a question about the bundle; dates come from the engine, "
             "the rest from the documents")
    _tdg_input(ask)
    ask.add_argument("question", help="a plain-English question")
    ask.add_argument("--model",
                     help="the model that writes the answer, e.g. llama3. "
                          "Env: TDG_ANSWER_MODEL, then TDG_LLM_MODEL")
    ask.add_argument("--base-url",
                     help="where that model is, e.g. "
                          "http://localhost:11434/v1. Env: "
                          "TDG_ANSWER_BASE_URL, then OPENAI_BASE_URL")
    ask.add_argument("--passages", type=int, default=8,
                     help="how many document sentences to retrieve (default 8)")
    ask.add_argument("--embed-model", help="optional embedder for retrieval")
    ask.add_argument("--embed-base-url")
    ask.add_argument("--show-prompt", action="store_true",
                     help="print what the model was given instead of the answer")
    ask.add_argument("--json", action="store_true",
                     help="answer plus every fact and passage it rested on")

    ctx = sub.add_parser(
        "context",
        help="grounded temporal facts for a prompt, with quotes and gaps "
             "stated (for retrieval-augmented generation)")
    _tdg_input(ctx)
    ctx.add_argument("--about", metavar="TERMS",
                     help="only facts matching these words, e.g. termination")
    ctx.add_argument("--since", metavar="DATE")
    ctx.add_argument("--until", metavar="DATE")
    ctx.add_argument("--on", metavar="DATE")
    ctx.add_argument("--limit", type=int)
    ctx.add_argument("--json", action="store_true",
                     help="machine-readable, with character offsets")

    stale = sub.add_parser(
        "stale",
        help="what stops being trustworthy if one date turns out to be wrong")
    _tdg_input(stale)
    stale.add_argument("--changed", required=True, metavar="DOC:FACT",
                       help="the fact whose date is in doubt")
    stale.add_argument("--json", action="store_true")

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
    if args.cmd in ("interval", "contradictions", "whatif", "deadline",
                    "stale", "context", "ask"):
        return _cmd_capability(args)

    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    unknown = [f for f in formats if f not in EXPORTERS]
    if unknown:
        parser.error(f"unknown format(s) {unknown}; choose from {sorted(EXPORTERS)}")

    if not args.input.is_dir():
        parser.error(f"{args.input} is not a directory")

    missed: dict[str, list] = {}
    if args.from_tdgs:
        tdgs, failures = _load_tdg_bundle(args.input, args.matter_field)
        extractor_label = "none (pre-extracted TDGs)"
    else:
        from tdg_chrono.models import resolve, summarise

        extractor_cfg = resolve("extract", model=args.model,
                                base_url=args.base_url)
        embed_cfg = resolve("embed", model=args.embed_model,
                            base_url=args.embed_base_url)
        print(f"models: {summarise([extractor_cfg, embed_cfg])}",
              file=sys.stderr)
        tdgs, failures, missed = _extract_bundle(
            args.input, args.extractor, clean=args.clean,
            model=extractor_cfg.model, base_url=extractor_cfg.base_url)
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
        print(f"error: no usable documents in {args.input}", file=sys.stderr)
        if failures:
            print("       every file there failed to load:", file=sys.stderr)
            for f in failures:
                print(f"         {f}", file=sys.stderr)
        elif args.from_tdgs:
            print("       --from-tdgs expects a folder of TDG *.json files. "
                  "That folder has none.\n"
                  "       If it holds raw documents instead, drop --from-tdgs "
                  "and pass an extractor.", file=sys.stderr)
        else:
            print("       expected .txt, .md, .pdf or .docx documents in "
                  "that folder.\n"
                  "       If it holds already-extracted TDG *.json, add "
                  "--from-tdgs.", file=sys.stderr)
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

    embedder = _make_embedder(args.embed_model, args.embed_base_url)
    chron = build_chronology(
        outcome.tdgs,
        overrides=outcome.overrides,
        merge_threshold=args.merge_threshold,
        composed_linking=(args.linking == "composed"),
        embedder=embedder,
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
        print(f"\nWARNING  {len(failures)} document(s) could not be read and "
              "are missing from this timeline:", file=sys.stderr)
        for f in failures:
            print(f"         {f}", file=sys.stderr)
        print("         The chronology below was built from the rest. Fix or "
              "remove those files and run again for a complete one.",
              file=sys.stderr)

    if missed:
        from tdg_chrono.recall import format_report
        for line in format_report(missed):
            print(line)

    bad_relations = chron.meta.get("relations", [])
    if bad_relations:
        print(f"\nrelation audit: {len(bad_relations)} link(s) added by the "
              "extractor, not found in the document.")
        print("  It measured the gap between two dates and recorded it as a "
              "rule. Ignored in all calculations.")
        for r in bad_relations:
            print(f"  - {r['summary']}")

    unverified = chron.meta.get("counts", {}).get("unverified_quotes", 0)
    if unverified:
        prov = chron.meta.get("provenance", {})
        print(f"\nprovenance: {unverified} quote(s) could not be checked "
              "against the text of the document they came from. They are "
              "still shown, but they do not corroborate any other document.")
        for doc_id, rep in sorted(prov.items()):
            if not rep["quotes_unverified"]:
                continue
            why = ("no source text shipped" if not rep["has_source_text"]
                   else "quote not found in the text")
            print(f"  - {doc_id}: {len(rep['quotes_unverified'])} of "
                  f"{rep['facts']} ({why})")

    if args.linking == "composed":
        _report_matter_separation(outcome.tdgs, args.matter_field)
    return 0


def _report_matter_separation(tdgs, matter_field: str) -> None:
    """State what kept matters apart, or that nothing did.

    A run that linked everything because it had no way to tell matters apart
    must say so. Silence here would read as a guarantee the run cannot make.
    """
    total = len(tdgs)
    declared = sum(1 for t in tdgs.values() if t.matter is not None)
    named = sum(1 for t in tdgs.values() if t.parties)
    distinct = {t.matter for t in tdgs.values() if t.matter is not None}

    if declared == total and total:
        print(f"\nlinking: '{matter_field}' declared on all {total} documents "
              f"({len(distinct)} distinct). Documents in different matters "
              "were not linked.")
    elif declared:
        print(f"\nlinking: '{matter_field}' declared on {declared}/{total} "
              f"documents ({len(distinct)} distinct). Documents without it "
              "fall back to party names, then link freely.")
    elif named == total and total:
        print(f"\nlinking: no '{matter_field}' declared; parties named on all "
              f"{total} documents. Documents sharing no party were not linked.")
    elif named:
        print(f"\nlinking: no '{matter_field}' declared; parties named on "
              f"{named}/{total} documents. The remaining {total - named} link "
              "freely, because nothing distinguishes their matter.")
    else:
        print(f"\nWARNING  no '{matter_field}' and no parties on any "
              "document.", file=sys.stderr)
        print("         Nothing distinguishes one case from another here, so "
              "every document was linked to every other. If this folder holds "
              "more than one case, rows from different cases can be merged "
              "into one and reported as a disagreement between them.",
              file=sys.stderr)
        print("         Fine for a single-case bundle. Otherwise declare a "
              f"'{matter_field}' on each document, or use an extractor that "
              "names the parties.", file=sys.stderr)


def _explain(exc: Exception) -> str:
    """Turn an internal failure into something a user can act on.

    Four ordinary mistakes used to end in a Python traceback: naming a
    document that is not in the bundle, naming a fact that is not in the
    document, mistyping a date, and pointing at a rule file that does not
    exist. A traceback tells the reader nothing about which of those they
    did.
    """
    import json as _json

    if isinstance(exc, FileNotFoundError):
        return (f"error: no such file: {exc.filename}\n"
                "       Check the path. For --rule this should be a "
                "statute.tdg.json inside a rule pack folder.")
    if isinstance(exc, _json.JSONDecodeError):
        return (f"error: {exc.doc[:0] or 'that file'} is not valid JSON "
                f"(line {exc.lineno}, column {exc.colno}): {exc.msg}")
    if isinstance(exc, KeyError):
        # _find already builds a helpful message; unwrap it from the KeyError
        return f"error: {exc.args[0] if exc.args else exc}"
    if isinstance(exc, ValueError):
        return f"error: {exc}"
    if isinstance(exc, PermissionError):
        return f"error: not allowed to read {exc.filename}"
    return f"error: {type(exc).__name__}: {exc}"


def _entry() -> int:
    import json as _json
    try:
        return main()
    except SystemExit as e:
        # A message-carrying SystemExit is a user error we already phrased.
        if isinstance(e.code, str):
            print(e.code, file=sys.stderr)
            return 1
        raise
    except (FileNotFoundError, PermissionError, _json.JSONDecodeError,
            KeyError, ValueError) as e:
        print(_explain(e), file=sys.stderr)
        return 2
    except BrokenPipeError:
        import os
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0


if __name__ == "__main__":
    sys.exit(_entry())
