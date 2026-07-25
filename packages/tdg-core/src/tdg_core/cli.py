"""The `tdg` CLI: format validation and deadline checking with --explain."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from tdg_core.io import build_tdg
from tdg_core.validate import validate_path


def _load(path: str):
    return build_tdg(json.loads(Path(path).read_text()))


def _cmd_validate(args) -> int:
    failed = 0
    for path in args.paths:
        for file, errors in validate_path(path).items():
            if errors:
                failed += 1
                print(f"FAIL {file}")
                for e in errors:
                    print(f"  - {e}")
            elif not args.quiet:
                print(f"ok   {file}")
    return 1 if failed else 0


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


def _cmd_check(args) -> int:
    from tdg_core.entailment import check_entailment
    from tdg_core.trace import render_text, render_html, render_line

    pack_aliases = Path(args.rule).parent / "aliases.json"
    if pack_aliases.exists():
        from tdg_core.entailment import use_rulepack_vocabulary
        use_rulepack_vocabulary(pack_aliases)
        print(f"loaded pack vocabulary: {pack_aliases}", file=sys.stderr)

    rule_tdg = _load(args.rule)
    instance_tdg = _load(args.instance)
    results = check_entailment(
        rule_tdg, instance_tdg,
        tolled_from=_tolling_date(args, "start"),
        tolled_to=_tolling_date(args, "end"),
    )
    if not results:
        print("No temporal rules discovered in the rule document "
              "(no additive dependency with a readable period).", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
    else:
        for i, r in enumerate(results):
            if i:
                print("\n" + "─" * 60 + "\n")
            print(render_text(r) if args.explain else render_line(r))
    if args.html:
        Path(args.html).write_text(render_html(results))
        print(f"\nwrote {args.html}", file=sys.stderr)
    return 0 if all(r.verdict != "INDETERMINATE" for r in results) else 2


def _cmd_rulepack(args) -> int:
    """Validate a rule pack: schema-valid statute, rules discoverable with
    zero engine changes, gold cases match expected.json."""
    from tdg_core.entailment import (check_entailment, find_rules,
                                     use_rulepack_vocabulary)
    from tdg_core.validate import validate_tdg_dict

    pack = Path(args.path)
    statute_path = pack / "statute.tdg.json"
    ok = True

    def fail(msg):
        nonlocal ok
        ok = False
        print(f"  FAIL: {msg}")

    print(f"Rule pack: {pack}")
    if not statute_path.exists():
        fail("statute.tdg.json missing")
        return 1

    data = json.loads(statute_path.read_text())
    errors = validate_tdg_dict(data)
    if errors:
        for e in errors:
            fail(f"statute schema: {e}")
    else:
        print("  ok: statute.tdg.json is schema-valid")

    aliases = pack / "aliases.json"
    if aliases.exists():
        use_rulepack_vocabulary(aliases)
        print("  ok: aliases.json loaded (extends default vocabulary)")

    statute = build_tdg(data)
    rules = find_rules(statute)
    if not rules:
        fail("no temporal rule discoverable from the statute TDG")
    else:
        for r in rules:
            src = r.offset.inclusivity_source
            print(f"  ok: rule discovered: {r.description}  "
                  f"[anchor-day counting: {src}]")

    expected_path = pack / "expected.json"
    gold_dir = pack / "gold_cases"
    if expected_path.exists() and gold_dir.is_dir():
        expected = json.loads(expected_path.read_text())
        for name, exp in sorted(expected.items()):
            case_path = gold_dir / name
            if not case_path.exists():
                fail(f"gold case listed in expected.json but missing: {name}")
                continue
            case = _load(case_path)
            results = check_entailment(statute, case)
            if not results:
                fail(f"{name}: engine produced no result")
                continue
            r = results[0]
            if r.verdict != exp["verdict"]:
                fail(f"{name}: verdict {r.verdict} != expected {exp['verdict']}")
            elif exp.get("deadline") not in (None, r.deadline_computed):
                fail(f"{name}: deadline {r.deadline_computed} != expected {exp['deadline']}")
            else:
                print(f"  ok: gold case {name}: {r.verdict}"
                      + (f" (deadline {r.deadline_computed})" if r.deadline_computed else ""))
    else:
        print("  - no gold_cases/expected.json (allowed, but packs should ship them)")

    print("PASS — pack usable with zero engine changes" if ok else "FAIL")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tdg", description="TDG format and engine tools")
    sub = parser.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="validate TDG JSON against the v1 schema")
    v.add_argument("paths", nargs="+")
    v.add_argument("-q", "--quiet", action="store_true")
    v.set_defaults(fn=_cmd_validate)

    c = sub.add_parser("check", help="check an instance against a rule document's time limits")
    c.add_argument("--rule", required=True, help="rule/statute TDG JSON")
    c.add_argument("--instance", required=True, help="instance/case TDG JSON")
    c.add_argument("--explain", action="store_true",
                   help="full derivation: statute sentence, discovered vs assumed "
                        "counting, anchor match, passed-over candidates, arithmetic")
    c.add_argument("--json", action="store_true", help="machine-readable output (includes verdict)")
    c.add_argument("--html", metavar="PATH", help="also write an HTML trace")
    c.add_argument("--tolled-from", metavar="DATE",
                   help="start of a period the statute does not count "
                        "against its limit (ISO date). What qualifies is "
                        "declared by the rule pack.")
    c.add_argument("--tolled-to", metavar="DATE",
                   help="end of that period (ISO date)")
    c.add_argument("--acas-a", metavar="DATE", help=argparse.SUPPRESS)
    c.add_argument("--acas-b", metavar="DATE", help=argparse.SUPPRESS)
    c.set_defaults(fn=_cmd_check)

    rp = sub.add_parser("rulepack", help="rule pack tools")
    rpsub = rp.add_subparsers(dest="rpcmd", required=True)
    rpv = rpsub.add_parser("validate", help="validate a rule pack directory")
    rpv.add_argument("path")
    rpv.set_defaults(fn=_cmd_rulepack)

    args = parser.parse_args(argv)
    return args.fn(args)


def _entry() -> int:
    try:
        return main()
    except BrokenPipeError:
        import os
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0


if __name__ == "__main__":
    sys.exit(_entry())
