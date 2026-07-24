"""TDG format validation against the packaged JSON Schema.

Library use:
    from tdg_core.validate import validate_tdg_dict, validate_path
    errors = validate_tdg_dict(data)          # [] means valid

CLI:
    tdg validate file.json
    tdg validate results_dir/                 # every *.json in the dir
    (non-zero exit on any failure)
"""

from __future__ import annotations

import json
import sys
from importlib import resources
from pathlib import Path

import jsonschema

_SCHEMA = None


def _schema() -> dict:
    global _SCHEMA
    if _SCHEMA is None:
        with resources.files("tdg_core.schema").joinpath("tdg-v1.json").open() as f:
            _SCHEMA = json.load(f)
    return _SCHEMA


def validate_tdg_dict(data: dict) -> list[str]:
    """Validate a TDG dict. Returns a list of error strings; empty = valid.

    Beyond the schema, checks referential integrity: every dependency
    endpoint and every ``is_duplicate_of`` must name an existing fact id.
    """
    validator = jsonschema.Draft202012Validator(_schema())
    errors = [
        f"{'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}"
        for e in validator.iter_errors(data)
    ]
    fact_ids = {f.get("id") for f in data.get("facts", []) if isinstance(f, dict)}
    for i, dep in enumerate(data.get("dependencies", [])):
        if not isinstance(dep, dict):
            continue
        for end in ("from_id", "to_id"):
            fid = dep.get(end)
            if fid is not None and fid not in fact_ids:
                errors.append(f"dependencies/{i}/{end}: no fact with id {fid!r}")
    for f in data.get("facts", []):
        if isinstance(f, dict) and f.get("is_duplicate_of") and f["is_duplicate_of"] not in fact_ids:
            errors.append(f"facts/{f.get('id')}: is_duplicate_of names missing fact {f['is_duplicate_of']!r}")
    return errors


def validate_path(path: str | Path) -> dict[str, list[str]]:
    """Validate one file or every ``*.json`` in a directory.

    Returns {path: [errors]}. A JSON parse failure is reported as one error.
    """
    p = Path(path)
    targets = sorted(p.glob("*.json")) if p.is_dir() else [p]
    report: dict[str, list[str]] = {}
    for t in targets:
        try:
            data = json.loads(t.read_text())
        except (OSError, json.JSONDecodeError) as e:
            report[str(t)] = [f"unreadable: {e}"]
            continue
        report[str(t)] = validate_tdg_dict(data)
    return report


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="tdg", description="TDG format tools")
    sub = parser.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate", help="Validate TDG JSON files against the v1 schema")
    v.add_argument("paths", nargs="+", help="Files or directories of *.json")
    v.add_argument("-q", "--quiet", action="store_true", help="Only print failures")
    args = parser.parse_args(argv)

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


if __name__ == "__main__":
    sys.exit(main())
