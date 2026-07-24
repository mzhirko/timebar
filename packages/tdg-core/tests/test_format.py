"""Format tests: schema validation, round-trip, provenance offsets, version gate."""

import json
from datetime import date

import pytest

from tdg_core.io import build_tdg
from tdg_core.tdg import (SCHEMA_VERSION, TemporalDependencyGraph,
                          TemporalFact, TemporalDependency, TimexSpan)
from tdg_core.validate import validate_tdg_dict


def _tdg():
    return TemporalDependencyGraph(
        document_id="doc1", document_type="legal",
        source_text="Terminated on 12 July 2025. Claim within 3 months.",
        facts=[TemporalFact(
            id="f1", entity="termination", role="END",
            timex=TimexSpan(text="12 July 2025", timex_type="DATE",
                            value="2025-07-12", start_char=14, end_char=26,
                            date_parsed=date(2025, 7, 12)),
            sentence="Terminated on 12 July 2025.")],
        dependencies=[TemporalDependency(
            from_id="f1", to_id="f1", constraint_type="ordering",
            constraint_expr="start < end")])


def test_emitted_tdg_validates():
    assert validate_tdg_dict(_tdg().to_dict()) == []


def test_schema_version_emitted():
    assert _tdg().to_dict()["schema_version"] == SCHEMA_VERSION


def test_provenance_offsets_roundtrip():
    d = _tdg().to_dict()
    reloaded = build_tdg(json.loads(json.dumps(d)))
    f = reloaded.facts[0]
    assert (f.timex.start_char, f.timex.end_char) == (14, 26)
    assert f.timex.date_parsed == date(2025, 7, 12)


def test_unknown_major_rejected():
    d = _tdg().to_dict()
    d["schema_version"] = "2.0"
    with pytest.raises(ValueError, match="schema_version"):
        build_tdg(d)


def test_dangling_dependency_flagged():
    d = _tdg().to_dict()
    d["dependencies"][0]["to_id"] = "ghost"
    errs = validate_tdg_dict(d)
    assert any("ghost" in e for e in errs)
