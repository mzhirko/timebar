"""Isolation guards: no jurisdiction knowledge in engine code OR engine
data; rule packs usable with zero engine changes. These make the
'no hardcoded parts' claim checkable rather than asserted."""

from pathlib import Path

import pytest

CORE = Path(__file__).parent.parent / "src" / "tdg_core"
RULEPACKS = Path(__file__).parents[3] / "rulepacks"
PACK = RULEPACKS / "uk" / "era-1996-s111"

# Vocabulary that must live in rule packs, never in the engine (source OR data).
FORBIDDEN = ["dismissal", "edt", "et1", "claim form", "harassment",
             "discriminatory act", "termination date", "acas"]

# Statutes and bodies that must not be named in engine code at all, in any
# form. Quoted-string scanning alone missed a UK conciliation mechanism that
# lived in the engine for months as a function name, parameters and comments,
# so these are matched anywhere in the file.
FORBIDDEN_ANYWHERE = ["acas", "employment rights act", "era 1996",
                      "equality act", "limitation act", "s.207b", "s207b"]


def _scan(path: Path) -> list[str]:
    text = path.read_text().lower()
    return [w for w in FORBIDDEN if f'"{w}"' in text]


def _scan_anywhere(path: Path) -> list[str]:
    text = path.read_text().lower()
    return [w for w in FORBIDDEN_ANYWHERE if w in text]


def test_no_jurisdiction_vocabulary_in_engine_source():
    leaked = _scan(CORE / "entailment.py")
    assert not leaked, f"jurisdiction vocabulary hardcoded in engine: {leaked}"


def test_no_statute_named_anywhere_in_engine_source():
    """Not only in strings: identifiers, comments and docstrings too.

    A statute named in a parameter or a helper's name is still the engine
    knowing about one jurisdiction, and it is how the UK conciliation rule
    escaped the string-only scan.
    """
    # cli.py is excluded on purpose: it is the command-line surface and
    # still accepts the superseded --acas-a/--acas-b flag names so existing
    # scripts keep working. The engine itself must stay neutral.
    offenders = {}
    for f in sorted(CORE.glob("*.py")):
        if f.name == "cli.py":
            continue
        leaked = _scan_anywhere(f)
        if leaked:
            offenders[f.name] = leaked
    assert not offenders, f"statute named in engine code: {offenders}"


def test_tolling_shape_is_generic_and_pack_declared():
    """The engine holds the mechanism; the pack supplies the specifics."""
    import json

    from tdg_core.entailment import (TollingRule, active_tolling,
                                     load_alias_file, register_tolling)

    register_tolling(None)
    assert active_tolling() is None, "engine must ship with no tolling rule"

    load_alias_file(PACK / "aliases.json")
    rule = active_tolling()
    try:
        assert rule is not None, "pack should declare its own tolling rule"
        assert rule.authority, "a declared rule should cite its authority"
        declared = json.loads((PACK / "aliases.json").read_text())["tolling"]
        assert rule.label == declared["label"]
        assert rule.floor_after_end == declared["floor_after_end"]
    finally:
        # load_alias_file registers the pack's concept aliases globally;
        # leaving them installed would leak into other tests.
        from tdg_core.entailment import _load_default_aliases, load_alias_dict
        load_alias_dict({}, replace=True, source="<test-reset>")
        _load_default_aliases()
        register_tolling(None)


def test_no_jurisdiction_vocabulary_in_engine_data():
    for f in (CORE / "data").glob("*.json"):
        leaked = _scan(f)
        assert not leaked, f"jurisdiction vocabulary in engine data {f.name}: {leaked}"


def test_engine_ships_with_no_concept_aliases():
    import tdg_core.entailment as e
    # language-layer cues load; jurisdiction concept aliases do NOT
    assert e._ACTION_CUES, "generic English cues should load"
    packish = {"effective date of termination", "date of the act"}
    assert not packish & set(e._ANCHOR_ALIASES), (
        "engine default vocabulary contains rule-pack concepts")


def test_rulepack_vocabulary_is_additive_and_tracked():
    import tdg_core.entailment as e
    before = dict(e._ANCHOR_ALIASES)
    n_sources = len(e._ALIAS_SOURCES)
    e.load_alias_file(PACK / "aliases.json")
    try:
        assert "effective date of termination" in e._ANCHOR_ALIASES
        assert len(e._ALIAS_SOURCES) == n_sources + 1
        assert str(PACK / "aliases.json") in e._ALIAS_SOURCES[-1]
    finally:
        e._ANCHOR_ALIASES.clear()
        e._ANCHOR_ALIASES.update(before)
        e._ALIAS_SOURCES.pop()


@pytest.mark.skipif(not RULEPACKS.exists(), reason="rulepacks dir not present")
def test_every_shipped_rulepack_validates_with_zero_engine_changes():
    from tdg_core.cli import main
    packs = sorted(p.parent for p in RULEPACKS.rglob("statute.tdg.json"))
    assert len(packs) >= 2, "expected at least the two shipped packs"
    for pack in packs:
        assert main(["rulepack", "validate", str(pack)]) == 0, pack


# ── one pack at a time: vocabulary must not accumulate ──────────────────

LA1980 = RULEPACKS / "uk" / "limitation-act-1980-s5"


def test_a_second_pack_replaces_the_first():
    """Pack vocabulary is global; checking two statutes must not blend them.

    Left additive, a tolling rule declared by one statute stayed installed
    while a statute granting no such extension was checked, which silently
    moves a deadline.
    """
    import tdg_core.entailment as e
    from tdg_core.embeddings import _ENTITY_BOILERPLATE

    try:
        e.use_rulepack_vocabulary(PACK / "aliases.json")
        assert "effective date of termination" in e._ANCHOR_ALIASES
        assert e.active_tolling() is not None
        assert _ENTITY_BOILERPLATE

        e.use_rulepack_vocabulary(LA1980 / "aliases.json")
        assert "effective date of termination" not in e._ANCHOR_ALIASES, (
            "the first pack's concepts survived into the second")
        assert e.active_tolling() is None, (
            "the first pack's tolling rule survived into the second")
        assert not _ENTITY_BOILERPLATE, (
            "the first pack's boilerplate survived into the second")
    finally:
        e.reset_vocabulary()


def test_language_defaults_survive_a_pack_swap():
    """Replacing a pack must not strip the generic English cues with it."""
    import tdg_core.entailment as e

    try:
        baseline = len(e._ACTION_CUES)
        assert baseline, "language cues should be loaded at import"
        e.use_rulepack_vocabulary(PACK / "aliases.json")
        e.use_rulepack_vocabulary(LA1980 / "aliases.json")
        assert len(e._ACTION_CUES) >= baseline
    finally:
        e.reset_vocabulary()


def test_reset_returns_to_a_freshly_started_engine():
    import tdg_core.entailment as e

    e.use_rulepack_vocabulary(PACK / "aliases.json")
    e.reset_vocabulary()
    assert not e._ANCHOR_ALIASES
    assert e.active_tolling() is None
    assert e._ACTION_CUES, "language layer must come back"


def test_composing_packs_deliberately_is_still_possible():
    """Additive loading remains available for callers that want it."""
    import tdg_core.entailment as e

    try:
        e.use_rulepack_vocabulary(PACK / "aliases.json")
        e.load_alias_file(LA1980 / "aliases.json")
        assert "effective date of termination" in e._ANCHOR_ALIASES
        assert "date on which the cause of action accrued" in e._ANCHOR_ALIASES
    finally:
        e.reset_vocabulary()
