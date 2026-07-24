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


def _scan(path: Path) -> list[str]:
    text = path.read_text().lower()
    return [w for w in FORBIDDEN if f'"{w}"' in text]


def test_no_jurisdiction_vocabulary_in_engine_source():
    leaked = _scan(CORE / "entailment.py")
    assert not leaked, f"jurisdiction vocabulary hardcoded in engine: {leaked}"


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
