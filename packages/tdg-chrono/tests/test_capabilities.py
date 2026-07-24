"""Capability command tests (Phase 1.7)."""

from datetime import date

import pytest

from tdg_chrono import capabilities as cap
from test_chronology import bundle  # noqa: F401


def test_interval_between(bundle):
    out = cap.interval_between(bundle, ("dismissal_letter", "f1"), ("et1", "f3"))
    assert out["relation"] == "BEFORE"
    assert "2025-07-12" in out["derivation"]


def test_interval_contains_yes_no_and_open_end(bundle):
    # employment started 2019-03-03 (letter f2); the letter also has the END (f1)
    yes = cap.interval_contains(bundle, "dismissal_letter", "employment", date(2025, 6, 1))
    assert yes["answer"].startswith("YES")
    # query must match the facts' wording; "employment" only matches the
    # START fact, so the interval is honestly open-ended:
    open_end = cap.interval_contains(bundle, "dismissal_letter", "employment", date(2026, 1, 1))
    assert open_end["answer"] == "YES (no end found)"
    # matching the END fact's wording closes it:
    no = cap.interval_contains(bundle, "dismissal_letter", "termination", date(2026, 1, 1))
    assert no["answer"] == "NO" and "after end" in no["derivation"]
    unknown = cap.interval_contains(bundle, "et1", "employment", date(2025, 6, 1))
    assert unknown["answer"] == "UNKNOWN" and "request" in unknown


def test_whatif_cascade_and_no_mutation(bundle):
    before = bundle["et1"].to_json()
    out = cap.whatif(bundle, ("et1", "f5"), date(2025, 10, 13))
    assert out["shift_days"] == 7
    derived = [c for c in out["changes"] if c["entity"] == "response deadline"]
    assert derived and derived[0]["now"] == "2025-11-10"
    assert bundle["et1"].to_json() == before   # sources untouched


def test_whatif_requires_dated_root(bundle):
    with pytest.raises(ValueError, match="no resolved date"):
        cap.whatif(bundle, ("et1", "f4"), date(2025, 1, 1))


def test_merged_instance_prefixes_ids(bundle):
    inst = cap.merged_instance(bundle)
    ids = {f.id for f in inst.facts}
    assert "et1:f3" in ids and "dismissal_letter:f1" in ids
    assert len(ids) == sum(len(t.facts) for t in bundle.values())


def test_contradiction_report_shape(bundle):
    out = cap.contradiction_report(bundle)
    assert set(out) == {"documents", "contradictions", "count"}
    for c in out["contradictions"]:
        assert c["a"]["quote"] and c["b"]["quote"]
