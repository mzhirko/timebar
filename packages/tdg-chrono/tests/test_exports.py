"""Exports smoke tests: every format writes, footer present, disputes marked."""

from datetime import date
from pathlib import Path

from tdg_chrono.chronology import (Chronology, ChronologyEvent, SourceRef,
                                   DisputedValue)
from tdg_chrono.exports import EXPORTERS


def _chron():
    src = SourceRef("et1", "f1", "quote", "2025-07-14",
                    date(2025, 7, 14), 0, 0, 0.9)
    src2 = SourceRef("letter", "f1", "other quote", "2025-07-12",
                     date(2025, 7, 12), 0, 0, 0.9)
    ev = ChronologyEvent(
        "ev001", date(2025, 7, 12), "day", "termination (end)",
        [src, src2], "disputed",
        disputed_values=[DisputedValue("2025-07-12", date(2025, 7, 12), ["letter"]),
                         DisputedValue("2025-07-14", date(2025, 7, 14), ["et1"])])
    return Chronology(events=[ev], unplaced=[],
                      meta={"extractor": "test", "timestamp": "t",
                            "counts": {"events": 1, "disputed": 1, "unplaced": 0}})


def test_all_formats_write(tmp_path: Path):
    for fmt, fn in EXPORTERS.items():
        p = fn(_chron(), tmp_path / f"chronology.{fmt}")
        assert p.exists() and p.stat().st_size > 0


def test_csv_footer_and_dispute(tmp_path: Path):
    p = EXPORTERS["csv"](_chron(), tmp_path / "c.csv")
    text = p.read_text()
    assert "tdg-chrono" in text and "disputed" in text
    assert "2025-07-12" in text and "2025-07-14" in text
