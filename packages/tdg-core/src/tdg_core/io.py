"""
Unified TDG I/O: loading, saving, and shared constants.

Replaces the 4 duplicate loaders that existed across run_cross_doc.py,
test_cross_doc.py, evaluate_use_case_a.py, and evaluate_use_cases_bc.py.

Usage:
    from tdg_core.io import load_tdg, load_tdg_dir, build_tdg, DEFAULT_SIMILARITY_THRESHOLD

    # Load a single TDG from a JSON file
    tdg = load_tdg("../data/results_contracts/en_contracts_seed0.json")

    # Load all TDGs from a directory
    tdgs = load_tdg_dir("../data/results_contracts/")

    # Build a TDG from a dict (e.g. ground truth JSON entry)
    tdg = build_tdg(data_dict)
"""

from __future__ import annotations

import glob
import json
import os
from datetime import date
from typing import Optional

from tdg_core.tdg import (
    SCHEMA_VERSION,
    TemporalDependencyGraph,
    TemporalFact,
    TemporalDependency,
    TimexSpan,
)

# ─── Shared constants ─────────────────────────────────────────────────────

DEFAULT_SIMILARITY_THRESHOLD = 0.4


# ─── TDG construction from dict ──────────────────────────────────────────

def build_tdg(
    data: dict,
    document_id: Optional[str] = None,
    document_type: Optional[str] = None,
    max_source_text: int = 0,
    matter_field: str = "matter",
) -> TemporalDependencyGraph:
    """Build a TemporalDependencyGraph from a JSON-like dict.

    Works with both LLM extraction output and ground truth annotations.
    All optional fields have safe defaults.

    Args:
        data: dict with "facts", "dependencies", and optional metadata
        document_id: override for data["document_id"]
        document_type: override for data["document_type"]
        max_source_text: truncate source_text to this many chars (0 = keep all)
        matter_field: which key carries the matter identifier. Configurable
            because what identifies a matter differs by practice — a case
            number, a claimant, an internal file reference — and none of them
            should be assumed.
    """
    ver = str(data.get("schema_version", SCHEMA_VERSION))
    if ver.split(".")[0] != SCHEMA_VERSION.split(".")[0]:
        raise ValueError(
            f"Unsupported TDG schema_version {ver!r}; "
            f"this build reads major version {SCHEMA_VERSION.split('.')[0]}"
        )

    facts = []
    for fd in data.get("facts", []):
        dp = None
        if fd.get("date_parsed"):
            dp = date.fromisoformat(fd["date_parsed"])

        facts.append(TemporalFact(
            id=fd["id"],
            entity=fd["entity"],
            role=fd["role"],
            timex=TimexSpan(
                text=fd.get("raw_text", ""),
                timex_type=fd.get("timex_type", "DATE"),
                value=fd.get("value"),
                start_char=fd.get("start_char", 0),
                end_char=fd.get("end_char", 0),
                date_parsed=dp,
                duration_days=fd.get("duration_days"),
            ),
            sentence=fd.get("sentence", ""),
            confidence=fd.get("confidence", 0.9),
            temporal_content=fd.get("temporal_content", True),
        ))

    deps = []
    for dd in data.get("dependencies", []):
        deps.append(TemporalDependency(
            from_id=dd["from_id"],
            to_id=dd["to_id"],
            constraint_type=dd["constraint_type"],
            constraint_expr=dd.get("constraint_expr", ""),
            delta_days=dd.get("delta_days"),
            verified=dd.get("verified", False),
            confidence=dd.get("confidence", 0.85),
            allen_relation=dd.get("allen_relation"),
            corroboration_count=dd.get("corroboration_count", 1),
        ))

    doc_id = document_id or data.get("document_id", "unknown")
    doc_type = document_type or data.get("document_type", "legal")
    source_text = data.get("source_text", "")
    if max_source_text > 0:
        source_text = source_text[:max_source_text]

    matter = data.get(matter_field)
    raw_parties = data.get("parties") or []
    if isinstance(raw_parties, str):
        raw_parties = [raw_parties]
    parties = [str(p).strip() for p in raw_parties if str(p).strip()]
    return TemporalDependencyGraph(
        document_id=doc_id,
        document_type=doc_type,
        source_text=source_text,
        facts=facts,
        dependencies=deps,
        matter=str(matter) if matter is not None else None,
        parties=parties,
    )


# ─── File loaders ─────────────────────────────────────────────────────────

def load_tdg(
    path: str,
    document_id: Optional[str] = None,
    document_type: Optional[str] = None,
    max_source_text: int = 0,
) -> TemporalDependencyGraph:
    """Load a TDG from a JSON file.

    Args:
        path: path to JSON file (output of demo_llm.py or ground truth)
        document_id: override for the document_id in the JSON
        document_type: override for the document_type in the JSON
        max_source_text: truncate source_text (0 = keep all; provenance offsets need the full text)
    """
    with open(path) as f:
        data = json.load(f)
    return build_tdg(data, document_id, document_type, max_source_text)


_GENERIC_DOC_IDS = {"cli_input", "doc_001", "unknown", ""}


def _doc_id_from_path(path: str) -> str:
    """Derive a document_id from a file path: 'ahmed_tdg.json' → 'ahmed_tdg'."""
    return os.path.splitext(os.path.basename(path))[0]


def load_tdg_dir(
    directory: str,
    pattern: str = "*.json",
    **kwargs,
) -> dict[str, TemporalDependencyGraph]:
    """Load all TDG JSON files from a directory.

    Returns dict mapping document_id → TDG.
    Uses filename as document_id when the JSON contains a generic ID
    (like 'cli_input') or when IDs would collide.

    Args:
        directory: path to directory containing JSON files
        pattern: glob pattern (default: *.json)
        **kwargs: passed to load_tdg()
    """
    tdgs = {}
    for path in sorted(glob.glob(os.path.join(directory, pattern))):
        try:
            tdg = load_tdg(path, **kwargs)
            doc_id = tdg.document_id
            # Fall back to filename if the ID is generic or would collide
            if doc_id in _GENERIC_DOC_IDS or doc_id in tdgs:
                doc_id = _doc_id_from_path(path)
                tdg.document_id = doc_id
            tdgs[doc_id] = tdg
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Warning: skipping {path}: {e}")
    return tdgs


def load_json(path: str) -> dict:
    """Load a JSON file and return the raw dict. For ground truth, configs, etc."""
    with open(path) as f:
        return json.load(f)