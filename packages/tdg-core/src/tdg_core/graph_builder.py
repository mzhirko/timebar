"""
TDG graph builder.

Takes extracted temporal facts and builds the dependency graph:
1. Groups facts by entity
2. Detects additive constraints (end = start + duration)
3. Detects ordering constraints (start < end, birth < death)
4. Detects interval constraints (event during period)
5. Verifies arithmetic where possible
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Optional

from tdg_core.tdg import (
    ConstraintType, TemporalDependency, TemporalDependencyGraph, TemporalFact,
)


# Roles that imply ordering when co-occurring for the same entity
_ORDERED_ROLE_PAIRS = [
    ("START", "END"),       # start < end  (universal)
    ("START", "DURATION"),  # start defines duration anchor
    ("DURATION", "END"),    # duration defines end relative to start
]

# Strip article/section prefixes so "Art.4 consultations" and "Art.5 consultations"
# are grouped together as "consultations". Same regex as cross_doc.py.
_ARTICLE_PREFIX = re.compile(
    r"^(Art\.?\s*\d+[\.\d]*\s*|"
    r"[Ss]ection\s*\d+[\(\)\.\d\w]*\s*|"
    r"§\s*\d+[\(\)\.\d\w]*\s*|"
    r"Article\s*\d+[\.\d]*\s*)",
    re.IGNORECASE,
)


def _base_entity(entity: str) -> str:
    """Strip article prefix and lowercase for grouping.

    'Art.4 consultations' → 'consultations'
    'Section 111(2)(a) complaint' → 'complaint'
    'Agreement' → 'agreement'
    """
    stripped = _ARTICLE_PREFIX.sub("", entity).strip()
    return stripped.lower() if stripped else entity.lower()


def _group_facts_by_entity(facts: list[TemporalFact]) -> dict[str, list[TemporalFact]]:
    """Group facts by base entity name.

    Uses _base_entity() to strip article prefixes so that
    'Art.4 consultations' and 'Art.5 consultations' end up in the same group.
    All facts are included — repeated mentions from different clauses are
    distinct legal obligations and belong in the graph.
    """
    groups: dict[str, list[TemporalFact]] = {}
    for f in facts:
        key = _base_entity(f.entity)
        groups.setdefault(key, []).append(f)
    return groups


def _compute_delta_days(f_start: TemporalFact, f_end: TemporalFact) -> Optional[int]:
    """Compute the delta in days between two date facts."""
    d1 = f_start.timex.date_parsed
    d2 = f_end.timex.date_parsed
    if d1 and d2:
        return (d2 - d1).days
    return None


class GraphBuilder:
    """Builds a TDG from extracted temporal facts."""

    def __init__(self, base_tolerance_days: int = 15):
        """
        Args:
            base_tolerance_days: Base slack for verifying additive constraints.
                Actual tolerance scales with duration (adds ~1 day per year to
                account for leap years in the 365-day/year approximation).
        """
        self.base_tolerance = base_tolerance_days

    def _tolerance_for(self, delta_days: int) -> int:
        """Scale tolerance with duration length."""
        years = abs(delta_days) / 365
        return self.base_tolerance + int(years * 1.5)  # ~1.5 days per year for leap years

    def build(
        self,
        facts: list[TemporalFact],
        document_id: str = "doc_001",
        document_type: str = "unknown",
        source_text: str = "",
    ) -> TemporalDependencyGraph:
        """Build TDG from a list of temporal facts."""
        deps: list[TemporalDependency] = []
        groups = _group_facts_by_entity(facts)

        for entity, entity_facts in groups.items():
            # --- Detect additive constraints: end = start + duration ---
            deps.extend(self._detect_additive(entity_facts))

            # --- Detect ordering constraints ---
            deps.extend(self._detect_ordering(entity_facts))

        tdg = TemporalDependencyGraph(
            document_id=document_id,
            document_type=document_type,
            source_text=source_text,
            facts=facts,
            dependencies=deps,
        )
        return tdg

    def _detect_additive(self, entity_facts: list[TemporalFact]) -> list[TemporalDependency]:
        """
        Detect additive constraints: if an entity has START, END, and DURATION,
        then end = start + duration.

        Two modes:
        1. Verified: both dates present, arithmetic checks out (within tolerance)
        2. Structural: no dates, but START+DURATION or DURATION+END pattern
           exists — creates unverified edges to preserve graph connectivity
        """
        deps = []
        starts = [f for f in entity_facts if f.role == "START"]
        ends = [f for f in entity_facts if f.role == "END"]
        durations = [f for f in entity_facts if f.role == "DURATION"]

        connected_starts = set()  # track which starts got verified edges
        connected_ends = set()
        connected_durations = set()

        # --- Pass 1: verified edges (both dates present) ---
        for s in starts:
            for e in ends:
                actual_delta = _compute_delta_days(s, e)
                if actual_delta is not None and actual_delta <= 0:
                    continue  # end must be after start

                # Try to find a matching duration
                best_dur = None
                best_diff = float("inf")
                for d in durations:
                    if d.timex.duration_days is not None and actual_delta is not None:
                        diff = abs(actual_delta - d.timex.duration_days)
                        if diff < best_diff:
                            best_diff = diff
                            best_dur = d

                if best_dur and best_diff <= self._tolerance_for(actual_delta or best_dur.timex.duration_days or 0):
                    # Verified three-way: start + duration ≈ end
                    expr = f"end = start + {best_dur.timex.value or best_dur.timex.text}"
                    deps.append(TemporalDependency(
                        from_id=s.id, to_id=e.id,
                        constraint_type="additive",
                        constraint_expr=expr,
                        delta_days=actual_delta,
                        confidence=0.9,
                        verified=True,
                    ))
                    deps.append(TemporalDependency(
                        from_id=s.id, to_id=best_dur.id,
                        constraint_type="additive",
                        constraint_expr="duration = end - start",
                        delta_days=actual_delta,
                        confidence=0.85,
                        verified=True,
                    ))
                    connected_starts.add(s.id)
                    connected_ends.add(e.id)
                    connected_durations.add(best_dur.id)

                elif actual_delta is not None and actual_delta > 0:
                    # No matching duration — end is a fixed date, not computed from
                    # start. Use ordering to preserve structure without implying ripple.
                    deps.append(TemporalDependency(
                        from_id=s.id, to_id=e.id,
                        constraint_type="ordering",
                        constraint_expr=f"{s.id} precedes {e.id} by {actual_delta} days",
                        delta_days=actual_delta,
                        confidence=0.7,
                        verified=True,
                    ))
                    connected_starts.add(s.id)
                    connected_ends.add(e.id)

        # --- Pass 2: structural edges for undated facts ---
        # When dates are missing, we can still detect the temporal pattern
        # (e.g. "agreement lasts 1 year" without knowing which year).
        # These edges are unverified but preserve graph connectivity.
        for d in durations:
            if d.id in connected_durations:
                continue
            if d.timex.duration_days is None:
                continue

            # Link unconnected START → DURATION
            for s in starts:
                if s.id in connected_starts:
                    continue
                deps.append(TemporalDependency(
                    from_id=s.id, to_id=d.id,
                    constraint_type="additive",
                    constraint_expr=f"duration {d.timex.value or d.timex.text} from {s.entity}",
                    delta_days=d.timex.duration_days,
                    confidence=0.5,
                    verified=False,
                ))
                connected_starts.add(s.id)
                connected_durations.add(d.id)
                break  # one duration per start

            # Link unconnected DURATION → END
            for e in ends:
                if e.id in connected_ends:
                    continue
                deps.append(TemporalDependency(
                    from_id=d.id, to_id=e.id,
                    constraint_type="additive",
                    constraint_expr=f"{e.entity} = start + {d.timex.value or d.timex.text}",
                    delta_days=d.timex.duration_days,
                    confidence=0.5,
                    verified=False,
                ))
                connected_ends.add(e.id)
                connected_durations.add(d.id)
                break  # one end per duration

        return deps

    def _detect_ordering(self, entity_facts: list[TemporalFact]) -> list[TemporalDependency]:
        """
        Detect ordering constraints: start < end, etc.
        Only add ordering if not already covered by an additive constraint.
        """
        deps = []
        role_map: dict[str, list[TemporalFact]] = {}
        for f in entity_facts:
            role_map.setdefault(f.role, []).append(f)

        for role_a, role_b in _ORDERED_ROLE_PAIRS:
            if role_a == "START" and role_b == "END":
                # Already handled by additive; add ordering only if no additive found
                continue

        # Ordering between CONTAINS and START/END
        contains_facts = role_map.get("CONTAINS", [])
        for c in contains_facts:
            for s in role_map.get("START", []):
                deps.append(TemporalDependency(
                    from_id=s.id, to_id=c.id,
                    constraint_type="interval",
                    constraint_expr=f"{c.id} in [{s.id}, ...]",
                    confidence=0.6,
                ))

        return deps

    def verify_consistency(self, tdg: TemporalDependencyGraph) -> list[dict]:
        """
        Check all additive constraints for arithmetic consistency.
        Returns list of inconsistencies found.
        """
        fact_map = {f.id: f for f in tdg.facts}
        issues = []

        for dep in tdg.dependencies:
            if dep.constraint_type != "additive" or dep.delta_days is None:
                continue

            src = fact_map.get(dep.from_id)
            dst = fact_map.get(dep.to_id)
            if not src or not dst:
                continue

            if src.timex.date_parsed and dst.timex.date_parsed:
                actual = (dst.timex.date_parsed - src.timex.date_parsed).days
                expected = dep.delta_days
                if abs(actual - expected) > self._tolerance_for(expected):
                    issues.append({
                        "type": "arithmetic_mismatch",
                        "from": dep.from_id,
                        "to": dep.to_id,
                        "expected_delta": expected,
                        "actual_delta": actual,
                        "diff": abs(actual - expected),
                    })

        return issues