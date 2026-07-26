"""
Edit scenario generation for TempChain / LegalEditBench dataset construction.

Given a TDG, generates counterfactual edit scenarios:
- Perturb a root fact (e.g., change start date by ±N days)
- Propagate expected cascades through additive dependencies
- Output structured edit scenarios matching the research framing doc schema
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

from tdg_core.tdg import TemporalDependencyGraph, TemporalFact


# Default perturbation deltas (in days)
DEFAULT_DELTAS = [30, -30, 90, -90, 365, -365]


def _supported(tdg):
    """Dependencies whose period the document actually states.

    A generated scenario riding an invented rule reads exactly like one
    riding a real rule, so the same filter applies here as everywhere else.
    """
    from tdg_core.provenance import trusted_dependencies
    return trusted_dependencies(tdg)


def generate_edit_scenarios(
    tdg: TemporalDependencyGraph,
    deltas: Optional[list[int]] = None,
) -> list[dict]:
    """
    Generate edit scenarios for a TDG.

    For each root fact (no incoming additive edges) with a parsed date,
    perturb by each delta and compute expected cascades.
    """
    if deltas is None:
        deltas = DEFAULT_DELTAS

    fact_map = {f.id: f for f in tdg.facts}

    # Find root facts: no incoming additive/ordering edges
    has_incoming = set()
    for dep in _supported(tdg):
        if dep.constraint_type in ("additive", "ordering"):
            has_incoming.add(dep.to_id)

    roots = [f for f in tdg.facts
             if f.id not in has_incoming and f.timex.date_parsed is not None]

    scenarios = []
    for root in roots:
        for delta_days in deltas:
            old_date = root.timex.date_parsed
            new_date = old_date + timedelta(days=delta_days)

            # Propagate through additive edges from this root
            cascades = _propagate_cascades(tdg, root, delta_days, fact_map)

            if cascades:  # only include scenarios that actually cascade
                scenarios.append({
                    "edit": {
                        "target_id": root.id,
                        "entity": root.entity,
                        "role": root.role,
                        "old_value": old_date.isoformat(),
                        "new_value": new_date.isoformat(),
                        "delta_days": delta_days,
                    },
                    "expected_cascades": cascades,
                    "ripple_depth": _max_depth(tdg, root.id),
                    "ripple_breadth": len(cascades),
                })

    return scenarios


def _propagate_cascades(
    tdg: TemporalDependencyGraph,
    root: TemporalFact,
    delta_days: int,
    fact_map: dict[str, TemporalFact],
) -> list[dict]:
    """
    BFS propagation of a date change through additive dependencies.
    """
    cascades = []
    visited = {root.id}
    queue = [(root.id, delta_days)]

    while queue:
        current_id, current_delta = queue.pop(0)

        for dep in _supported(tdg):
            if dep.from_id != current_id or dep.constraint_type != "additive":
                continue
            if dep.to_id in visited:
                continue

            visited.add(dep.to_id)
            target = fact_map.get(dep.to_id)
            if not target:
                continue

            if target.timex.date_parsed:
                old_val = target.timex.date_parsed
                new_val = old_val + timedelta(days=current_delta)
                cascades.append({
                    "fact_id": target.id,
                    "entity": target.entity,
                    "role": target.role,
                    "old_value": old_val.isoformat(),
                    "new_value": new_val.isoformat(),
                    "constraint": dep.constraint_expr,
                })
                # Continue propagation
                queue.append((dep.to_id, current_delta))

            elif target.role == "DURATION" and target.timex.duration_days is not None:
                # Duration doesn't change when start shifts (if we're maintaining duration)
                # But we note it for completeness
                cascades.append({
                    "fact_id": target.id,
                    "entity": target.entity,
                    "role": target.role,
                    "old_value": target.timex.value,
                    "new_value": target.timex.value,  # unchanged
                    "constraint": dep.constraint_expr,
                    "note": "duration maintained; end date shifts",
                })

    return cascades


def _max_depth(tdg: TemporalDependencyGraph, root_id: str) -> int:
    """Compute max dependency depth from a root via BFS."""
    depth = 0
    visited = {root_id}
    level = [root_id]
    while level:
        next_level = []
        for nid in level:
            for dep in _supported(tdg):
                if dep.from_id == nid and dep.to_id not in visited:
                    visited.add(dep.to_id)
                    next_level.append(dep.to_id)
        if next_level:
            depth += 1
        level = next_level
    return depth
