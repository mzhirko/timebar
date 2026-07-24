"""Derivation trace renderers (Phase 1.6).

Shows the working, not just the answer. Per D3, the Desk-facing text is a
derivation: it states the computed deadline and the margin, and lets the
professional draw the conclusion. The verdict label stays in the JSON for
Lab/Stack use.

An INDETERMINATE renders as a diagnosis with a request: what is missing
and what to supply — the same message a repair loop would consume.
"""

from __future__ import annotations

import html as _html
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid import cycle at runtime typing only
    from tdg_core.entailment import EntailmentResult


def _offset_phrase(off: dict) -> str:
    parts = []
    if off.get("years"):
        parts.append(f"{off['years']} year(s)")
    if off.get("months"):
        parts.append(f"{off['months']} month(s)")
    if off.get("days"):
        parts.append(f"{off['days']} day(s)")
    phrase = " + ".join(parts) or "0 days"
    if off.get("minus_one_day"):
        phrase += " − 1 day"
    return phrase


def render_text(result: "EntailmentResult") -> str:
    """Plain-text derivation for ``--explain``."""
    t = result.trace or {}
    rule = t.get("rule", {})
    out: list[str] = []
    w = out.append

    w(f"Rule ({rule.get('source_doc', result.rule_doc)}):")
    w(f"  {rule.get('description', result.rule_description)}")
    if rule.get("statute_sentence"):
        w(f'  stated in: "{rule["statute_sentence"].strip()}"')

    off = rule.get("offset", {})
    w("")
    w(f"Period: {_offset_phrase(off)}")
    src = rule.get("inclusivity_source", "assumed")
    if src == "discovered":
        w(f"  anchor-day counting: DISCOVERED from the statute's own wording "
          f'("{rule.get("inclusivity_evidence", "")}")')
    else:
        w("  anchor-day counting: ASSUMED (connector not readable in the "
          "clause — verify whether the anchor day counts as day 1)")

    am = t.get("anchor_match")
    if am:
        w("")
        w(f"Anchor: '{rule.get('anchor_concept', '')}' matched to "
          f"{am['entity']} = {am['date']}  "
          f"[{result.instance_doc}/{am['fact_id']}, conf {am['confidence']}]")
        if am.get("quote"):
            w(f'  quote: "{am["quote"].strip()}"')

    ar = t.get("arithmetic")
    if ar:
        w("")
        if "conciliation" in t:
            c = t["conciliation"]
            w(f"Deadline: {ar['anchor_date']} + {_offset_phrase(off)} "
              f"= {ar['deadline_base']}")
            w(f"  early conciliation Day A {c['day_a']} → Day B {c['day_b']}: "
              f"clock paused, +{c['extension_days']}d → "
              f"effective deadline {ar['deadline_effective']}")
        else:
            w(f"Deadline: {ar['anchor_date']} + {_offset_phrase(off)} "
              f"= {ar['deadline_effective']}")

    sel = t.get("action_selection")
    if sel:
        s = sel["selected"]
        w("")
        w(f"Action: {s['entity']} = {s['date']}  "
          f"[{result.instance_doc}/{s['fact_id']}, score {s['score']}, via {s['how']}]")
        if s.get("quote"):
            w(f'  quote: "{s["quote"].strip()}"')
        for p in sel.get("passed_over", []):
            w(f"  passed over: {p['entity']} ({p['date']}, score {p['score']})")

    w("")
    if result.verdict == "INDETERMINATE":
        w(f"Result: cannot answer — {result.explanation}")
        if t.get("request"):
            w(f"  → {t['request']}")
    else:
        m = result.days_over or 0
        rel = (f"{abs(m)} day(s) before the deadline" if m < 0
               else "on the deadline" if m == 0
               else f"{abs(m)} day(s) after the deadline")
        w(f"Result: the action ({result.action_date}) falls {rel} "
          f"({result.deadline_computed}).")
        w(f"  match confidence {result.match_confidence:.2f}"
          + (" · conciliation extension applied" if result.acas_applied else ""))
    return "\n".join(out)


def render_line(result: "EntailmentResult") -> str:
    """One-line derivation without a verdict label (D3). Default CLI output."""
    if result.verdict == "INDETERMINATE":
        line = f"cannot answer: {result.explanation}"
        req = (result.trace or {}).get("request")
        return line + (f" -> {req}" if req else "")
    m = result.days_over or 0
    rel = (f"{abs(m)}d before deadline" if m < 0 else "on the deadline"
           if m == 0 else f"{abs(m)}d AFTER deadline")
    return (f"anchor {result.anchor_date} -> deadline {result.deadline_computed}"
            f" -> action {result.action_date} ({rel}; "
            f"conf {result.match_confidence:.2f}). Use --explain for the full working.")


def render_html(results: list["EntailmentResult"], title: str = "Derivation trace") -> str:
    """Standalone HTML page for one or more derivations. No dependencies."""
    def esc(x) -> str:
        return _html.escape(str(x))

    blocks = []
    for r in results:
        badge = ("indeterminate" if r.verdict == "INDETERMINATE" else "answered")
        blocks.append(
            f'<section class="trace {badge}">'
            f"<h2>{esc(r.rule_description)}</h2>"
            f"<pre>{esc(render_text(r))}</pre>"
            f"</section>")
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{esc(title)}</title>
<style>
 body {{ font-family: Georgia, serif; max-width: 60rem; margin: 2rem auto;
        padding: 0 1rem; color: #1a1a1a; }}
 h1 {{ font-size: 1.4rem; }}
 section.trace {{ border-left: 4px solid #1f3864; padding: .5rem 1rem;
                 margin: 1.5rem 0; background: #f7f8fa; }}
 section.indeterminate {{ border-left-color: #b45309; background: #fdf7ef; }}
 pre {{ white-space: pre-wrap; font-family: ui-monospace, monospace;
       font-size: .85rem; line-height: 1.5; }}
 footer {{ font-size: .75rem; color: #666; margin-top: 2rem;
          border-top: 1px solid #ddd; padding-top: .5rem; }}
</style></head><body>
<h1>{esc(title)}</h1>
{''.join(blocks)}
<footer>Derivation, not advice. Every date above is quoted from a source
document or computed by calendar arithmetic from quoted dates; verify the
quotes before relying on the result.</footer>
</body></html>"""
