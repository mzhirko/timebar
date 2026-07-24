# tdg-core

The foundation of Timebar, installable on its own. Three things:

**The TDG format.** A TDG (Temporal Dependency Graph) is a JSON file
describing one document's dated facts — each with the exact sentence
and character position it came from — plus the constraints between
them ("payment due = effective date + 30 days"). It is the shared
currency of the whole system: any extractor can produce it, any tool
can consume it, in any language. Schema: `src/tdg_core/schema/tdg-v1.json`.
Validate anything with `tdg validate <file|dir>`.

**The engine.** Deterministic calendar arithmetic over TDGs. Given a
statute expressed as a TDG and a case expressed as a TDG, it discovers
the time limit from the statute's own wording (the period, the anchor,
and whether the first day counts), computes the deadline with real
calendar months and leap years, and returns a full derivation — or an
abstention naming exactly what's missing. Try it:
`tdg check --rule statute.tdg.json --instance case.json --explain`.

**The cross-document linker.** Recognises when facts in different
documents describe the same real-world event, and when they contradict
each other.

## Guarantees

- **No LLM, no NLP, no network.** This package depends only on
  `python-dateutil`, `networkx` and `jsonschema`. CI installs it alone
  and runs the full test suite offline — "deterministic, exact by
  construction" is a checked property, not a claim.
- **Jurisdiction-neutral.** No statute knowledge in code or in package
  data (enforced by `tests/test_isolation.py`). Statutes arrive as
  rule packs; jurisdiction vocabulary arrives with the pack and every
  derivation records which vocabulary was loaded.
- **Derivations, not verdicts.** Human-facing output shows the working
  and the margin; the legal conclusion is the reader's.

For the end-user tool built on this, see `tdg-chrono` and the
repository README.
