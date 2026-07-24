"""tdg-core: the TDG format, the entailment engine, and cross-document linking.

Deterministic and dependency-light by design: no LLM, no spaCy, no
network. This property is enforced in CI (the engine-pure job) and by
tests/test_engine_pure.py.
"""
__version__ = "0.1.0"

from tdg_core.tdg import (  # noqa: F401
    SCHEMA_VERSION,
    TemporalDependencyGraph,
    TemporalFact,
    TemporalDependency,
    TimexSpan,
)
