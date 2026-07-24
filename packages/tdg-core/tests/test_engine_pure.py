"""D2 enforced: importing the whole engine must not touch LLM/NLP deps.

The CI engine-pure job additionally installs tdg-core into a bare venv
(where openai/spacy simply don't exist); this test catches accidental
module-level imports even in a fat dev environment.
"""

import subprocess
import sys

FORBIDDEN = ["openai", "spacy", "chromadb", "requests", "httpx", "torch"]

PROBE = f"""
import sys
import tdg_core
import tdg_core.entailment, tdg_core.cross_doc, tdg_core.allen_classifier
import tdg_core.embeddings, tdg_core.io, tdg_core.validate, tdg_core.extractor
bad = [m for m in {FORBIDDEN!r} if m in sys.modules]
assert not bad, f"engine import pulled in {{bad}}"
print("pure")
"""

def test_engine_imports_pull_no_llm_or_nlp_deps():
    out = subprocess.run([sys.executable, "-c", PROBE],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "pure" in out.stdout
