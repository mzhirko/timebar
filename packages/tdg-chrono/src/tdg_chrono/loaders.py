"""Document intake: plain text always; PDF/DOCX behind extras."""

from __future__ import annotations

from pathlib import Path

from tdg_core.text_cleaner import clean_to_paragraphs


def load_document_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        raw = path.read_text(errors="replace")
    elif suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise ImportError(
                "PDF intake needs the [pdf] extra: pip install 'tdg-chrono[pdf]'") from e
        raw = "\n".join((page.extract_text() or "") for page in PdfReader(path).pages)
    elif suffix == ".docx":
        import docx  # already a dependency (exports)
        raw = "\n".join(p.text for p in docx.Document(path).paragraphs)
    else:
        raise ValueError(f"unsupported document type: {path.name}")
    return "\n\n".join(clean_to_paragraphs(raw))
