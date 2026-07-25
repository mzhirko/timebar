"""Document intake: plain text always; PDF/DOCX behind extras.

Intake defaults to raw text with light whitespace normalization. Nothing is
deleted on the default path — normalization only rejoins hyphenated line
breaks and collapses runs of whitespace, so every character of evidence
reaches the extractor.

The aggressive paragraph cleaner is opt-in (``--clean``). It was written for
genuinely messy two-column PDF corpora and is too aggressive for ordinary
correspondence: routing everything through it silently deleted 35% of a
dismissal letter, including the termination date that fixes the limitation
clock. It also measurably degrades extraction — on a two-document bundle,
gpt-oss:20b returned zero facts from cleaned text and nine from raw. Raw
intake is the configuration the pipeline was evaluated on.
"""

from __future__ import annotations

import re
from pathlib import Path

from tdg_core.text_cleaner import clean_to_paragraphs


def normalize_whitespace(text: str) -> str:
    """Tidy whitespace without removing content.

    Every transformation here is content-preserving: no line, paragraph or
    character of substance is dropped. This is the whole default path, and
    it is deliberately dull.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Rejoin words split across a line break ("ware-\nhouse" -> "warehouse")
    text = re.sub(r"(\w)[-‐]\n[ \t]*(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)          # collapse spaces and tabs
    text = re.sub(r"[ \t]*\n", "\n", text)       # strip trailing spaces
    text = re.sub(r"\n{3,}", "\n\n", text)       # cap runs of blank lines
    return text.strip()


def load_document_raw(path: Path) -> str:
    """Document text exactly as it comes off disk, before any processing.

    The recall audit sweeps this rather than the processed text, so that a
    date lost in preprocessing is reported instead of disappearing.
    """
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        return path.read_text(errors="replace")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise ImportError(
                "PDF intake needs the [pdf] extra: pip install 'tdg-chrono[pdf]'") from e
        return "\n".join((page.extract_text() or "") for page in PdfReader(path).pages)
    if suffix == ".docx":
        import docx  # already a dependency (exports)
        return "\n".join(p.text for p in docx.Document(path).paragraphs)
    raise ValueError(f"unsupported document type: {path.name}")


def load_document_text(path: Path, *, clean: bool = False) -> str:
    """Text as the extractor sees it.

    By default this is the raw document with whitespace normalized. Pass
    clean=True only for noisy PDF extractions where headers, footers and
    two-column artifacts genuinely outweigh the risk of losing body text.
    """
    return process_text(load_document_raw(path), clean=clean)


def process_text(raw: str, *, clean: bool = False) -> str:
    """Apply the chosen intake policy to already-loaded text."""
    text, _ = process_with_report(raw, clean=clean)
    return text


def process_with_report(raw: str, *, clean: bool = False) -> tuple[str, list[str]]:
    """Apply the intake policy and report anything it discarded.

    The default path discards nothing, so the report is always empty; with
    clean=True it lists every span the cleaner removed, so a run that drops
    part of a document says so.
    """
    if clean:
        from tdg_core.text_cleaner import clean_with_report
        paragraphs, dropped = clean_with_report(raw)
        return "\n\n".join(paragraphs), dropped
    return normalize_whitespace(raw), []
