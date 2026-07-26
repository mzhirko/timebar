"""Grounded temporal context for retrieval-augmented generation.

Retrieval systems are weak exactly where this tool is strong. A vector store
returns passages; answering "was the claim in time?" needs arithmetic over
dates scattered across several passages, and a model asked to do that
arithmetic will produce a fluent wrong answer with no way to tell.

This module is the piece worth plugging in, and deliberately not more than
that. It is not a retriever, an index, or an orchestration loop; those exist
and are somebody else's job. It turns a bundle into a block of temporal
facts that a prompt can carry, with three properties an ordinary retrieved
passage does not have:

  - every date is quoted from a named document, so an answer built on it can
    cite the sentence rather than gesture at a chunk;
  - a disagreement stays a disagreement. Two documents giving different
    termination dates appear as two dates, not as whichever one retrieval
    happened to rank first;
  - the block states what could *not* be established. A model told nothing
    about the gaps will fill them; a model told "one fact has no readable
    date" has been given the chance not to.

The companion piece is ``tdg-chrono stale``, which answers the other half:
when a date turns out to be wrong, which stored answers are now wrong too.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, Optional

from tdg_chrono.chronology import Chronology, ChronologyEvent


def _tokens(text: str) -> set:
    return {t for t in "".join(
        c.lower() if c.isalnum() else " " for c in (text or "")).split() if t}


@dataclass
class TemporalContext:
    """Selected facts plus an honest account of what is missing."""

    events: list
    documents: list
    matters: dict
    unplaced: int
    unverified_quotes: int
    unsupported_relations: int
    total_events: int

    @property
    def selected(self) -> int:
        return len(self.events)


def select(chron: Chronology, *, about: str = "", since: Optional[date] = None,
           until: Optional[date] = None, on: Optional[date] = None,
           limit: Optional[int] = None) -> TemporalContext:
    """Pick the facts a question is about, in chronological order.

    Matching is plain token overlap against the event label and its quotes.
    Anything cleverer belongs in the retriever that calls this; the point
    here is that whatever comes back is quoted and checkable.
    """
    wanted = _tokens(about)
    chosen = []
    for e in chron.events:
        if on is not None and e.date != on:
            continue
        if since is not None and (e.date is None or e.date < since):
            continue
        if until is not None and (e.date is None or e.date > until):
            continue
        if wanted:
            hay = _tokens(e.label) | {t for s in e.sources for t in _tokens(s.quote)}
            if not (wanted & hay):
                continue
        chosen.append(e)
    if limit is not None:
        chosen = chosen[:limit]

    counts = chron.meta.get("counts", {})
    return TemporalContext(
        events=chosen,
        documents=sorted(chron.meta.get("documents", [])),
        matters=dict(chron.meta.get("matters", {})),
        unplaced=counts.get("unplaced", 0),
        unverified_quotes=counts.get("unverified_quotes", 0),
        unsupported_relations=counts.get("unsupported_relations", 0),
        total_events=counts.get("events", len(chron.events)),
    )


def _event_lines(e: ChronologyEvent) -> list:
    when = e.date.isoformat() if e.date else "no date"
    docs = sorted({s.doc_id for s in e.sources})
    lines = []
    if e.status == "disputed":
        lines.append(f"{when}  {e.label}  [DISPUTED: the documents do not agree]")
        for s in sorted(e.sources, key=lambda s: s.doc_id):
            value = (s.date_parsed.isoformat() if s.date_parsed
                     else (s.value or "unspecified"))
            lines.append(f'    {s.doc_id} says {value}: "{(s.quote or "").strip()}"')
        return lines
    if e.status == "derived":
        lines.append(f"{when}  {e.label}  [calculated, not stated in any document]")
        if e.derivation:
            lines.append(f"    working: {e.derivation}")
        return lines
    tag = "agreed by" if e.status == "agreed" else "stated only by"
    lines.append(f"{when}  {e.label}  [{tag} {', '.join(docs)}]")
    quote = (e.sources[0].quote or "").strip() if e.sources else ""
    if quote:
        lines.append(f'    "{quote}"')
    return lines


def render(context: TemporalContext, *, source: str = "") -> str:
    """The block to put in a prompt.

    Ends with what is missing and an instruction not to fill it in. Both
    matter: the gaps are the part a language model is most likely to paper
    over, and this is the only place it will be told about them.
    """
    where = f" in {source}" if source else ""
    head = [f"TEMPORAL FACTS from {len(context.documents)} document(s){where}.",
            "Every date below is quoted from a named document."]

    def matter_of(e):
        return next((context.matters[s.doc_id] for s in e.sources
                     if s.doc_id in context.matters), None)

    distinct = {m for m in (matter_of(e) for e in context.events) if m}
    body = []
    if len(distinct) > 1:
        # Facts from two cases look identical side by side and are about
        # different people. Presenting them ungrouped invites exactly the
        # merge this tool spends its effort preventing everywhere else.
        head.append(
            f"These documents cover {len(distinct)} SEPARATE CASES, listed "
            "apart below. Never combine facts across them, and never treat a "
            "date in one case as evidence about another.")
        for matter in sorted(distinct):
            body.append("")
            body.append(f"CASE {matter}:")
            for e in context.events:
                if matter_of(e) == matter:
                    body.extend("  " + line for line in _event_lines(e))
        loose = [e for e in context.events if matter_of(e) is None]
        if loose:
            body.append("")
            body.append("NOT ASSIGNED TO A CASE:")
            for e in loose:
                body.extend("  " + line for line in _event_lines(e))
    else:
        for e in context.events:
            body.extend(_event_lines(e))

    if not [b for b in body if b.strip()]:
        body = ["  (no facts match this question)"]

    tail = ["", "WHAT IS NOT ESTABLISHED"]
    if context.selected < context.total_events:
        tail.append(f"  {context.total_events - context.selected} further "
                    "fact(s) exist in these documents but do not match this "
                    "question.")
    if context.unplaced:
        tail.append(f"  {context.unplaced} fact(s) have no readable date and "
                    "are not listed above.")
    if context.unverified_quotes:
        tail.append(f"  {context.unverified_quotes} quote(s) could not be "
                    "checked against the document they came from.")
    if context.unsupported_relations:
        tail.append(f"  {context.unsupported_relations} stated relationship(s) "
                    "between dates were rejected as unsupported by the text.")
    if len(tail) == 2:
        tail.append("  Nothing. Every fact in these documents is listed and "
                    "every quote checks out.")
    tail += [
        "",
        "Answer only from the dates above. Do not calculate a date that is "
        "not listed, and do not resolve a DISPUTED date: report both versions "
        "and say the documents disagree."
        + (" Keep the cases separate: a fact from one case says nothing about "
           "another, even where the dates coincide." if len(distinct) > 1
           else ""),
    ]
    return "\n".join(head + [""] + body + tail)


def to_dict(context: TemporalContext) -> dict:
    """The same thing for a program rather than a prompt."""
    return {
        "documents": context.documents,
        "matters": context.matters,
        "selected": context.selected,
        "total_events": context.total_events,
        "not_established": {
            "unmatched_events": context.total_events - context.selected,
            "undated_facts": context.unplaced,
            "unverified_quotes": context.unverified_quotes,
            "unsupported_relations": context.unsupported_relations,
        },
        "facts": [{
            "date": e.date.isoformat() if e.date else None,
            "label": e.label,
            "status": e.status,
            "derivation": e.derivation,
            "versions": [{
                "document": s.doc_id,
                "fact_id": s.fact_id,
                "value": (s.date_parsed.isoformat() if s.date_parsed
                          else s.value),
                "quote": s.quote,
                "start_char": s.start_char,
                "end_char": s.end_char,
            } for s in e.sources],
        } for e in context.events],
    }


# ─── Passage retrieval ───────────────────────────────────────────────────
#
# The half a retrieval system normally supplies. Kept small on purpose: it
# scores sentences by IDF-weighted overlap with the question, using the same
# statistics the linker already computes over the bundle. No vector store,
# no index to keep in sync, nothing to install. An embedder can be passed in
# where one is configured, which buys paraphrase at the cost of precision.

from dataclasses import dataclass as _dataclass


@_dataclass
class Passage:
    """One sentence of a document, with where it came from."""

    doc_id: str
    text: str
    start_char: int
    end_char: int
    score: float


def _sentences(text: str):
    """Split into sentences, keeping each one's offset in the original."""
    out, start = [], 0
    for i, ch in enumerate(text):
        if ch in ".;:\n" and i + 1 < len(text) and (
                text[i + 1].isspace() or text[i + 1] == "\n"):
            chunk = text[start:i + 1]
            if chunk.strip():
                out.append((start, i + 1, chunk.strip()))
            start = i + 1
    if text[start:].strip():
        out.append((start, len(text), text[start:].strip()))
    return out


def retrieve(tdgs: dict, question: str, *, limit: int = 8,
             embedder=None) -> list:
    """The passages most likely to bear on the question, best first.

    Deliberately simple and local. Retrieval quality is not what this tool
    contributes; being able to say exactly which sentence an answer rests on
    is, and that needs offsets more than it needs a vector database.
    """
    from tdg_core.linking import BundleStatistics, document_text, tokenise

    stats = BundleStatistics(
        [(t.source_text or "") or document_text(t) for t in tdgs.values()])
    wanted = set(tokenise(question))
    found = []
    for doc_id, tdg in tdgs.items():
        for start, end, sentence in _sentences(tdg.source_text or ""):
            tokens = set(tokenise(sentence))
            shared = wanted & tokens
            score = sum(stats.idf(t) for t in shared) / (1 + len(tokens) ** 0.5)
            if embedder is not None and score <= 0:
                try:
                    score = embedder.similarity(question, sentence) * 0.5
                except Exception:  # noqa: BLE001 — retrieval must not fail the run
                    pass
            if score > 0:
                found.append(Passage(doc_id, sentence, start, end, score))
    found.sort(key=lambda p: -p.score)
    return found[:limit]


# ─── Asking a question ───────────────────────────────────────────────────

ANSWER_SYSTEM_PROMPT = """You answer questions about a set of legal documents.

You are given two things, in this order and for a reason:

1. TEMPORAL FACTS — dates already established from these documents by a
   deterministic engine. They are computed, checked against the source text,
   and authoritative. Where one is marked DISPUTED the documents genuinely
   conflict and no version has been chosen.

2. PASSAGES — sentences retrieved from the documents, each with the file it
   came from.

Rules:
- Take every date from the TEMPORAL FACTS. Do not read a date out of a
  passage and use it instead, and do not do arithmetic on dates yourself:
  if a date is not in the facts, say it is not established.
- Where a fact is DISPUTED, report both versions and say the documents
  disagree. Never pick one.
- Quote the sentence you relied on, naming its document, like this:
  [dismissal_letter] "Your employment terminates with effect from 12 July 2025."
- Answer only from what you are given. If neither the facts nor the passages
  answer the question, say so plainly and say what is missing.
- Be brief. This is assistive output for a professional who will check it."""


def build_prompt(chron, tdgs: dict, question: str, *, limit: int = 8,
                 embedder=None) -> tuple:
    """Assemble the question, the established dates, and the passages.

    The temporal block goes first deliberately. A model that has read the
    computed dates before it reads the prose is far less likely to pick a
    date out of a sentence and do its own arithmetic on it, which is the
    failure this whole arrangement exists to avoid.
    """
    context = select(chron, about=question)
    passages = retrieve(tdgs, question, limit=limit, embedder=embedder)

    parts = [render(context), "", "PASSAGES from the documents:"]
    if passages:
        for p in passages:
            parts.append(f'[{p.doc_id}] "{p.text}"')
    else:
        parts.append("  (no passage matched this question)")
    parts += ["", f"QUESTION: {question}"]
    return "\n".join(parts), context, passages


def ask(chron, tdgs: dict, question: str, *, model: str,
        base_url: Optional[str] = None, limit: int = 8, embedder=None,
        client=None) -> dict:
    """Answer a question about the bundle, grounded in it.

    The model writes the answer, as in any retrieval-augmented system; what
    differs is that the dates it is working from were computed rather than
    read out of prose, and it is told which of them the documents disagree
    about. Returns the answer together with everything it was given, so the
    reader can check the answer against its own inputs.
    """
    prompt, context, passages = build_prompt(
        chron, tdgs, question, limit=limit, embedder=embedder)

    if client is None:
        from tdg_chrono.models import client_for, resolve
        client = client_for(resolve("answer", model=model, base_url=base_url))

    reply = client.chat.completions.create(
        model=model, temperature=0.0,
        messages=[{"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                  {"role": "user", "content": prompt}])
    answer = (reply.choices[0].message.content or "").strip()

    unsupported = check_answer(answer, context)
    return {
        "question": question,
        "answer": answer,
        "unsupported_dates": unsupported,
        "facts_used": to_dict(context),
        "passages_used": [{"document": p.doc_id, "quote": p.text,
                           "start_char": p.start_char, "end_char": p.end_char}
                          for p in passages],
        "prompt": prompt,
    }


def check_answer(answer: str, context: TemporalContext) -> list:
    """Dates the answer states that were never established.

    The model still writes the prose, and prose is where it can go wrong: on
    one run it reported a termination "on 12 July 2025" while quoting a
    sentence that said September. The date was not in the facts it was given
    and contradicted its own citation, and nothing would have caught it.

    Every date in the established facts is known, so any other date in the
    answer is either arithmetic the model did itself or a slip. Both are
    worth flagging, and neither needs a model to detect.
    """
    import re

    from tdg_core.text_cleaner import mentions_date

    established = {e.date for e in context.events if e.date}
    for e in context.events:
        for s in e.sources:
            if s.date_parsed:
                established.add(s.date_parsed)

    # Any date-looking span in the answer, in the forms a model writes.
    pattern = re.compile(
        r"\b\d{4}-\d{2}-\d{2}\b"
        r"|\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{4}\b"
        r"|\b(?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+\d{1,2},?\s+\d{4}\b",
        re.IGNORECASE)

    unsupported = []
    for match in pattern.finditer(answer):
        span = match.group(0)
        if not any(mentions_date(span, d) for d in established):
            unsupported.append(span)
    return sorted(set(unsupported))
