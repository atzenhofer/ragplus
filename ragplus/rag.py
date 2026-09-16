"""RAG: evidence selection (witness independence), cited synthesis, and the intermediate
artifacts returned with the answer."""
from __future__ import annotations

from collections import Counter

import numpy as np

from . import llm, search
from .corpus import Document
from .index import Index

SYSTEM = (
    "You are a source-critical research assistant for the humanities. Answer ONLY from the "
    "numbered passages provided. Cite every claim with bracketed passage numbers like [2]. "
    "Give temporal and spatial context (dates, places) for what you report. Prefer independent "
    "witnesses over repeated wire copies. Write the answer in the language the passages are "
    "predominantly written in, and keep names, places and quoted wording in their original "
    "spelling. Do NOT use outside knowledge. End with a short "
    "'Uncertainty' line stating how well-supported the answer is and what is missing. If the "
    "passages do not answer the question, say so plainly."
)


def _select_evidence(index: Index, results: list[dict], max_passages: int = 6,
                     dup_threshold: float = 0.93,
                     text_field: str = "text") -> tuple[list[Document], list[str]]:
    """Walk the results in rank order and keep one document per origin: a document that
    copies one already kept (`derived_from`), or whose embedding is within the duplicate
    threshold of one, is skipped. Returns the kept documents and the skipped ids."""
    position_of = {doc.id: position for position, doc in enumerate(index.docs)}
    by_id = {doc.id: doc for doc in index.docs}
    picked: list[Document] = []
    picked_vectors: list[np.ndarray] = []
    duplicates: list[str] = []
    seen_origins: set[str] = set()
    for row in results:
        doc = by_id[row["id"]]
        origin = doc.derived_from or doc.id
        if origin in seen_origins:
            duplicates.append(doc.id)
            continue
        vector = index.matrix(text_field)[position_of[doc.id]]
        closest = max((float(vector @ kept) for kept in picked_vectors), default=0.0)
        seen_origins.add(origin)
        if closest >= dup_threshold:
            duplicates.append(doc.id)
            continue
        picked.append(doc)
        picked_vectors.append(vector)
        if len(picked) >= max_passages:
            break
    return picked, duplicates


def _passage_text(doc: Document, text_field: str) -> str:
    """The passage as retrieved: the text, or the transcription when that was searched."""
    return doc.htr if text_field == "htr" and doc.htr else f"{doc.title}. {doc.text}"


def _passages_block(docs: list[Document], text_field: str = "text") -> str:
    """The numbered passages as the prompt shows them."""
    return "\n\n".join(
        f"[{number}] ({doc.date}, {doc.source}, {doc.region}, {doc.language}, {doc.genre}) "
        f"{_passage_text(doc, text_field)}"
        for number, doc in enumerate(docs, 1))


def _extractive(docs: list[Document]) -> str:
    """The answer when no LLM is reached: the first sentence of each passage, numbered."""
    leads = " ".join(f"{doc.text.split('. ')[0].strip()}. [{number}]"
                     for number, doc in enumerate(docs, 1))
    return ("**No LLM reached; this is the first sentence of each passage.** " + leads
            + "\n\nUncertainty: not a synthesis, so nothing was checked across passages.")


CONFIDENCE_BASIS = (
    "Confidence counts the independent passages the answer was built from: 1-2 low, "
    "3-4 moderate, 5 or more reasonable. It measures how much evidence was used. Whether "
    "that evidence supports the answer, and whether the answer is correct, is not measured. "
    "An answer drawn from six weak or badly transcribed passages still reads as "
    "reasonable. Read it with the selection and coverage below, and with the gaps."
)


def _confidence(passage_count: int) -> str:
    """The evidence-count band: low, moderate or reasonable."""
    if passage_count <= 2:
        return "low"
    if passage_count <= 4:
        return "moderate"
    return "reasonable"


def _duplicate_note(chosen_by: str, duplicates: list[str]) -> str:
    """What the duplicate check did, for the reader of the answer."""
    if chosen_by == "scholar":
        return "Passages marked by the scholar are used as given, without a duplicate check."
    if duplicates:
        return (f"{len(duplicates)} near-duplicates dropped, so one report copied several "
                "times counts once.")
    return "No near-duplicates among the candidates."


def _selection_record(chosen_by: str, proposed: list[Document],
                      evidence: list[Document]) -> dict:
    """What the system proposed, what was used, and how the two differ."""
    proposed_ids = [doc.id for doc in proposed]
    kept_ids = [doc.id for doc in evidence]
    return {
        "by": chosen_by,
        "proposed": proposed_ids,
        "used": kept_ids,
        "declined": [doc_id for doc_id in proposed_ids if doc_id not in kept_ids],
        "added": [doc_id for doc_id in kept_ids if doc_id not in proposed_ids],
    }


def answer(index: Index, ctx: search.Context,
           selected_ids: list[str] | None = None) -> dict:
    res = search.run(index, ctx)
    proposed, duplicates = _select_evidence(index, res["results"], text_field=ctx.text_field)

    if selected_ids:
        by_id = {doc.id: doc for doc in index.docs}
        evidence = [by_id[doc_id] for doc_id in selected_ids if doc_id in by_id]
        chosen_by = "scholar"
    else:
        evidence = proposed
        chosen_by = "system"

    citations = [{"n": number, **doc.as_meta()} for number, doc in enumerate(evidence, 1)]

    if not evidence:
        text, used_llm = "No documents match the current context, so no answer can be grounded.", False
    else:
        user = (f"Research question ({ctx.mode} mode): {ctx.query}\n\n"
                f"Passages:\n{_passages_block(evidence, ctx.text_field)}\n\n"
                "Write a concise, cited answer.")
        try:
            text = llm.chat([{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": user}], max_tokens=900)
            used_llm = True
        except llm.LLMUnavailable:
            text = _extractive(evidence)
            used_llm = False

    covered_decades = sorted({doc.decade for doc in evidence})
    covered_regions = sorted({doc.region for doc in evidence})
    reason = "marked by the scholar" if chosen_by == "scholar" else "top-ranked independent witness"
    artifacts = {
        "selected_sources": [
            {"n": number, "id": doc.id, "title": doc.title, "date": doc.date,
             "source": doc.source, "region": doc.region, "language": doc.language,
             "reason": reason}
            for number, doc in enumerate(evidence, 1)
        ],
        "selection": _selection_record(chosen_by, proposed, evidence),
        "independence_note": _duplicate_note(chosen_by, duplicates),
        "coverage": {
            "decades": [f"{decade}s" for decade in covered_decades],
            "regions": covered_regions,
            "languages": sorted({doc.language for doc in evidence}),
            "source_mix": dict(Counter(doc.source for doc in evidence)),
        },
        "gaps": res["gaps"]["messages"],
        "confidence": _confidence(len(evidence)),
        "confidence_basis": CONFIDENCE_BASIS,
        "confidence_rationale": (
            f"Grounded in {len(evidence)} independent passage(s) across "
            f"{len(covered_decades)} decade(s) and {len(covered_regions)} region(s)."),
    }

    return {
        "query": ctx.query,
        "answer": text,
        "used_llm": used_llm,
        "citations": citations,
        "artifacts": artifacts,
        "results": res["results"],
        "serendipity": res["serendipity"],
        "gaps": res["gaps"],
    }
