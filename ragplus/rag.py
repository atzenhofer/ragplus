"""RAG: evidence selection (witness independence), cited synthesis, and the intermediate
artifacts returned with the answer."""
from __future__ import annotations

from collections import Counter

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
    """Pick independent witnesses: drop declared wire-copies (synthetic `derived_from`)
    and embedding near-duplicates (real syndication), keeping one representative each.
    Returns the witnesses and the ids skipped as duplicates."""
    pos = {d.id: i for i, d in enumerate(index.docs)}
    by_id = {d.id: d for d in index.docs}
    picked: list[Document] = []
    duplicates: list[str] = []
    picked_vecs: list = []
    seen_origins: set[str] = set()
    for r in results:
        d = by_id[r["id"]]
        origin = d.derived_from or d.id
        if origin in seen_origins:
            duplicates.append(d.id)
            continue
        vec = index.matrix(text_field)[pos[d.id]]
        if picked_vecs and max(float(vec @ pv) for pv in picked_vecs) >= dup_threshold:
            seen_origins.add(origin)
            duplicates.append(d.id)
            continue
        seen_origins.add(origin)
        picked.append(d)
        picked_vecs.append(vec)
        if len(picked) >= max_passages:
            break
    return picked, duplicates


def _passage_text(doc: Document, text_field: str) -> str:
    """The passage as retrieved: the text, or the transcription when that was searched."""
    return doc.htr if text_field == "htr" and doc.htr else f"{doc.title}. {doc.text}"


def _passages_block(docs: list[Document], text_field: str = "text") -> str:
    lines = []
    for i, d in enumerate(docs, 1):
        lines.append(f"[{i}] ({d.date}, {d.source}, {d.region}, {d.language}, {d.genre}) "
                     f"{_passage_text(d, text_field)}")
    return "\n\n".join(lines)


def _extractive(docs: list[Document]) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        lead = d.text.split(". ")[0].strip()
        parts.append(f"{lead}. [{i}]")
    return ("**(extractive fallback: no LLM reached)** " + " ".join(parts)
            + "\n\nUncertainty: assembled directly from passage leads, no synthesis.")


CONFIDENCE_BASIS = (
    "Confidence counts the independent passages the answer was built from: 1-2 low, "
    "3-4 moderate, 5 or more reasonable. It measures how much evidence was used. Whether "
    "that evidence supports the answer, and whether the answer is correct, is not measured. "
    "An answer drawn from six weak or badly transcribed passages still reads as "
    "reasonable. Read it with the selection and coverage below, and with the gaps."
)


def _confidence(n: int) -> str:
    """The evidence-count band: low, moderate or reasonable."""
    return "low" if n <= 2 else ("moderate" if n <= 4 else "reasonable")


def _selection_record(chosen_by: str, proposed: list[Document],
                      evidence: list[Document]) -> dict:
    """What the system proposed, what was used, and how the two differ."""
    proposed_ids = [d.id for d in proposed]
    kept_ids = [d.id for d in evidence]
    return {
        "by": chosen_by,
        "proposed": proposed_ids,
        "used": kept_ids,
        "declined": [i for i in proposed_ids if i not in kept_ids],
        "added": [i for i in kept_ids if i not in proposed_ids],
    }


def answer(index: Index, ctx: search.Context,
           selected_ids: list[str] | None = None) -> dict:
    res = search.run(index, ctx)
    proposed, duplicates = _select_evidence(index, res["results"], text_field=ctx.text_field)

    if selected_ids:
        by_id = {d.id: d for d in index.docs}
        evidence = [by_id[i] for i in selected_ids if i in by_id]
        chosen_by = "scholar"
    else:
        evidence = proposed
        chosen_by = "system"

    citations = [{"n": i + 1, **d.as_meta()} for i, d in enumerate(evidence)]

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

    covered_decades = sorted({d.decade for d in evidence})
    covered_regions = sorted({d.region for d in evidence})
    n = len(evidence)
    conf = _confidence(n)
    artifacts = {
        "selected_sources": [
            {"n": i + 1, "id": d.id, "title": d.title, "date": d.date,
             "source": d.source, "region": d.region, "language": d.language,
             "reason": "marked by the scholar" if chosen_by == "scholar"
                       else "top-ranked independent witness"}
            for i, d in enumerate(evidence)
        ],
        "selection": _selection_record(chosen_by, proposed, evidence),
        "independence_note": (
            "Passages marked by the scholar are used as given, without a duplicate check."
            if chosen_by == "scholar" else
            f"{len(duplicates)} candidate(s) dropped as near-duplicates / wire copies "
            "to avoid counting circular reporting as corroboration."
            if duplicates else "No near-duplicates detected among candidates."),
        "coverage": {
            "decades": [f"{d}s" for d in covered_decades],
            "regions": covered_regions,
            "languages": sorted({d.language for d in evidence}),
            "source_mix": dict(Counter(d.source for d in evidence)),
        },
        "gaps": res["gaps"]["messages"],
        "confidence": conf,
        "confidence_basis": CONFIDENCE_BASIS,
        "confidence_rationale": (
            f"Grounded in {n} independent passage(s) across "
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
