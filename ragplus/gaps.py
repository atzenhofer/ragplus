"""Gap analysis over the query and context.

Two signals:
  1. Filter-exclusion: relevant items removed by the current context filters.
  2. Coverage: facet cells that are (near-)empty in the relevant neighbourhood.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from .corpus import Document

# Above this many distinct values a field behaves as free text rather than a facet, and
# listing its absent cells says nothing useful (e.g. a place of issue).
MAX_FACET_VALUES = 25
MAX_LISTED = 8


def _by_facet(docs: list[Document], facet: str) -> Counter:
    return Counter(getattr(d, facet) for d in docs)


def _listing(values: list[str]) -> str:
    """Comma-separated, capped, naming how many were left out."""
    shown = ", ".join(values[:MAX_LISTED])
    rest = len(values) - MAX_LISTED
    return f"{shown} (and {rest} more)" if rest > 0 else shown


def analyze(docs: list[Document], dense: np.ndarray, kept_idx: list[int],
            neigh_size: int = 50) -> dict:
    """Report filter-exclusion and coverage gaps for the current query and context."""
    kept = set(kept_idx)
    all_idx = list(range(len(docs)))

    order = list(np.argsort(-dense))
    neigh = order[:neigh_size]
    relevant_cut = dense[order[min(neigh_size, len(order)) - 1]] if order else 0.0

    messages: list[str] = []

    excluded_relevant = [i for i in neigh if i not in kept]
    excl = {"count": len(excluded_relevant)}
    if excluded_relevant:
        ex_docs = [docs[i] for i in excluded_relevant]
        for facet in ("language", "region", "source"):
            c = _by_facet(ex_docs, facet)
            excl[facet] = dict(c.most_common())
        top = ", ".join(f"{v} {k}" for k, v in _by_facet(ex_docs, "region").most_common(3))
        messages.append(
            f"Your current filters excluded {len(excluded_relevant)} otherwise-relevant "
            f"item(s) (by region: {top}). Widen the context to see them."
        )

    neigh_docs = [docs[i] for i in neigh]
    coverage = {}
    for facet in ("decade", "language", "region"):
        corpus_vals = set(getattr(d, facet) for d in docs)
        neigh_c = _by_facet(neigh_docs, facet)
        thin = sorted(str(v) for v in corpus_vals if neigh_c.get(v, 0) == 0)
        free_text = len(corpus_vals) > MAX_FACET_VALUES
        coverage[facet] = {
            "neighbourhood": dict(neigh_c),
            "absent": thin[:MAX_LISTED],
            "absent_count": len(thin),
            "distinct_values": len(corpus_vals),
            "free_text": free_text,
        }
        if thin and facet in ("language", "region") and not free_text:
            messages.append(
                f"This theme has little/no coverage for {facet}(s): {_listing(thin)}."
            )
        elif free_text and facet in ("language", "region"):
            messages.append(
                f"Coverage for {facet} not reported: {len(corpus_vals)} distinct values in this "
                f"corpus, so it behaves as free text rather than a facet."
            )

    if neigh_docs:
        decs = _by_facet(neigh_docs, "decade")
        span = range(min(decs), max(decs) + 10, 10)
        empty_decs = [d for d in span if decs.get(d, 0) == 0]
        if empty_decs:
            messages.append(
                "Temporal gap: no relevant material in "
                + ", ".join(f"{d}s" for d in empty_decs) + "."
            )

    return {
        "messages": messages,
        "excluded": excl,
        "coverage": coverage,
        "relevance_cut": float(relevant_cut),
    }
