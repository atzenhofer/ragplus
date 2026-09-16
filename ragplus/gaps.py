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
    return Counter(getattr(doc, facet) for doc in docs)


def _listing(values: list[str]) -> str:
    """Comma-separated, capped, naming how many were left out."""
    shown = ", ".join(values[:MAX_LISTED])
    rest = len(values) - MAX_LISTED
    return f"{shown} (and {rest} more)" if rest > 0 else shown


def analyze(docs: list[Document], dense: np.ndarray, kept_positions: list[int],
            neighbourhood_size: int = 50) -> dict:
    """Report filter-exclusion and coverage gaps for the current query and context."""
    kept = set(kept_positions)
    order = list(np.argsort(-dense))
    neighbourhood = order[:neighbourhood_size]
    relevance_cut = dense[order[min(neighbourhood_size, len(order)) - 1]] if order else 0.0
    messages: list[str] = []

    excluded_positions = [position for position in neighbourhood if position not in kept]
    excluded = {"count": len(excluded_positions)}
    if excluded_positions:
        excluded_docs = [docs[position] for position in excluded_positions]
        for facet in ("language", "region", "source"):
            excluded[facet] = dict(_by_facet(excluded_docs, facet).most_common())
        top_regions = ", ".join(f"{count} {region}" for region, count
                                in _by_facet(excluded_docs, "region").most_common(3))
        messages.append(
            f"The filters excluded {len(excluded_positions)} relevant documents "
            f"(by region: {top_regions}). Widen the context to see them."
        )

    neighbourhood_docs = [docs[position] for position in neighbourhood]
    coverage = {}
    for facet in ("decade", "language", "region"):
        corpus_values = set(getattr(doc, facet) for doc in docs)
        counts = _by_facet(neighbourhood_docs, facet)
        absent = sorted(str(value) for value in corpus_values if counts.get(value, 0) == 0)
        free_text = len(corpus_values) > MAX_FACET_VALUES
        coverage[facet] = {
            "neighbourhood": dict(counts),
            "absent": absent[:MAX_LISTED],
            "absent_count": len(absent),
            "distinct_values": len(corpus_values),
            "free_text": free_text,
        }
        if facet == "decade":
            continue
        if free_text:
            messages.append(
                f"Coverage for {facet} not reported: {len(corpus_values)} distinct values in "
                f"this corpus, so it behaves as free text rather than a facet."
            )
        elif absent:
            messages.append(
                f"Little or no material on this theme for these {facet}s: {_listing(absent)}."
            )

    if neighbourhood_docs:
        decades = _by_facet(neighbourhood_docs, "decade")
        span = range(min(decades), max(decades) + 10, 10)
        empty_decades = [decade for decade in span if decades.get(decade, 0) == 0]
        if empty_decades:
            messages.append("Temporal gap: no relevant material in "
                            + ", ".join(f"{decade}s" for decade in empty_decades) + ".")

    return {
        "messages": messages,
        "excluded": excluded,
        "coverage": coverage,
        "relevance_cut": float(relevance_cut),
    }
