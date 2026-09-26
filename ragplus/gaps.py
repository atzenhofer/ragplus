"""Gap analysis over the query and context.

Two signals:
  1. Filter-exclusion: relevant items removed by the current context filters.
  2. Coverage: facet cells that are (near-)empty in the relevant neighbourhood.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from .corpus import FACETS, SOURCE_FACET, Document

# Above this many distinct values a field behaves as free text rather than a facet, and
# listing its absent cells says nothing useful (e.g. a place of issue).
MAX_FACET_VALUES = 25
MAX_LISTED = 8
COVERAGE_FACETS = ("language", "place_of_issue", "authority")


def _values(doc: Document, facet: str) -> list:
    """The document's values of a facet; the decade counts as one."""
    if facet == "decade":
        return [doc.decade]
    return doc.facets.get(facet) or ["unknown"]


def _by_facet(docs: list[Document], facet: str) -> Counter:
    return Counter(value for doc in docs for value in _values(doc, facet))


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
        for facet in COVERAGE_FACETS:
            excluded[facet] = dict(_by_facet(excluded_docs, facet).most_common())
        top_sources = ", ".join(f"{count} {source}" for source, count
                                in _by_facet(excluded_docs, SOURCE_FACET).most_common(3))
        messages.append(
            f"The filters excluded {len(excluded_positions)} relevant documents "
            f"(by {FACETS[SOURCE_FACET].lower()}: {top_sources}). Widen the context to see them."
        )

    neighbourhood_docs = [docs[position] for position in neighbourhood]
    coverage = {}
    for facet in ("decade", *COVERAGE_FACETS):
        corpus_values = {value for doc in docs for value in _values(doc, facet)}
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
        label = FACETS[facet].lower()
        if free_text:
            messages.append(
                f"The {label} has {len(corpus_values)} values in this corpus, too many to "
                f"report its coverage."
            )
        elif absent:
            messages.append(
                f"No relevant documents with {label}: {_listing(absent)}."
            )

    if neighbourhood_docs:
        decades = _by_facet(neighbourhood_docs, "decade")
        span = range(min(decades), max(decades) + 10, 10)
        empty_decades = [decade for decade in span if decades.get(decade, 0) == 0]
        if empty_decades:
            messages.append("No relevant documents from the "
                            + _listing([f"{decade}s" for decade in empty_decades]) + ".")

    return {
        "messages": messages,
        "excluded": excluded,
        "coverage": coverage,
        "relevance_cut": float(relevance_cut),
    }
