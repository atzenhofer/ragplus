"""Recommendation primitives: normalisation, source-balancing, MMR diversity,
serendipity, and per-item badges."""
from __future__ import annotations

from collections import Counter
from statistics import median

import numpy as np

from .corpus import Document


def minmax(values: np.ndarray) -> np.ndarray:
    """Scale values to [0, 1]; all zeros when they are constant."""
    if values.size == 0:
        return values
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-9:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def source_frequencies(docs: list[Document]) -> dict[str, int]:
    """Document count per source across the corpus."""
    return dict(Counter(doc.source for doc in docs))


def dominant_language(docs: list[Document]) -> str:
    """The corpus's most common known language; "" when none is recorded."""
    known = Counter(doc.language for doc in docs if doc.language and doc.language != "unknown")
    return known.most_common(1)[0][0] if known else ""


def balance_boost(docs: list[Document], pool_docs: list[Document], strength: float) -> np.ndarray:
    """Up-weight items from rarer sources: a factor per pool item in [1, 1 + strength]."""
    frequencies = source_frequencies(docs)
    counts = np.array([frequencies[doc.source] for doc in pool_docs], dtype=np.float32)
    popularity = minmax(np.log1p(counts))          # 0 = rarest source, 1 = most common
    return 1.0 + strength * (1.0 - popularity)


def mmr(relevance: np.ndarray, similarity: np.ndarray, diversity: float, k: int) -> list[int]:
    """Maximal Marginal Relevance: pick k pool-local indices, trading relevance against
    similarity to what is already picked. diversity 0 = relevance only, 1 = diversity only."""
    relevance = minmax(relevance)
    selected: list[int] = []
    candidates = list(range(len(relevance)))
    while len(selected) < min(k, len(relevance)) and candidates:
        best, best_score = candidates[0], -1e18
        for candidate in candidates:
            closest = max((float(similarity[candidate][chosen]) for chosen in selected),
                          default=0.0)
            score = (1.0 - diversity) * float(relevance[candidate]) - diversity * closest
            if score > best_score:
                best_score, best = score, candidate
        selected.append(best)
        candidates.remove(best)
    return selected


def facet_diff(doc: Document, other: Document) -> float:
    """Fraction of the four facets (source, region, decade, language) on which two differ."""
    differences = [doc.source != other.source, doc.region != other.region,
                   doc.decade != other.decade, doc.language != other.language]
    return sum(differences) / len(differences)


def serendipity_picks(pool_docs: list[Document], relevance: np.ndarray, anchor: Document,
                      chosen_ids: set[str], n: int, strength: float) -> list[tuple[int, float]]:
    """Relevant items from another source, region, decade or language than the top result.
    Returns (pool_index, score) pairs, best first."""
    if strength <= 0:
        return []
    scaled = minmax(relevance)
    scored = []
    for position, doc in enumerate(pool_docs):
        if doc.id in chosen_ids or doc.id == anchor.id:
            continue
        scored.append((position, float(scaled[position]) * facet_diff(doc, anchor)))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:n]


def badges(docs: list[Document], doc: Document, main_decades: list[int]) -> list[str]:
    """Tags saying why an item stands out: rare source, other period, duplicate, language."""
    frequencies = source_frequencies(docs)
    median_count = median(frequencies.values()) if frequencies else 0
    tags = []
    if frequencies.get(doc.source, 0) <= median_count:
        tags.append("underrepresented-source")
    if main_decades and doc.decade != Counter(main_decades).most_common(1)[0][0]:
        tags.append("adjacent-period")
    if doc.derived_from is not None:
        tags.append("near-duplicate")
    main_language = dominant_language(docs)
    if doc.language and doc.language != "unknown" and main_language and doc.language != main_language:
        tags.append(f"lang:{doc.language}")
    return tags
