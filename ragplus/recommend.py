"""Recommendation primitives: normalisation, source-balancing, MMR diversity,
serendipity, and per-item badges."""
from __future__ import annotations

from collections import Counter
from statistics import median

import numpy as np

from .corpus import Document


def minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def source_frequencies(docs: list[Document]) -> dict[str, int]:
    """Document count per source across the corpus."""
    return dict(Counter(d.source for d in docs))


def dominant_language(docs: list[Document]) -> str:
    """The corpus's most common known language; "" when none is recorded."""
    known = Counter(d.language for d in docs if d.language and d.language != "unknown")
    return known.most_common(1)[0][0] if known else ""


def balance_boost(docs: list[Document], pool_docs: list[Document], strength: float) -> np.ndarray:
    """Anti-popularity / 'due weight': up-weight items from rarer sources.

    Returns a multiplicative factor per pool item in [1, 1+strength]."""
    freq = source_frequencies(docs)
    counts = np.array([freq[d.source] for d in pool_docs], dtype=np.float32)
    pop = minmax(np.log1p(counts))          # 0 = rarest source, 1 = most common
    return 1.0 + strength * (1.0 - pop)


def mmr(rel: np.ndarray, sim: np.ndarray, lam: float, k: int) -> list[int]:
    """Maximal Marginal Relevance. `lam` is the *diversity* weight (0..1):
    0 = pure relevance, 1 = maximally diverse. Returns pool-local indices."""
    n = len(rel)
    k = min(k, n)
    rel = minmax(rel)
    selected: list[int] = []
    cand = list(range(n))
    while len(selected) < k and cand:
        best, best_s = cand[0], -1e18
        for c in cand:
            div = 0.0 if not selected else max(float(sim[c][s]) for s in selected)
            s = (1.0 - lam) * float(rel[c]) - lam * div
            if s > best_s:
                best_s, best = s, c
        selected.append(best)
        cand.remove(best)
    return selected


def facet_diff(a: Document, b: Document) -> float:
    """Fraction of facets on which two documents differ (source/region/decade/language)."""
    diffs = [a.source != b.source, a.region != b.region,
             a.decade != b.decade, a.language != b.language]
    return sum(diffs) / len(diffs)


def serendipity_picks(pool_docs: list[Document], rel: np.ndarray, anchor: Document,
                      chosen_ids: set[str], n: int, strength: float) -> list[tuple[int, float]]:
    """Relevant-but-unexpected items: still on-topic, but from a different corner of the
    corpus than the top result. Returns (pool_index, score) pairs, best first."""
    if strength <= 0:
        return []
    r = minmax(rel)
    scored = []
    for i, d in enumerate(pool_docs):
        if d.id in chosen_ids or d.id == anchor.id:
            continue
        score = float(r[i]) * facet_diff(d, anchor)
        scored.append((i, score))
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored[:n]


def badges(docs: list[Document], d: Document, main_decades: list[int]) -> list[str]:
    """Explanatory tags for a recommended item (rarity, period, duplication, language)."""
    freq = source_frequencies(docs)
    med = median(freq.values()) if freq else 0
    out = []
    if freq.get(d.source, 0) <= med:
        out.append("underrepresented-source")
    if main_decades and d.decade != Counter(main_decades).most_common(1)[0][0]:
        out.append("adjacent-period")
    if d.derived_from is not None:
        out.append("near-duplicate")
    main_lang = dominant_language(docs)
    if d.language and d.language != "unknown" and main_lang and d.language != main_lang:
        out.append(f"lang:{d.language}")
    return out
