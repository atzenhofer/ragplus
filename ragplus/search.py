"""Query context and the retrieval + recommendation pipeline.

`Context` holds the user's settings; `run` turns it into a candidate pool, a re-ranked
result set, a serendipity side-list, and a gap analysis.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from . import gaps, recommend
from .corpus import FACETS, Document
from .index import Index


@dataclass
class Context:
    query: str
    mode: str = "explore"                 # precision | explore | gap (label + prompt hint)
    text_field: str = "text"              # text = abstract, tenor = transcription
    year_from: int | None = None
    year_to: int | None = None
    filters: dict[str, list[str]] = field(default_factory=dict)   # facet -> accepted values
    k: int = 10                           # results returned
    pool: int = 60                        # candidate pool before re-ranking
    alpha: float = 0.6                    # lexical(0) .. dense(1)
    diversity: float = 0.3                # MMR lambda: relevance(0) .. diversity(1)
    serendipity: float = 0.3              # 0 .. 1
    balance_sources: bool = True          # up-weight rare sources

    def clamp(self) -> "Context":
        if self.text_field not in ("text", "tenor"):
            self.text_field = "text"
        self.alpha = min(1.0, max(0.0, self.alpha))
        self.diversity = min(1.0, max(0.0, self.diversity))
        self.serendipity = min(1.0, max(0.0, self.serendipity))
        self.k = max(1, min(50, self.k))
        self.pool = max(self.k, min(200, self.pool))
        return self


def _matches(doc: Document, ctx: Context) -> bool:
    """Whether the document passes the context's year range and facet filters."""
    if ctx.year_from and doc.year < ctx.year_from:
        return False
    if ctx.year_to and doc.year > ctx.year_to:
        return False
    return all(set(doc.facets.get(facet, ())) & set(values)
               for facet, values in ctx.filters.items() if values)


def _filter(index: Index, ctx: Context) -> list[int]:
    return [position for position, doc in enumerate(index.docs) if _matches(doc, ctx)]


def _different_facets(doc: Document, anchor: Document) -> list[str]:
    """The facets on which the document differs from the anchor."""
    differs = [FACETS[facet].lower() for facet in recommend.CONTRAST_FACETS
               if doc.first(facet) != anchor.first(facet)]
    return differs + (["period"] if doc.decade != anchor.decade else [])


def _and_list(words: list[str]) -> str:
    """The words as "a", "a and b" or "a, b and c"."""
    if len(words) < 2:
        return "".join(words)
    return ", ".join(words[:-1]) + " and " + words[-1]


def _result_row(doc: Document, score: float, badges: list[str], why: str) -> dict:
    row = doc.as_meta()
    row.update(score=round(float(score), 4), badges=badges, why=why,
               snippet=doc.snippet())
    return row


def run(index: Index, ctx: Context) -> dict:
    """Turn a Context into results, serendipity picks, and a gap analysis."""
    ctx = ctx.clamp()
    docs = index.docs
    text_field = ctx.text_field if ctx.text_field in index.fields else "text"
    dense_all = index.dense_scores(ctx.query, text_field)   # whole corpus, for gaps

    kept = _filter(index, ctx)
    if not kept:
        return {"context": ctx.__dict__, "results": [], "serendipity": [],
                "gaps": gaps.analyze(docs, dense_all, kept),
                "pool_size": 0, "filtered_count": 0, "on_gpu": index.on_gpu,
                "note": "No documents match the current context filters."}

    kept_array = np.array(kept)
    dense = dense_all[kept_array]
    lexical = index.lexical_scores(ctx.query)[kept_array]
    relevance = (ctx.alpha * recommend.minmax(dense)
                 + (1 - ctx.alpha) * recommend.minmax(lexical))

    kept_docs = [docs[position] for position in kept]
    if ctx.balance_sources:
        relevance = relevance * recommend.balance_boost(docs, kept_docs, strength=0.6)

    pool_order = list(np.argsort(-relevance))[: ctx.pool]        # positions in kept
    pool_global = [kept[position] for position in pool_order]
    pool_docs = [docs[position] for position in pool_global]
    pool_relevance = relevance[pool_order]
    similarity = index.pairwise(pool_global, text_field)

    chosen_local = recommend.mmr(pool_relevance, similarity, diversity=ctx.diversity, k=ctx.k)
    chosen_decades = [pool_docs[position].decade for position in chosen_local]
    why_suffix = " · diversified (MMR)" if ctx.diversity > 0 else ""

    results = []
    chosen_ids = set()
    for rank, local in enumerate(chosen_local):
        doc = pool_docs[local]
        chosen_ids.add(doc.id)
        badges = recommend.badges(docs, doc, chosen_decades)
        results.append(_result_row(doc, pool_relevance[local], badges,
                                   f"hybrid rank {rank + 1}{why_suffix}"))

    serendipity = []
    if results:
        anchor = pool_docs[chosen_local[0]]
        picks = recommend.serendipity_picks(pool_docs, pool_relevance, anchor, chosen_ids,
                                            count=5, strength=ctx.serendipity)
        for local, score in picks:
            doc = pool_docs[local]
            why = "relevant, from another " + _and_list(_different_facets(doc, anchor))
            badges = recommend.badges(docs, doc, chosen_decades)
            serendipity.append(_result_row(doc, score, badges, why))

    gap = gaps.analyze(docs, dense_all, kept)

    return {
        "context": ctx.__dict__,
        "results": results,
        "serendipity": serendipity,
        "gaps": gap,
        "pool_size": len(pool_global),
        "filtered_count": len(kept),
        "corpus_size": len(docs),
        "text_field": text_field,
        "text_fields": index.fields,
        "on_gpu": index.on_gpu,
        "source_mix": dict(Counter(pool_docs[local].source for local in chosen_local)),
    }
