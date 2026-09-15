"""Query context and the retrieval + recommendation pipeline.

`Context` holds the user's settings; `run` turns it into a candidate pool, a re-ranked
result set, a serendipity side-list, and a gap analysis.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from . import gaps, recommend
from .corpus import Document
from .index import Index


@dataclass
class Context:
    query: str
    mode: str = "explore"                 # precision | explore | gap (label + prompt hint)
    text_field: str = "text"              # text = summary or full text, htr = transcription
    year_from: int | None = None
    year_to: int | None = None
    sources: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    regions: list[str] = field(default_factory=list)
    genres: list[str] = field(default_factory=list)
    k: int = 10                           # results returned
    pool: int = 60                        # candidate pool before re-ranking
    alpha: float = 0.6                    # lexical(0) .. dense(1)
    diversity: float = 0.3                # MMR lambda: relevance(0) .. diversity(1)
    serendipity: float = 0.3              # 0 .. 1
    balance_sources: bool = True          # anti-popularity / due weight

    def clamp(self) -> "Context":
        if self.text_field not in ("text", "htr"):
            self.text_field = "text"
        self.alpha = min(1.0, max(0.0, self.alpha))
        self.diversity = min(1.0, max(0.0, self.diversity))
        self.serendipity = min(1.0, max(0.0, self.serendipity))
        self.k = max(1, min(50, self.k))
        self.pool = max(self.k, min(200, self.pool))
        return self


def _filter(index: Index, ctx: Context) -> list[int]:
    keep = []
    for i, d in enumerate(index.docs):
        if ctx.year_from and d.year < ctx.year_from:
            continue
        if ctx.year_to and d.year > ctx.year_to:
            continue
        if ctx.sources and d.source not in ctx.sources:
            continue
        if ctx.languages and d.language not in ctx.languages:
            continue
        if ctx.regions and d.region not in ctx.regions:
            continue
        if ctx.genres and d.genre not in ctx.genres:
            continue
        keep.append(i)
    return keep


def _result_row(d: Document, score: float, badges: list[str], why: str) -> dict:
    row = d.as_meta()
    row.update(score=round(float(score), 4), badges=badges, why=why,
               snippet=d.snippet())
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

    kept_arr = np.array(kept)
    dense = dense_all[kept_arr]
    lex = index.lexical_scores(ctx.query)[kept_arr]
    rel = ctx.alpha * recommend.minmax(dense) + (1 - ctx.alpha) * recommend.minmax(lex)

    pool_docs_all = [docs[i] for i in kept]
    if ctx.balance_sources:
        rel = rel * recommend.balance_boost(docs, pool_docs_all, strength=0.6)

    pool_order = list(np.argsort(-rel))[: ctx.pool]        # indices into kept/rel
    pool_global = [kept[i] for i in pool_order]
    pool_docs = [docs[i] for i in pool_global]
    pool_rel = rel[pool_order]
    sim = index.pairwise(pool_global, text_field)

    chosen_local = recommend.mmr(pool_rel, sim, lam=ctx.diversity, k=ctx.k)
    chosen_decades = [pool_docs[i].decade for i in chosen_local]

    results = []
    chosen_ids = set()
    for rank, li in enumerate(chosen_local):
        d = pool_docs[li]
        chosen_ids.add(d.id)
        bdg = recommend.badges(docs, d, chosen_decades)
        why = f"hybrid rank {rank + 1}" + (
            " · diversified (MMR)" if ctx.diversity > 0 else "")
        results.append(_result_row(d, pool_rel[li], bdg, why))

    serendipity = []
    if results:
        anchor = pool_docs[chosen_local[0]]
        for li, sc in recommend.serendipity_picks(
                pool_docs, pool_rel, anchor, chosen_ids,
                n=5, strength=ctx.serendipity):
            d = pool_docs[li]
            why = "relevant but from a different " + "/".join(
                f for f, cond in (("source", d.source != anchor.source),
                                  ("period", d.decade != anchor.decade),
                                  ("region", d.region != anchor.region),
                                  ("language", d.language != anchor.language)) if cond
            )
            serendipity.append(_result_row(d, sc, recommend.badges(docs, d, chosen_decades), why))

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
        "source_mix": dict(Counter(r["source"] for r in results)),
    }
