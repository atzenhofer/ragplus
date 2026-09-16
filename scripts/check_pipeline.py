"""Offline pipeline check: build the index, run a search and a RAG answer, print a summary.

Run from the repo root:  .venv/bin/python scripts/check_pipeline.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragplus import llm, rag, search
from ragplus.config import settings
from ragplus.corpus import load_corpus
from ragplus.index import Index
from ragplus.search import Context


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    docs = load_corpus(settings.corpus_path)
    idx = Index(docs)
    print(f"corpus={len(docs)}  embed={settings.embed_model}  "
          f"device={idx.device}  gpu={idx.gpu_name}")

    ctx = Context(query="imagery of urban poverty in the mid-19th century",
                  mode="explore", k=6)
    res = search.run(idx, ctx)
    print(f"\nsearch: {len(res['results'])} results, pool={res['pool_size']}, "
          f"matched={res['filtered_count']}/{res['corpus_size']}")
    for r in res["results"][:5]:
        print(f"  {r['score']:.3f}  {r['date']}  {r['source']:<18} "
              f"{','.join(r['badges']):<28} {r['title'][:46]}")

    print("\ngaps:")
    for m in res["gaps"]["messages"]:
        print(f"  - {m}")
    print("serendipity sources:", [r["source"] for r in res["serendipity"]])

    print(f"\nllm_available={llm.available()}")
    ans = rag.answer(idx, ctx)
    zw = ans["artifacts"]
    print(f"used_llm={ans['used_llm']}  confidence={zw['confidence']}  "
          f"independence: {zw['independence_note']}")
    print("\nanswer:\n" + ans["answer"][:700])


if __name__ == "__main__":
    main()
