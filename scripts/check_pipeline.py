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
    index = Index(docs)
    print(f"corpus={len(docs)}  embed={settings.embed_model}  "
          f"device={index.device}  gpu={index.gpu_name}")

    ctx = Context(query="imagery of urban poverty in the mid-19th century",
                  mode="explore", k=6)
    result = search.run(index, ctx)
    print(f"\nsearch: {len(result['results'])} results, pool={result['pool_size']}, "
          f"matched={result['filtered_count']}/{result['corpus_size']}")
    for row in result["results"][:5]:
        print(f"  {row['score']:.3f}  {row['date']}  {row['source']:<18} "
              f"{','.join(row['badges']):<28} {row['title'][:46]}")

    print("\ngaps:")
    for message in result["gaps"]["messages"]:
        print(f"  - {message}")
    print("serendipity sources:", [row["source"] for row in result["serendipity"]])

    print(f"\nllm_available={llm.available()}")
    answer = rag.answer(index, ctx)
    artifacts = answer["artifacts"]
    print(f"used_llm={answer['used_llm']}  confidence={artifacts['confidence']}  "
          f"independence: {artifacts['duplicate_note']}")
    print("\nanswer:\n" + answer["answer"][:700])


if __name__ == "__main__":
    main()
