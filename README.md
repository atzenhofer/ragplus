# ragplus

A WIP research prototype for retrieval, recommendation and retrieval-augmented generation (RAG)
over a corpus of historical sources. The user sets the query context (period, facets, ranking
settings), sees which relevant documents the filters excluded, and can choose the passages an
answer is built from.

The prototype has no tests and several known issues (listed at the end).

## What it does

- Hybrid retrieval with BM25 and dense embeddings, filtered by year, source, language, region
  and genre.
- Re-ranking with MMR diversity and source balancing, a serendipity list, and badges that say
  why a result is shown.
- Gap analysis: relevant documents the filters excluded, and decades, languages and regions
  that the relevant documents do not cover.
- RAG with near-duplicate removal, a cited answer with an uncertainty line, and the selection,
  coverage and confidence behind it. Results the user marks become the evidence.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and an OpenAI-compatible endpoint that serves a chat
model and an embedding model. The repository contains no keys.

```bash
uv sync
cp .env.example .env    # set your key and, for another provider, base URL and model names
./run.sh                # UI at http://localhost:8000
```

The defaults point at the DHInfra endpoint (`api.dhinfra.uni-graz.at`) with `qwen3.5-397b` and
`qwen3-embedding-8b`. Embeddings can come from a separate endpoint (`EMBED_API_BASE`,
`EMBED_API_KEY`). If the LLM cannot be reached, the answer falls back to the first sentence of
each passage. Without embeddings the server does not start.

The first start embeds the corpus and stores the matrix in `.cache/`; later starts load it.
`.venv/bin/python scripts/check_pipeline.py` runs one search and one answer without the UI.

For embeddings on your own machine: `uv sync --extra local-embed`, then `EMBED_BACKEND=local`
and a Hugging Face model id in `EMBED_MODEL`.

## Example data

`data/corpus.jsonl` holds 408 synthetic news items from 1820 to 1928, written by
`data/generate_corpus.py` (seed 42). The newspapers, people and articles are invented; the
cities are real. Every text is English, and the language field is only a label. The corpus has
uneven facets and clusters of wire copies, so the gap analysis and the duplicate removal have
something to report.

## Your own data

Write the corpus as a `.jsonl` file with one JSON object per document, one per line, and pass
it to `run.sh`, for example `./run.sh ../letters.jsonl`. The repository contains no converter, so
each project writes its own. The last two columns show how medieval charters and letters fill
each field.

| Field | Used for | Charters | Letters |
|---|---|---|---|
| `id` | citations, marking results | charter id | letter id |
| `title` | display, BM25, embeddings, LLM passages | signature, issuer, place | sender and recipient |
| `text` | BM25, embeddings, LLM passages | regest | summary or full text |
| `date` | display, LLM passages | date of issue (YYYY-MM-DD) | date of writing |
| `year` | year filter, decades | year of the earliest date | year of writing |
| `source` | filter, source balance, badges, serendipity | archive | collection |
| `region` | filter, gap analysis, serendipity | place of issue | place of writing |
| `language` | filter, gap analysis, badges, serendipity | language of the charter | language of the letter |
| `genre` | filter | original or copy | letter or draft |
| `topics` (optional) | returned with each result | index terms | keywords |
| `derived_from` (optional) | duplicate removal | id of the charter it copies | id of the letter it copies |
| `url` (optional) | link in the result list | archive page | edition page |
| `htr` (optional) | second searchable text, UI toggle | HTR transcription | transcription |

Limits:

- Every record needs all required fields, or loading stops. Use `unknown` for a missing
  value; `year` must be an integer, so undated items need one.
- The facets are fixed to `source`, `region`, `language` and `genre`, plus the year. A further
  facet, such as author, needs changes in `search`, `gaps`, `recommend`, `app` and the UI, or
  goes into one of the four fields.
- The gap analysis treats a field with more than 25 distinct values as free text and reports no
  coverage for it. Places often exceed this.
- Documents are not split into passages, and embeddings see the first 2,000 characters.
  Passage-sized documents work best.
- Other fields in a record are ignored.

## How a search runs

`search.run` applies the steps in this order:

1. Embed the query and compute its cosine similarity to every document.
2. Apply the filters. The year range includes both ends. Facets combine with AND, values within
   a facet with OR. Removed documents take no part in steps 3 to 8.
3. Score title and text with BM25.
4. Min-max normalise both scores within the remaining documents and blend them:
   `rel = alpha * dense + (1 - alpha) * bm25`.
5. If source balance is on: `rel *= 1 + 0.6 * (1 - minmax(log1p(documents per source)))`.
6. Keep the top 60 as the pool.
7. Pick `k` results from the pool with MMR:
   `score = (1 - diversity) * rel - diversity * (highest similarity to a result already picked)`.
8. Pick up to 5 serendipity items from the rest of the pool:
   `rel * (share of source, region, decade and language that differ from the top result)`.
9. Run the gap analysis on the 50 documents of the whole corpus with the highest dense score.

## Settings

| Setting | Range | Effect |
|---|---|---|
| `alpha` | 0 (BM25) to 1 (dense) | BM25 finds the spelling that was typed; dense retrieval finds paraphrases and spelling variants. Acts before the pool cut, so it decides which documents are candidates. |
| `diversity` | 0 to 1 | Higher values give results that are less similar to each other and less relevant. Reorders the pool of 60. |
| `serendipity` | 0 to 1 | Side list of relevant items from other sources, regions, decades or languages. |
| balance | on or off | Up to 60% more weight for documents from small sources, applied before the pool cut. Trades precision for representation. |
| `k` | 1 to 50 | Number of results. The system chooses its evidence from these. |
| mode | preset | Sets `alpha`, `diversity`, `serendipity` and balance: precision (0.75, 0.10, 0, off), explore (0.60, 0.40, 0.40, on), gap (0.50, 0.70, 0.70, on). The mode name also goes into the LLM prompt. |

## Answers and confidence

The system walks the results in rank order. It skips a document whose `derived_from` origin is
already used or whose cosine similarity to a chosen document is 0.93 or more, and stops at 6
passages. Results marked by the user replace this selection and are used as given.

Confidence counts the passages used: 1 or 2 is low, 3 or 4 moderate, 5 or more reasonable. It
says how much evidence the answer used. Whether the passages support the answer, and whether the
answer is correct, is measured nowhere in the pipeline.

## Known issues

- Serendipity works as a switch. Every value above 0 returns the same 5 items
  (`recommend.serendipity_picks`).
- The gap analysis ranks by the dense score alone and ignores `alpha` and balance. With `alpha`
  near 0, results and gap messages describe different rankings (`gaps.analyze`).
- Coverage gaps compare 50 documents with every value in the corpus, so most queries report
  some.
- BM25 always reads title and text. The transcription toggle changes the dense side only.
- The API encoder cuts every text at 2,000 characters (`embed.MAX_CHARS`).
- The cache key covers the whole corpus. Changing one document re-embeds all of them.
- The UI sends no pool size, so the pool is always 60.
- In the example corpus, language filters and language badges follow the labels, while all
  texts are English.
- There are no tests; `scripts/check_pipeline.py` is the only check.
- The local embedding backend has not been tested with the current dependencies.

## Layout

```
ragplus/    app, config, corpus, embed, index, search, recommend, gaps, rag, llm, static/index.html
data/       corpus.jsonl and generate_corpus.py
scripts/    check_pipeline.py
```

## Disclaimer

The web interface (`ragplus/static/index.html`), synthetic data (`data/generate_corpus.py` and the `data/corpus.jsonl` it writes) was fully generated with Claude Opus 5.

## License

Apache-2.0, see `LICENSE`.
