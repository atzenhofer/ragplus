# ragplus

A research prototype for retrieval, recommendation and retrieval-augmented generation (RAG)
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

## Corpora

The app runs on whichever corpus file it is given: `./run.sh` uses `data/corpus.jsonl`,
`./run.sh path/to/file.jsonl` uses that file. Two corpora exist.

**Synthetic news, in the repository.** `data/corpus.jsonl` holds 408 invented news items from
1820 to 1928, written by `data/generate_corpus.py` (seed 42). The newspapers, people and
articles are invented; the cities are real. Every text is English, and the language field is
only a label. The corpus has uneven facets and clusters of wire copies, so the gap analysis and
the duplicate removal have something to report. It exists so the app can be tried without any
data of your own.

**Charters, not in the repository.** The corpus the prototype is built for. It holds the charters
of one archive from [Monasterium](https://www.monasterium.net), one record per charter, with the
regest and, where one exists, the machine transcription of the original. The data is not published
here; it is shared directly on request. It has the same format, so `./run.sh charters.jsonl`
runs it.

The format is one JSON object per line with these fields. The last two columns show what each
holds in the two corpora.

| Field | Used for | Synthetic news | Charters |
|---|---|---|---|
| `id` | citations, marking results | item id | charter md5 |
| `title` | display, BM25, embeddings, LLM passages | headline and newspaper | signature and place of issue |
| `text` | BM25, embeddings, LLM passages | article text | regest |
| `date` | display, LLM passages | date of publication, YYYY-MM-DD | date of issue; a range keeps its first day |
| `year` | year filter, decades | year of publication | year of issue |
| `source` | filter, source balance, badges, serendipity | newspaper | archive |
| `region` | filter, gap analysis, serendipity | city | place of issue, or `unknown` |
| `language` | filter, gap analysis, badges, serendipity | label | `unknown` where the record has none |
| `genre` | filter | news, editorial, review, notice | `charter:original`, `charter:copy`, `charter:draft`, `charter:mixed`, `charter:unknown` |
| `topics` (optional) | returned with each result | keywords | place names |
| `derived_from` (optional) | duplicate removal | id of the report it copies | unused |
| `url` (optional) | link in the result list | none | the charter on Monasterium |
| `htr` (optional) | second searchable text, UI toggle | none | machine transcription |

Limits:

- Every record needs all required fields, or loading stops. Use `unknown` for a missing
  value; `year` must be an integer, so undated items need one.
- The facets are fixed to `source`, `region`, `language` and `genre`, plus the year. A further
  facet, such as author, needs changes in `search`, `gaps`, `recommend`, `app` and the UI, or
  goes into one of the four fields.
- The gap analysis treats a field with more than 25 distinct values as free text and reports no
  coverage for it. Places often exceed this.
- Documents are not split into passages. Embeddings see the first `EMBED_MAX_CHARS` characters
  of each text, 16,000 by default.
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
- The cache key covers the whole corpus. Changing one document re-embeds all of them; an
  interrupted run resumes from its last 512-document checkpoint.
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
