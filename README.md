# ragplus

Retrieval, recommendation and retrieval-augmented generation over historical documents. The
scholar sets the context (years, facets, how results are weighed), sees which relevant
documents the filters left out, and chooses the passages an answer is built from. Search
blends BM25 with embeddings; ranking adds diversity, balance across archives and a
serendipity list; each answer cites its passages and says what it did not cover.

## Set up

Requires [uv](https://docs.astral.sh/uv/), ddp_api, and an OpenAI-compatible endpoint with a
chat model and an embedding model.

    uv sync
    cp .env.example .env

`.env` holds the key and `DDP_API_URL`. The defaults are the DHInfra endpoint with
`qwen3.5-397b` and `qwen3-embedding-8b`.

## Run

Build a corpus from ddp_api; `--help` lists the filters:

    uv run -m ragplus.ddp data/corpus.jsonl authority=AT-StiAK language=de has=text

Then start the app at http://localhost:8000:

    ./run.sh

The first start embeds the corpus into `.cache/`; later starts load it. `./run.sh other.jsonl`
runs another corpus. Without an LLM the answer is the first sentence of each passage.

## Check

    uv run scripts/check_pipeline.py "a question"

runs one search and one answer without the UI.

## Disclaimer

The web interface (`ragplus/static/index.html`) was fully generated with Claude Opus 5.

## License

Apache-2.0, see `LICENSE`.
