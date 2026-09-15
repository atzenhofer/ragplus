#!/usr/bin/env bash
set -euo pipefail

if [[ ${1-} == *.jsonl ]]; then
  export CORPUS_PATH="$(realpath "$1")"
  shift
fi
cd "$(dirname "$0")"

echo "corpus: ${CORPUS_PATH:-data/corpus.jsonl}"
exec .venv/bin/python -m uvicorn ragplus.app:app --host 127.0.0.1 --port 8000 "$@"
