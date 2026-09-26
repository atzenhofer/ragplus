"""Build a corpus from ddp_api: search with filters, fetch each record, write JSONL.

    .venv/bin/python -m ragplus.ddp data/corpus.jsonl authority=AT-StiAK language=de has=text
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from .config import settings

PAGE_SIZE = 100
ROLES = ("issuer", "recipient")
RETRIES = 8


def _get(client: httpx.Client, path: str, params: dict | None = None) -> dict:
    """One GET, waiting as Retry-After says while the API answers 429."""
    for attempt in range(RETRIES):
        response = client.get(path, params=params)
        if response.status_code != 429 or attempt == RETRIES - 1:
            break
        time.sleep(float(response.headers.get("Retry-After", 1)))
    response.raise_for_status()
    return response.json()


def _text(record: dict, kind: str) -> str:
    """The body of the record's text of one kind, or ""."""
    return next((text["body"] for text in record["texts"] if text["kind"] == kind), "") or ""


def _mentions(record: dict) -> list[dict]:
    """Every mention of the record: those in its texts and those outside a text."""
    in_texts = [mention for text in record["texts"] for mention in text["mentions"]]
    return in_texts + record["mentions"]


def _named(mentions: list[dict], role: str) -> list[str]:
    """The names in one role, each once, in order of appearance."""
    names = (mention["surface"] for mention in mentions if mention["role"] == role)
    return list(dict.fromkeys(names))


def _term(tradition: dict, name: str) -> list[str]:
    """The notation of the form or material concept, when the record has one."""
    term = tradition.get(f"{name}_term") or {}
    return [term["notation"]] if term.get("notation") else []


def _facets(record: dict) -> dict[str, list[str]]:
    """The record's facet values, named as ddp_api names them."""
    identity = record["identities"][0]
    mentions = _mentions(record)
    tradition = record["tradition"] or {}
    facets = {
        "authority": [identity["authority"]["label"]] if identity.get("authority") else [],
        "context": [identity["context"]["label"]] if identity.get("context") else [],
        "language": record["languages"]["tags"],
        "place_of_issue": _named(mentions, "place_of_issue"),
        **{role: _named(mentions, role) for role in ROLES},
        "form": _term(tradition, "form"),
        "material": _term(tradition, "material"),
        "label": [label["token"] or label["label"] for label in record["labels"]
                  if label["token"] or label["label"]],
    }
    return {name: values for name, values in facets.items() if values}


def document_of(record: dict) -> dict | None:
    """One corpus line from one record; None when the record has no date or no text."""
    abstract, tenor = _text(record, "abstract"), _text(record, "tenor")
    start = record["dates"]["from"]
    if not start or not (abstract or tenor):
        return None
    identity = record["identities"][0]
    archive = identity["authority"]["label"] if identity.get("authority") else ""
    title = ", ".join(part for part in (archive, identity["descriptor"]) if part)
    web = settings.ddp_web_url.rstrip("/")
    return {
        "id": record["id"],
        "atom_id": record["atom_id"],
        "title": title,
        "text": abstract or tenor[:1000],
        "tenor": tenor,
        "date": start,
        "year": int(start[:4]),
        "facets": _facets(record),
        "version_of": next((relation["target"] for relation in record["relations"]
                            if relation["type"] == "version_of"), None),
        "legacy_url": record.get("legacy_url") or "",
        "url": f"{web}/documents/{record['id']}" if web else "",
    }


def search_ids(client: httpx.Client, filters: dict[str, list[str]], query: str,
               limit: int | None) -> list[str]:
    """The ids of every document the search returns, page by page."""
    ids: list[str] = []
    page = 1
    while limit is None or len(ids) < limit:
        body = _get(client, "/search/documents", {
            **filters, "q": query, "page": page, "page_size": PAGE_SIZE, "sort": "date_asc"})
        if body["meta"].get("unavailable"):
            raise SystemExit(f"search unavailable: {body['meta']['unavailable']}")
        hits = body["data"]
        ids.extend(hit["id"] for hit in hits)
        if len(hits) < PAGE_SIZE or len(ids) >= body["meta"]["total"]:
            break
        page += 1
    return ids[:limit]


def fetch_record(client: httpx.Client, document_id: str) -> dict:
    """The full record of one document."""
    return _get(client, f"/documents/{document_id}")


def build(output: Path, filters: dict[str, list[str]], query: str = "",
          limit: int | None = None, workers: int = 4) -> None:
    """Write the documents the filters select to `output`, one JSON object per line."""
    if not settings.ddp_api_url:
        raise SystemExit("DDP_API_URL is not set.")
    with httpx.Client(base_url=settings.ddp_api_url.rstrip("/"), timeout=60) as client:
        ids = search_ids(client, filters, query, limit)
        print(f"{len(ids)} documents match the filters", file=sys.stderr)
        with ThreadPoolExecutor(workers) as pool:
            records = list(pool.map(lambda document_id: fetch_record(client, document_id), ids))

    lines = [line for line in map(document_of, records) if line]
    id_of_atom = {line["atom_id"]: line["id"] for line in lines}
    for line in lines:
        line["derived_from"] = id_of_atom.get(line.pop("version_of"))
        line.pop("atom_id")

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    with partial.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(json.dumps(line, ensure_ascii=False) + "\n")
    partial.replace(output)
    with_tenor = sum(1 for line in lines if line["tenor"])
    versions = sum(1 for line in lines if line["derived_from"])
    print(f"Wrote {len(lines)} documents to {output}. {with_tenor} have a tenor, {versions} are "
          f"a version of another document in the corpus.", file=sys.stderr)
    print(f"Skipped {len(records) - len(lines)} records without a date or without text.",
          file=sys.stderr)


def _filters(pairs: list[str]) -> dict[str, list[str]]:
    """The name=value arguments as a filter dict; a name may repeat."""
    filters: dict[str, list[str]] = {}
    for pair in pairs:
        name, separator, value = pair.partition("=")
        if not separator or not value:
            raise SystemExit(f"filter {pair!r} is not name=value")
        filters.setdefault(name, []).append(value)
    return filters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=Path)
    parser.add_argument("filters", nargs="*", help="search filters as name=value, repeatable")
    parser.add_argument("--query", default="", help="full-text query")
    parser.add_argument("--limit", type=int, help="at most this many documents")
    parser.add_argument("--workers", type=int, default=4, help="records fetched in parallel")
    args = parser.parse_args()
    build(args.output, _filters(args.filters), args.query, args.limit, args.workers)


if __name__ == "__main__":
    main()
