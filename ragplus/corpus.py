"""Corpus loading and the Document model."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# The facets a document carries, with the label the interface shows, in display order.
FACETS = {
    "authority": "Authority",
    "context": "Context",
    "language": "Language",
    "place_of_issue": "Place of issue",
    "issuer": "Issuer",
    "recipient": "Recipient",
    "form": "Form",
    "material": "Material",
    "label": "Index term",
}
# The facet that source balancing and the rare-source badge count by.
SOURCE_FACET = "authority"


@dataclass
class Document:
    id: str
    title: str
    text: str          # the abstract
    date: str          # ISO YYYY-MM-DD, the first day of the stated range
    year: int
    facets: dict[str, list[str]] = field(default_factory=dict)
    # id of the document this one is a version of, or None
    derived_from: str | None = None
    url: str = ""
    tenor: str = ""    # the transcription, where one exists

    @property
    def decade(self) -> int:
        return (self.year // 10) * 10

    @property
    def source(self) -> str:
        return self.first(SOURCE_FACET)

    def first(self, facet: str) -> str:
        """The first value of a facet, or "unknown"."""
        values = self.facets.get(facet)
        return values[0] if values else "unknown"

    def snippet(self, length: int = 240) -> str:
        """The text on one line, cut to `length` with an ellipsis."""
        text = self.text.strip().replace("\n", " ")
        return text if len(text) <= length else text[: length - 3].rstrip() + "..."

    def as_meta(self) -> dict:
        return {
            "id": self.id, "title": self.title, "date": self.date, "year": self.year,
            "facets": self.facets, "derived_from": self.derived_from,
            "url": self.url, "has_tenor": bool(self.tenor),
        }


def load_corpus(path: str | Path) -> list[Document]:
    """Read a JSONL corpus, one document per line; fields the model does not name are ignored."""
    docs: list[Document] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            docs.append(Document(
                id=record["id"], title=record["title"], text=record["text"],
                date=record["date"], year=int(record["year"]),
                facets={name: list(values) for name, values in record.get("facets", {}).items()},
                derived_from=record.get("derived_from"),
                url=record.get("url", ""), tenor=record.get("tenor", ""),
            ))
    return docs
