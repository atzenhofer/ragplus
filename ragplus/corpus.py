"""Corpus loading and the Document model."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Document:
    id: str
    title: str
    text: str
    date: str          # ISO YYYY-MM-DD
    year: int
    source: str        # publication / archive
    region: str        # place of publication
    language: str      # de | en | sl | ...
    genre: str         # news | editorial | review | notice
    topics: list[str] = field(default_factory=list)
    # id of the report this one copies, or None if independent
    derived_from: str | None = None
    url: str = ""
    htr: str = ""      # machine transcription of the source, where one exists

    @property
    def decade(self) -> int:
        return (self.year // 10) * 10

    def snippet(self, length: int = 240) -> str:
        """The text on one line, cut to `length` with an ellipsis."""
        text = self.text.strip().replace("\n", " ")
        return text if len(text) <= length else text[: length - 3].rstrip() + "..."

    def as_meta(self) -> dict:
        return {
            "id": self.id, "title": self.title, "date": self.date, "year": self.year,
            "source": self.source, "region": self.region, "language": self.language,
            "genre": self.genre, "topics": self.topics, "derived_from": self.derived_from,
            "url": self.url, "has_htr": bool(self.htr),
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
                date=record["date"], year=int(record["year"]), source=record["source"],
                region=record["region"], language=record["language"], genre=record["genre"],
                topics=list(record.get("topics", [])), derived_from=record.get("derived_from"),
                url=record.get("url", ""), htr=record.get("htr", ""),
            ))
    return docs
