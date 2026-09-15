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

    def snippet(self, n: int = 240) -> str:
        t = self.text.strip().replace("\n", " ")
        return t if len(t) <= n else t[: n - 3].rstrip() + "..."

    def as_meta(self) -> dict:
        return {
            "id": self.id, "title": self.title, "date": self.date, "year": self.year,
            "source": self.source, "region": self.region, "language": self.language,
            "genre": self.genre, "topics": self.topics, "derived_from": self.derived_from,
            "url": self.url, "has_htr": bool(self.htr),
        }


def load_corpus(path: str | Path) -> list[Document]:
    path = Path(path)
    docs: list[Document] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            docs.append(Document(
                id=d["id"], title=d["title"], text=d["text"], date=d["date"],
                year=int(d["year"]), source=d["source"], region=d["region"],
                language=d["language"], genre=d["genre"],
                topics=list(d.get("topics", [])), derived_from=d.get("derived_from"),
                url=d.get("url", ""), htr=d.get("htr", ""),
            ))
    return docs
