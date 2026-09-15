"""Generate a deterministic, multi-domain 'news' corpus for ragplus.

Everything in the corpus is invented: newspapers, people and articles. Region and language
are metadata labels; all texts are English. The corpus is shaped to exercise the pipeline:
  * several disciplines (social, art, literary, medical, labour and migration history, archaeology),
  * metadata per record (date, source, region, language, genre, topics),
  * uneven facets (few Slovene and Graz items, topics bound to plausible eras) for the gap analysis,
  * near-duplicate wire-copy clusters for the witness-independence step.

Run:  python data/generate_corpus.py   ->   data/corpus.jsonl
"""
from __future__ import annotations

import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent / "corpus.jsonl"
rng = random.Random(42)

# source -> (region, language, popularity tier, genre-lean)
SOURCES = {
    "Donau-Courier":            ("Vienna", "de", "high", "news"),
    "Ringstrassen-Bote":        ("Vienna", "de", "high", "news"),
    "Schlossberg-Rundschau":    ("Graz", "de", "medium", "news"),
    "Murinsel-Post":            ("Graz", "de", "low", "news"),
    "The Thameside Chronicle":  ("London", "en", "high", "news"),
    "The Irwell Examiner":      ("Manchester", "en", "medium", "editorial"),
    "Le Courrier de la Bievre": ("Paris", "fr", "medium", "news"),
    "Barjanski Vestnik":        ("Ljubljana", "sl", "low", "news"),
    "The Calton Hill Review":   ("Edinburgh", "en", "low", "review"),
    "Il Corriere dei Navigli":  ("Milan", "it", "low", "news"),
}

PLACES = {
    "Vienna": ["the Leopoldstadt", "the Ringstrasse", "the Prater", "Favoriten"],
    "Graz":   ["the Murvorstadt", "the Schlossberg", "the Lend district"],
    "London": ["Whitechapel", "the East End", "Southwark", "Clerkenwell"],
    "Manchester": ["Ancoats", "Salford", "the Oldham Road"],
    "Paris":  ["the Marais", "Belleville", "the Faubourg Saint-Antoine"],
    "Ljubljana": ["the Old Town", "Trnovo", "the Ljubljanica quay"],
    "Edinburgh": ["the Old Town", "Leith", "the Cowgate"],
    "Milan":  ["the Navigli", "Porta Ticinese", "the Brera"],
}
PERSONS = ["Herr Dr. Weiss", "Frau Novak", "Mr. Ashcroft", "Signora Conti",
           "Monsieur Laurent", "Miss Hartley", "Gospod Kranjc", "Councillor Reid",
           "Professor Halm", "Madame Girard", "Alderman Pryce", "Dr. Fischer"]

# topic -> (era_start, era_end, keywords, title templates, body templates, default genre)
TOPICS = {
    "urban_poverty": (1820, 1900,
        ["poverty", "tenement", "destitution", "relief", "workhouse", "slum", "charity", "hunger"],
        ["Distress among the poor of {place}", "Report on destitution in {place}",
         "The condition of the labouring poor at {place}"],
        ["A survey of {place} finds families crowded in damp tenements, dependent on parish relief.",
         "{person} describes widespread destitution and appeals for charitable relief this winter.",
         "Overseers report that the workhouse at {place} can no longer meet the demand for bread."],
        "news"),
    "womens_suffrage": (1860, 1920,
        ["suffrage", "franchise", "women", "vote", "petition", "emancipation", "assembly", "rights"],
        ["Meeting on the female franchise at {place}", "Petition for women's suffrage",
         "Debate on the emancipation of women"],
        ["{person} addressed a crowded assembly at {place} in favour of extending the franchise to women.",
         "A petition demanding the vote for women gathered many signatures across {place}.",
         "The question of female suffrage divided opinion at the meeting held in {place}."],
        "news"),
    "railway_expansion": (1835, 1910,
        ["railway", "locomotive", "line", "station", "gauge", "shares", "embankment", "traffic"],
        ["New railway line opened at {place}", "The railway question at {place}",
         "Shares in the {place} line"],
        ["The new railway line reached {place}, cutting the journey and opening trade to the interior.",
         "{person} laid the foundation of the station at {place} amid a great crowd.",
         "Speculation in railway shares ran high as the {place} embankment neared completion."],
        "news"),
    "theatre_review": (1820, 1930,
        ["theatre", "drama", "actress", "stage", "tragedy", "comedy", "audience", "playwright"],
        ["A new tragedy at the {place} theatre", "Review: the season at {place}",
         "The stage at {place}"],
        ["The company at {place} gave a spirited tragedy; {person} in the leading role drew warm applause.",
         "A new comedy opened at {place}; the playwright's wit pleased a full house.",
         "The stagecraft at {place} was uneven, though the actress carried the closing act."],
        "review"),
    "art_exhibition": (1830, 1930,
        ["exhibition", "painting", "gallery", "landscape", "portrait", "canvas", "salon", "sculpture"],
        ["Exhibition of paintings at {place}", "The annual salon at {place}",
         "New landscapes shown at {place}"],
        ["The gallery at {place} opened its salon; the landscapes of {person} were much admired.",
         "A large exhibition of portraits and sculpture drew crowds to {place}.",
         "Critics found the canvases at {place} bold in colour if uneven in composition."],
        "review"),
    "cholera_epidemic": (1830, 1895,
        ["cholera", "epidemic", "sanitary", "quarantine", "outbreak", "disease", "water", "mortality"],
        ["Cholera reported at {place}", "Sanitary measures against the epidemic",
         "The outbreak in {place}"],
        ["Cases of cholera appeared in {place}; the sanitary board ordered the wells at {place} closed.",
         "{person} warned that impure water spread the epidemic through the crowded quarters of {place}.",
         "Mortality from the outbreak rose sharply before quarantine was imposed at {place}."],
        "news"),
    "labor_strike": (1840, 1925,
        ["strike", "wages", "workers", "union", "factory", "labour", "mill", "dispute"],
        ["Strike at the {place} mills", "The labour dispute at {place}",
         "Workers demand higher wages at {place}"],
        ["The workers at the {place} mills struck for higher wages and shorter hours.",
         "{person} negotiated between the union and the factory owners at {place}.",
         "The labour dispute at {place} closed several works and idled hundreds of hands."],
        "news"),
    "emigration": (1845, 1915,
        ["emigration", "emigrants", "steamer", "passage", "settlers", "harbour", "overseas", "departure"],
        ["Emigrants depart from {place}", "The emigration question at {place}",
         "Overseas passage from {place}"],
        ["A crowd of emigrants boarded the steamer at {place}, bound for a new life overseas.",
         "{person} lamented the steady departure of young families from {place} for the Americas.",
         "The agent at {place} reported brisk demand for cheap passage across the ocean."],
        "news"),
    "archaeology_find": (1850, 1930,
        ["excavation", "antiquities", "inscription", "ruins", "Roman", "artefacts", "dig", "coins"],
        ["Roman remains uncovered near {place}", "Antiquities found at {place}",
         "The excavation at {place}"],
        ["Workmen near {place} uncovered a Roman inscription and a hoard of coins.",
         "{person} described the artefacts from the {place} dig to a learned society.",
         "The excavation at {place} laid bare the ruins of an ancient wall and pavement."],
        "news"),
    "printing_press": (1820, 1900,
        ["press", "printing", "newspaper", "type", "circulation", "pamphlet", "censorship", "edition"],
        ["A new printing press at {place}", "The press and censorship at {place}",
         "Circulation of the {place} paper"],
        ["A steam printing press was set up at {place}, greatly enlarging the newspaper's circulation.",
         "{person} protested the censorship of pamphlets printed at {place}.",
         "The new edition from the {place} press sold out within the day."],
        "news"),
}

# Topics a source never carries, so the gap analysis finds empty cells.
TOPIC_SOURCE_BLOCK = {
    "Barjanski Vestnik": {"art_exhibition", "railway_expansion", "printing_press",
                          "archaeology_find", "theatre_review"},
    "Murinsel-Post": {"womens_suffrage", "emigration"},
    # a review organ
    "The Calton Hill Review": {"railway_expansion", "labor_strike", "cholera_epidemic",
                               "emigration", "urban_poverty"},
    "Il Corriere dei Navigli": {"womens_suffrage"},
}


def make_text(topic: str, source: str) -> tuple[str, str, int, list[str]]:
    era_s, era_e, kws, titles, bodies, genre_default = TOPICS[topic]
    region, lang, _tier, genre_lean = SOURCES[source]
    year = rng.randint(era_s, era_e)
    place = rng.choice(PLACES[region])
    person = rng.choice(PERSONS)
    title = rng.choice(titles).format(place=place)
    body = " ".join(rng.sample(bodies, k=min(2, len(bodies)))).format(place=place, person=person)
    # extra keywords give BM25 something to match
    body += " Observers noted questions of " + " and ".join(rng.sample(kws, 2)) + "."
    genre = genre_lean if genre_lean in ("review", "editorial") else genre_default
    return title, body, year, [topic]


def emit(records: list[dict], source: str, topic: str, doc_id: str,
         derived_from: str | None = None, base: dict | None = None) -> dict:
    region, lang, _tier, _lean = SOURCES[source]
    if base is None:
        title, body, year, topics = make_text(topic, source)
    else:
        # a wire copy: same event, lightly reworded, different paper
        title = base["title"]
        body = base["text"].replace("finds", "reports").replace("described", "recounted")
        body = "Our correspondent relays: " + body
        year = base["year"]
        topics = base["topics"]
    genre = TOPICS[topic][5]
    _, _, _, lean = SOURCES[source]
    if lean in ("review", "editorial"):
        genre = lean
    month, day = rng.randint(1, 12), rng.randint(1, 28)
    rec = {
        "id": doc_id, "title": title, "text": body,
        "date": f"{year:04d}-{month:02d}-{day:02d}", "year": year,
        "source": source, "region": region, "language": lang,
        "genre": genre, "topics": topics, "derived_from": derived_from,
    }
    records.append(rec)
    return rec


def main() -> None:
    records: list[dict] = []
    n = 0
    for _ in range(430):
        topic = rng.choice(list(TOPICS))
        source = rng.choice(list(SOURCES))
        if topic in TOPIC_SOURCE_BLOCK.get(source, set()):
            continue
        n += 1
        emit(records, source, topic, f"doc-{n:04d}")

    # wire-copy clusters
    for c in range(8):
        topic = rng.choice(["cholera_epidemic", "labor_strike", "railway_expansion",
                            "womens_suffrage", "emigration"])
        origin_src = rng.choice(["The Thameside Chronicle", "Donau-Courier", "Ringstrassen-Bote"])
        n += 1
        origin_id = f"doc-{n:04d}"
        base = emit(records, origin_src, topic, origin_id)
        others = [s for s in SOURCES if s != origin_src
                  and topic not in TOPIC_SOURCE_BLOCK.get(s, set())]
        for src in rng.sample(others, k=rng.randint(2, 3)):
            n += 1
            emit(records, src, topic, f"doc-{n:04d}", derived_from=origin_id, base=base)

    rng.shuffle(records)
    with OUT.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} documents -> {OUT}")


if __name__ == "__main__":
    main()
