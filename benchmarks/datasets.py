"""BEIR dataset loader.

Downloads from the bm25s release mirror
(``https://github.com/xhluca/bm25s/releases/download/data/<name>.zip``)
and caches under ``~/.cache/mojo-bm25s/datasets/<name>/``.

Each dataset zip unpacks to a BEIR-shaped directory:

    <name>/
      corpus.jsonl     # {"_id": str, "title": str, "text": str}
      queries.jsonl    # {"_id": str, "text": str}
      qrels/test.tsv   # query-doc-relevance ground truth (we don't use it)

Tokenization uses `bm25s.tokenize` (sklearn-style + PyStemmer English)
so the tokens we feed are exactly what bm25s's own test harness uses;
this keeps the parity tests honest.
"""

from __future__ import annotations

import json
import os
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import bm25s


BASE_URL = "https://github.com/xhluca/bm25s/releases/download/data/{}.zip"

CACHE_DIR = Path(
    os.environ.get("MOJO_BM25S_CACHE", "~/.cache/mojo-bm25s/datasets")
).expanduser()


@dataclass
class BeirDataset:
    """A loaded BEIR-shaped dataset.

    `corpus` is the raw doc texts (title + text concatenated, BEIR-style).
    `queries` is the raw query texts. `tokens` accessors lazily tokenize
    using bm25s's default English-stopwords + PyStemmer pipeline.
    """

    name: str
    corpus: list[str]
    queries: list[str]

    def corpus_tokens(self):
        import Stemmer
        stemmer = Stemmer.Stemmer("english")
        return bm25s.tokenize(
            self.corpus, stopwords="en", stemmer=stemmer, return_ids=False,
            show_progress=False,
        )

    def query_tokens(self):
        import Stemmer
        stemmer = Stemmer.Stemmer("english")
        return bm25s.tokenize(
            self.queries, stopwords="en", stemmer=stemmer, return_ids=False,
            show_progress=False,
        )


def _download(name: str, dest_zip: Path) -> None:
    url = BASE_URL.format(name)
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest_zip.with_suffix(".zip.partial")
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as fh:
        while True:
            chunk = response.read(65_536)
            if not chunk:
                break
            fh.write(chunk)
    tmp.rename(dest_zip)


def _ensure(name: str) -> Path:
    """Download + unzip ``name`` if not already cached. Returns the
    unpacked directory."""
    dataset_dir = CACHE_DIR / name
    marker = dataset_dir / ".ready"
    if marker.exists():
        return dataset_dir

    zip_path = CACHE_DIR / f"{name}.zip"
    if not zip_path.exists():
        _download(name, zip_path)

    dataset_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(CACHE_DIR)
    marker.touch()
    return dataset_dir


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_beir(
    name: str,
    *,
    corpus_subsample: int | None = None,
    queries_subsample: int | None = None,
) -> BeirDataset:
    """Load a BEIR dataset by name (e.g. ``"scifact"``, ``"trec-covid"``).

    Downloads + unzips on first call; cached thereafter. The subsample
    knobs trim the loaded lists in-order (deterministic, not random)
    so the parity tests can run on a smaller slice for speed.
    """
    root = _ensure(name)
    corpus_path = root / "corpus.jsonl"
    queries_path = root / "queries.jsonl"
    if not corpus_path.exists() or not queries_path.exists():
        # Some zips unpack into <name>/<name>/ — adjust.
        nested = root / name
        if (nested / "corpus.jsonl").exists():
            corpus_path = nested / "corpus.jsonl"
            queries_path = nested / "queries.jsonl"
        else:
            raise FileNotFoundError(
                f"corpus.jsonl not found under {root}; layout unexpected"
            )

    corpus_texts = []
    for row in _iter_jsonl(corpus_path):
        title = row.get("title", "")
        text = row.get("text", "")
        corpus_texts.append(f"{title} {text}".strip())

    query_texts = [row["text"] for row in _iter_jsonl(queries_path)]

    if corpus_subsample is not None:
        corpus_texts = corpus_texts[:corpus_subsample]
    if queries_subsample is not None:
        query_texts = query_texts[:queries_subsample]

    return BeirDataset(name=name, corpus=corpus_texts, queries=query_texts)
