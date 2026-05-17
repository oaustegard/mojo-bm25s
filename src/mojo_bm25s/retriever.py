"""Standalone ``Retriever`` — Phase 2 headline class (issue #27).

Owns the full BM25 lifecycle without requiring ``bm25s`` for inference:

    tokenize → vocab → build_index → (save/load) → retrieve

Composition, not reimplementation: every piece already exists as a free
function in this package (``tokenize``, ``Vocab.from_corpus``,
``build_index``, ``csc_score``, ``topk``, ``save_index`` / ``load_index``);
the Retriever wires them together and persists the configuration
required to reconstruct the pipeline on load.

Design choices locked by ``tests/test_retriever.py`` and
``tests/parity/test_retriever_standalone.py``:

- ``index`` returns ``self`` for chaining (`r.index(...).retrieve(...)`).
- ``retrieve`` returns ``(scores: float32[batch, k], ids: int32[batch, k])``
  sorted descending per row, identical contract to ``retrieve_batch``.
- Stemmer integration: a user-supplied ``Callable[[str], str]`` is
  applied to BOTH corpus AND query tokens (otherwise the query would
  miss every stemmed vocab entry).
- nonoccurrence handling for ``bm25l`` / ``bm25+``: the per-query
  ``nonoccurrence_array[query_ids].sum()`` is added back to every doc
  score before top-k. Mirrors ``bm25s.BM25.get_scores_from_ids``.
- OOV filtering at retrieve time: ``vocab.tokens_to_ids`` returns ``-1``
  for unknown tokens; we filter those out before handing the array to
  the Mojo CSC kernel (which doesn't bounds-check).
- Empty corpus is rejected with a clear ``ValueError``: there is no
  vocab to score against, so silently returning zeros forever would be
  a usability trap.

Save/load:

- ``save`` delegates to ``save_index`` (vocab + CSC + meta on disk).
- ``load`` is a classmethod that reads back into a new Retriever with
  the saved hyperparameters.
- The **stemmer** is the persistence gotcha: arbitrary user callables
  aren't picklable, so we persist a ``stemmer_name`` field in meta:
    - ``None`` — no stemmer was used
    - ``"porter"`` — the in-tree ``mojo_bm25s.stem`` (Porter-1980 /
      Snowball-English) was used; loader re-attaches it
    - any other identity (`PyStemmer`, custom lambda, etc.) — saved as
      ``"unknown"`` and the loader leaves the stemmer field as ``None``.
      The loaded retriever can still serve queries IF the query strings
      go through the same external stem step before hitting ``retrieve``;
      we don't enforce this because we can't detect it.

The "porter" round-trip is the supported case for fully self-contained
indexes; users with a PyStemmer-backed Retriever must re-attach the
stemmer themselves after ``load``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple, Union

import numpy as np

from . import csc_score, topk
from .tokenize import tokenize, ENGLISH_STOPWORDS
from .stem import stem as _porter_stem
from .vocab import Vocab
from .index_builder import build_index
from .io import save_index, load_index


# Sidecar file beside the standard save_index layout, holding Retriever-
# specific config (stemmer name) that isn't part of the bare-index meta.
_RETRIEVER_META_FILE = "retriever_meta.json"
_RETRIEVER_META_VERSION = 1


def _stemmer_name(stemmer: Optional[Callable[[str], str]]) -> str:
    """Identify a stemmer for save/load.

    ``None`` → ``"none"``; the in-tree ``mojo_bm25s.stem`` is identified
    by identity (``is`` comparison) and saved as ``"porter"``; anything
    else is ``"unknown"`` (loader will not re-attach a stemmer and the
    caller is on their own to do query-side stemming consistently).
    """
    if stemmer is None:
        return "none"
    if stemmer is _porter_stem:
        return "porter"
    return "unknown"


def _resolve_named_stemmer(name: str) -> Optional[Callable[[str], str]]:
    """Inverse of ``_stemmer_name``: resolve a saved name back to a
    callable. ``"none"`` and ``"unknown"`` both return ``None`` — the
    latter signals "we couldn't persist the original callable".
    """
    if name == "porter":
        return _porter_stem
    return None


class Retriever:
    """End-to-end BM25 retriever — index from raw strings, retrieve in float32.

    Parameters
    ----------
    k1, b, delta:
        BM25 hyperparameters. Defaults match ``bm25s.BM25`` (1.5, 0.75, 0.5).
    method:
        TFC variant: ``"robertson"``, ``"lucene"``, ``"atire"``,
        ``"bm25l"``, or ``"bm25+"``. Default ``"lucene"``.
    idf_method:
        IDF variant (same choices). Defaults to ``method`` when ``None``,
        matching ``bm25s.BM25``.
    stopwords:
        Forwarded to ``tokenize``. ``None`` for no filtering, ``"en"`` /
        ``"english"`` for the baked-in English list, an iterable of
        strings for a custom list. Defaults to ``ENGLISH_STOPWORDS``.
    stemmer:
        Optional callable applied per-token to both corpus and queries.
        For PyStemmer: pass ``Stemmer.Stemmer('english').stemWord``. For
        the in-tree Porter implementation: pass ``mojo_bm25s.stem``. See
        the save/load gotcha in the module docstring.
    """

    __slots__ = (
        "k1", "b", "delta", "method", "idf_method",
        "_stopwords", "_stemmer",
        "_vocab", "_data", "_indices", "_indptr",
        "_n_docs", "_l_avg", "_nonoccurrence",
        "_indexed",
    )

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 0.5,
        method: str = "lucene",
        idf_method: Optional[str] = None,
        stopwords: Optional[Iterable[str]] = ENGLISH_STOPWORDS,
        stemmer: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self.delta = float(delta)
        self.method = str(method)
        self.idf_method = str(idf_method) if idf_method is not None else self.method
        self._stopwords = stopwords  # opaque to us; tokenize() resolves it
        self._stemmer = stemmer

        # Lazily populated by .index() (or .load()).
        self._vocab: Optional[Vocab] = None
        self._data: Optional[np.ndarray] = None
        self._indices: Optional[np.ndarray] = None
        self._indptr: Optional[np.ndarray] = None
        self._n_docs: int = 0
        self._l_avg: float = 0.0
        self._nonoccurrence: Optional[np.ndarray] = None
        self._indexed: bool = False

    # ------------------------------------------------------------------
    # Tokenization helpers
    # ------------------------------------------------------------------

    def _tokenize(self, texts: List[str]) -> List[List[str]]:
        """Tokenize + optionally stem.

        Single code path for both corpus (``index``) and query
        (``retrieve``) — if it drifts, the query would miss the corpus.
        """
        # tokenize() does the regex-split + lowercase + stopword filter.
        toks_per_doc = tokenize(texts, stopwords=self._stopwords, lowercase=True)
        if self._stemmer is None:
            return toks_per_doc
        stem = self._stemmer
        return [[stem(t) for t in doc] for doc in toks_per_doc]

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(self, corpus: List[str]) -> "Retriever":
        """Tokenize, build the vocab, and build the CSC index.

        Returns ``self`` for chaining (``r.index(corpus).retrieve(...)``).

        Raises ``ValueError`` if ``corpus`` is empty — a zero-vocab
        retriever would silently produce zeros for every query forever,
        which is rarely what the caller actually wants. Build with a
        non-empty corpus or skip indexing entirely.

        Calling ``index`` twice REPLACES the existing index (does not
        append). All previously indexed docs are dropped.
        """
        if len(corpus) == 0:
            raise ValueError(
                "Cannot index an empty corpus. Pass at least one document "
                "(or build the index incrementally elsewhere and skip "
                "Retriever.index)."
            )

        # Tokenize + stem.
        corpus_tokens = self._tokenize(corpus)

        # Build deterministic first-occurrence vocab.
        vocab = Vocab.from_corpus(corpus_tokens)

        # Map every doc's tokens to int32 IDs (no -1's possible here —
        # every token came from the corpus we just built the vocab from).
        corpus_ids = [vocab.tokens_to_ids(doc) for doc in corpus_tokens]

        # Build the BM25-scored CSC matrix.
        data, indices, indptr, n_docs, l_avg, nonoccurrence = build_index(
            corpus_ids,
            n_vocab=len(vocab),
            method=self.method,
            idf_method=self.idf_method,
            k1=self.k1, b=self.b, delta=self.delta,
        )

        # Atomic install — only after build_index succeeds. If we crashed
        # mid-build, prior state is preserved (though build_index doesn't
        # raise on valid input so this is mostly hygiene).
        self._vocab = vocab
        self._data = data
        self._indices = indices
        self._indptr = indptr
        self._n_docs = int(n_docs)
        self._l_avg = float(l_avg)
        self._nonoccurrence = nonoccurrence
        self._indexed = True
        return self

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _query_to_ids(self, query: str) -> np.ndarray:
        """Tokenize + stem a single query string; map to int32 vocab IDs,
        filter out OOV (-1) entries.

        Returns an empty int32 array if the query is all OOV or empty.
        """
        # tokenize() takes a list; we have one string.
        tokens = self._tokenize([query])[0]
        if not tokens:
            return np.zeros(0, dtype=np.int32)
        ids = self._vocab.tokens_to_ids(tokens)
        # tokens_to_ids returns -1 for OOV; the CSC kernel doesn't bounds-
        # check, so we MUST strip them here.
        return ids[ids >= 0]

    def retrieve(
        self, queries: List[str], k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve top-``k`` documents for each query.

        Returns ``(scores, ids)`` of shape ``(len(queries), k)`` with
        dtypes ``float32`` / ``int32``. Per-row sorted descending by
        score. Queries with no in-vocab tokens get all-zero rows. ``k``
        larger than ``n_docs`` pads with zeros at rank > ``n_docs``.

        Raises ``RuntimeError`` if ``index`` has not been called yet.
        """
        if not self._indexed:
            raise RuntimeError(
                "Retriever has no index — call .index(corpus) first."
            )
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")

        n_q = len(queries)
        # Cap k at n_docs for the kernel call; pad the rest with zeros.
        # The Mojo topk asserts k <= input_length, and our score vector
        # has length n_docs.
        kernel_k = min(int(k), max(self._n_docs, 1))

        scores_out = np.zeros((n_q, int(k)), dtype=np.float32)
        ids_out = np.zeros((n_q, int(k)), dtype=np.int32)

        if self._n_docs == 0:
            # No docs to retrieve from. Already-zero output is correct.
            return scores_out, ids_out

        has_nonoccurrence = self._nonoccurrence is not None

        for qi, qstr in enumerate(queries):
            qids = self._query_to_ids(qstr)
            if qids.size == 0 and not has_nonoccurrence:
                # No in-vocab tokens and no per-query nonoccurrence
                # offset → uniform zero scores. The top-k of a constant
                # zero vector is just zeros at arbitrary indices; we
                # leave the pre-zeroed output row alone.
                continue

            # Base score vector from the CSC kernel (length n_docs).
            scores = csc_score(
                self._data, self._indptr, self._indices,
                qids if qids.size > 0 else np.zeros(0, dtype=np.int32),
                n_docs=self._n_docs,
            )

            # Add the per-query nonoccurrence offset for bm25l/bm25+.
            if has_nonoccurrence and qids.size > 0:
                # Mirrors bm25s.BM25.get_scores_from_ids: sum of
                # nonoccurrence[q] across all query token ids.
                offset = float(self._nonoccurrence[qids].sum())
                scores = scores + np.float32(offset)

            # Top-k via the Mojo kernel.
            row_scores, row_ids = topk(scores, k=kernel_k)
            scores_out[qi, :kernel_k] = row_scores
            ids_out[qi, :kernel_k] = row_ids
            # Trailing slots (rank >= n_docs when k > n_docs) stay at
            # (score=0, id=0) — same convention as bm25s pads.

        return scores_out, ids_out

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Persist the index + retriever config to ``path``.

        Writes the standard ``save_index`` layout (vocab.bin / data.bin /
        indices.bin / indptr.bin / meta.json / optional nonoccurrence.bin)
        plus a ``retriever_meta.json`` sidecar for the stemmer identity.

        Raises ``RuntimeError`` if there is no index to save.
        """
        if not self._indexed:
            raise RuntimeError(
                "Cannot save() a Retriever before index() has been called."
            )
        path = Path(path)
        save_index(
            path,
            data=self._data, indices=self._indices, indptr=self._indptr,
            n_docs=self._n_docs, l_avg=self._l_avg, vocab=self._vocab,
            k1=self.k1, b=self.b, delta=self.delta,
            method=self.method, idf_method=self.idf_method,
            nonoccurrence=self._nonoccurrence,
        )
        # save_index has written/replaced path; drop our sidecar in.
        sidecar = {
            "version": _RETRIEVER_META_VERSION,
            "stemmer_name": _stemmer_name(self._stemmer),
            # stopwords are stored only as a name hint — the actual
            # filtering already happened at index time, so what matters
            # for re-tokenizing queries is reproducing the same set.
            # We persist either "default" (None / ENGLISH_STOPWORDS) or
            # an explicit list. Anything more exotic (lambdas, generators)
            # would have been consumed at index time and we can't recover.
            "stopwords": _serialize_stopwords(self._stopwords),
        }
        (path / _RETRIEVER_META_FILE).write_text(
            json.dumps(sidecar, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Retriever":
        """Load a Retriever previously written by ``save``.

        Reconstructs hyperparameters, vocab, and CSC arrays from the
        saved index. Re-attaches the stemmer when it can be identified
        (``"porter"`` → ``mojo_bm25s.stem``); otherwise leaves the
        stemmer field as ``None`` and the caller is responsible for
        stemming queries consistently with how the corpus was stemmed.
        """
        path = Path(path)
        loaded = load_index(path)

        # Sidecar may be absent for an index produced by ``save_index``
        # directly (no Retriever wrapper); in that case default to "no
        # stemmer / default stopwords" — same as a fresh Retriever().
        sidecar_path = path / _RETRIEVER_META_FILE
        if sidecar_path.exists():
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            stemmer_name = sidecar.get("stemmer_name", "none")
            stopwords = _deserialize_stopwords(sidecar.get("stopwords", "default"))
        else:
            stemmer_name = "none"
            stopwords = ENGLISH_STOPWORDS

        stemmer = _resolve_named_stemmer(stemmer_name)

        r = cls(
            k1=loaded.k1, b=loaded.b, delta=loaded.delta,
            method=loaded.method, idf_method=loaded.idf_method,
            stopwords=stopwords, stemmer=stemmer,
        )
        # Install the loaded index directly — skip rebuilding.
        r._vocab = loaded.vocab
        r._data = loaded.data
        r._indices = loaded.indices
        r._indptr = loaded.indptr
        r._n_docs = loaded.n_docs
        r._l_avg = loaded.l_avg
        r._nonoccurrence = loaded.nonoccurrence
        r._indexed = True
        return r


# ----------------------------------------------------------------------
# Stopwords serialization for the sidecar
# ----------------------------------------------------------------------

def _serialize_stopwords(stopwords) -> Union[str, List[str]]:
    """Persist the stopwords config for round-trip.

    Returns ``"default"`` when the config is the in-tree default (None
    or the ENGLISH_STOPWORDS frozenset); a list of strings otherwise.
    """
    if stopwords is None:
        return "none"
    if stopwords is ENGLISH_STOPWORDS:
        return "default"
    if isinstance(stopwords, str):
        # "en" / "english" / etc — preserve as-is, tokenize() will resolve.
        return stopwords
    # Iterable of strings — materialize to a sorted list for determinism.
    try:
        return sorted(str(t) for t in stopwords)
    except Exception:
        # Defensive fallback: treat as default.
        return "default"


def _deserialize_stopwords(value):
    """Inverse of ``_serialize_stopwords``."""
    if value == "none":
        return None
    if value == "default":
        return ENGLISH_STOPWORDS
    return value  # str or list — both accepted by tokenize()
