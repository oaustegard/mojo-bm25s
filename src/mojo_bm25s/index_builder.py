"""Mojo-native CSC index builder (issue #25).

Replaces ``bm25s.BM25.index()`` for the standalone path. Given a list
of per-document int32 token-id arrays plus an explicit ``n_vocab``, we
build the CSC matrix ``(data, indices, indptr)`` whose entry ``data[j]``
is the BM25 contribution of a single ``(token, doc)`` match — exactly
the layout that ``mojo_bm25s.csc_score`` / ``retrieve_batch`` consumes
at query time.

Implementation choice: ``.py``, not ``.mojo``.
The hot per-element math (TFC, IDF) already lives in
``src/mojo_bm25s/scoring.mojo`` and is exposed as
``mojo_bm25s.score_tfc`` / ``score_idf_array``. What's left at index
time is bookkeeping (counting per-token doc frequencies, walking docs
in CSC column order, writing into output buffers). That's pure-Python
orchestration — calling into Mojo per token already amortizes the
PythonObject crossing across all docs that contain that token. A
``.mojo`` version would mostly shuffle the same lists across the FFI
boundary for no measurable win on index-time (which is not the hot
path; retrieve is, and that already runs in Mojo).

Parity contract (locked by ``tests/test_index_builder.py``):

- Numerical: ``data`` matches ``bm25s.BM25(method=..., idf_method=...)
  .index(corpus_ids, vocab_dict).scores["data"]`` within
  ``atol=1e-5`` (float32 accumulation noise).
- Structural: ``indices`` and ``indptr`` are exact-equal to bm25s's CSC
  layout. Within each column the doc IDs are monotone ascending.
- Nonoccurrence: for ``bm25l`` / ``bm25+``, the returned
  ``nonoccurrence_array`` matches ``bm25s.BM25.nonoccurrence_array``
  within ``atol=1e-5``; for other methods, returned as ``None``.
- Empty corpus → empty CSC (``data, indices`` empty; ``indptr`` all
  zeros of length ``n_vocab+1``); ``l_avg`` returned as 0.0.
- Vocab tokens never observed (``df = 0``) → empty column (zero-width
  in ``indptr``); their IDF entry is zero (bm25s convention).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from . import score_tfc, score_idf_array


# bm25s only computes a nonoccurrence_array for these two methods.
_METHODS_REQUIRING_NONOCCURRENCE = ("bm25l", "bm25+")


def _compute_nonoccurrence(
    method: str,
    idf_method: str,
    df: np.ndarray,
    n_docs: int,
    l_avg: float,
    k1: float,
    b: float,
    delta: float,
) -> np.ndarray:
    """Per-token scalar score for tokens absent from a doc.

    Mirrors ``bm25s.scoring._build_nonoccurrence_array``: for each
    token with ``df != 0``, compute ``idf * tfc(tf=0, l_d=l_avg,
    l_avg=l_avg, ...)``. Tokens with ``df == 0`` stay at 0.0.

    The ``l_d == l_avg`` is bm25s's own convention — see the function
    signature in scoring.py, where it passes ``l_d=avg_doc_len,
    l_avg=avg_doc_len``.
    """
    n_vocab = df.shape[0]
    out = np.zeros(n_vocab, dtype=np.float32)
    nonzero = df > 0
    if not np.any(nonzero):
        return out

    df_nonzero = df[nonzero]
    idf_nonzero = score_idf_array(idf_method, df_nonzero, float(n_docs))

    # tfc(tf=0, l_d=l_avg, l_avg=l_avg, ...): same scalar for every
    # nonzero-df token (it's not a function of df), so one call suffices.
    tfc_scalar_arr = score_tfc(
        method,
        np.zeros(1, dtype=np.float32),
        float(l_avg), float(l_avg), float(k1), float(b), float(delta),
    )
    tfc_scalar = float(tfc_scalar_arr[0])

    out[nonzero] = idf_nonzero * np.float32(tfc_scalar)
    return out


def build_index(
    corpus_token_ids: Sequence[np.ndarray],
    n_vocab: int,
    *,
    method: str = "lucene",
    idf_method: Optional[str] = None,
    k1: float = 1.5,
    b: float = 0.75,
    delta: float = 0.5,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, int, float, Optional[np.ndarray]
]:
    """Build a BM25-scored CSC index from per-document token-id arrays.

    Parameters
    ----------
    corpus_token_ids:
        Sequence of per-document int32 arrays of token IDs. Empty
        sequence is allowed (returns an empty CSC).
    n_vocab:
        Vocabulary size. ``indptr`` has shape ``(n_vocab + 1,)``. Tokens
        that never occur in the corpus (``df == 0``) result in
        zero-width columns and zero IDF (matches bm25s).
    method:
        TFC method: ``"robertson"``, ``"lucene"``, ``"atire"``,
        ``"bm25l"``, or ``"bm25+"``. Default ``"lucene"`` (matches
        ``bm25s.BM25``).
    idf_method:
        IDF method (same choices). If ``None``, defaults to ``method``
        (matches ``bm25s.BM25``).
    k1, b, delta:
        BM25 hyperparameters. Defaults match ``bm25s.BM25`` (1.5, 0.75,
        0.5).

    Returns
    -------
    tuple
        ``(data, indices, indptr, n_docs, l_avg, nonoccurrence)`` where:

        - ``data`` is float32 of shape ``(nnz,)``.
        - ``indices`` is int32 of shape ``(nnz,)`` — doc IDs.
        - ``indptr`` is int32 of shape ``(n_vocab + 1,)``.
        - ``n_docs`` is the number of documents.
        - ``l_avg`` is the mean document length (float; 0.0 for empty
          corpus).
        - ``nonoccurrence`` is float32 of shape ``(n_vocab,)`` for
          ``method in ("bm25l", "bm25+")``, otherwise ``None``.
    """
    if idf_method is None:
        idf_method = method

    n_vocab = int(n_vocab)
    n_docs = len(corpus_token_ids)

    # ------------------------------------------------------------------
    # Empty corpus: nothing to compute. Return canonical empty layout.
    # ------------------------------------------------------------------
    if n_docs == 0:
        return (
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.int32),
            np.zeros(n_vocab + 1, dtype=np.int32),
            0,
            0.0,
            None,
        )

    # ------------------------------------------------------------------
    # Doc lengths + l_avg
    # ------------------------------------------------------------------
    l_d = np.fromiter(
        (len(doc) for doc in corpus_token_ids),
        dtype=np.int32, count=n_docs,
    )
    l_avg = float(l_d.mean())

    # ------------------------------------------------------------------
    # Build per-token postings: token_id -> list[(doc_id, tf)].
    # We iterate docs in order, so within each token's posting list the
    # doc_ids are already monotone ascending — which is exactly the CSC
    # column ordering bm25s produces (see _np_csc_jit_ready: counting
    # sort over docs that were appended in doc order). No re-sort needed.
    # ------------------------------------------------------------------
    postings: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for doc_id, ids in enumerate(corpus_token_ids):
        if len(ids) == 0:
            continue
        # Count tokens within this doc. Using np.unique with return_counts
        # is faster than Counter for typical doc sizes and keeps the dtype
        # in numpy land.
        u, counts = np.unique(np.asarray(ids, dtype=np.int32), return_counts=True)
        for tok_id, tf in zip(u.tolist(), counts.tolist()):
            postings[int(tok_id)].append((doc_id, int(tf)))

    # ------------------------------------------------------------------
    # Document frequencies (one entry per token in vocab)
    # ------------------------------------------------------------------
    df = np.zeros(n_vocab, dtype=np.float32)
    for tok_id, plist in postings.items():
        if 0 <= tok_id < n_vocab:
            df[tok_id] = len(plist)

    # ------------------------------------------------------------------
    # Vocab-wide IDF lookup table
    # ------------------------------------------------------------------
    # Mirror bm25s exactly: tokens with df == 0 get IDF = 0 (not the
    # natural score_idf value, which can be log(N) or even -inf). We do
    # this by zeroing those entries post-hoc — score_idf_array is dense.
    idf_array = score_idf_array(idf_method, df, float(n_docs))
    if np.any(df == 0):
        idf_array[df == 0] = 0.0

    # ------------------------------------------------------------------
    # CSC assembly — walk vocab in token-id order, write contiguous
    # blocks of (data, indices) per column, accumulate indptr cumulative.
    # ------------------------------------------------------------------
    nnz = int(df.sum())  # df sums to total # of (token, doc) pairs
    data = np.empty(nnz, dtype=np.float32)
    indices = np.empty(nnz, dtype=np.int32)
    indptr = np.zeros(n_vocab + 1, dtype=np.int32)

    # Optional nonoccurrence array (subtracted into stored scores for
    # bm25l / bm25+ so retrieve can sum-then-add-back).
    needs_nonoccurrence = method in _METHODS_REQUIRING_NONOCCURRENCE
    if needs_nonoccurrence:
        nonoccurrence = _compute_nonoccurrence(
            method, idf_method, df, n_docs, l_avg, k1, b, delta,
        )
    else:
        nonoccurrence = None

    cursor = 0
    for tok_id in range(n_vocab):
        plist = postings.get(tok_id)
        if not plist:
            indptr[tok_id + 1] = cursor
            continue

        n_entries = len(plist)
        # Unpack postings: doc_ids are already in ascending order because
        # we appended in doc_id order during the corpus walk.
        doc_ids = np.fromiter(
            (d for d, _ in plist), dtype=np.int32, count=n_entries,
        )
        tfs = np.fromiter(
            (t for _, t in plist), dtype=np.float32, count=n_entries,
        )
        doc_lens = l_d[doc_ids].astype(np.float32, copy=False)

        # Per-doc TFC: one Mojo call per (token, l_d). bm25s passes the
        # whole tf_array with a scalar l_d, then iterates docs in an
        # outer loop. We invert that: we already grouped by token, so we
        # need per-doc l_d. Call the kernel once per entry — same number
        # of crossings as bm25s did per-token in its outer loop.
        #
        # Actually we can do better: score_tfc takes a scalar l_d but
        # the kernel applies it across the whole tf_array. Since each
        # (token, doc) pair has its own l_d we *can't* batch by token.
        # But we *can* batch by (token, l_d) — docs of the same length
        # share l_d. Quick win only for very repetitive corpora; for now,
        # keep it simple (one call per entry). The orchestrator will
        # measure first.
        block = np.empty(n_entries, dtype=np.float32)
        for j in range(n_entries):
            tfc_arr = score_tfc(
                method,
                tfs[j:j + 1],  # length-1 view
                float(doc_lens[j]), float(l_avg),
                float(k1), float(b), float(delta),
            )
            block[j] = tfc_arr[0]

        # data[j] = tfc * idf[token]; bm25s also subtracts the
        # nonoccurrence for bm25l/bm25+ here so retrieve can simply sum
        # and add back the per-query nonoccurrence sum.
        idf_t = np.float32(idf_array[tok_id])
        block *= idf_t
        if needs_nonoccurrence:
            block -= np.float32(nonoccurrence[tok_id])

        data[cursor:cursor + n_entries] = block
        indices[cursor:cursor + n_entries] = doc_ids
        cursor += n_entries
        indptr[tok_id + 1] = cursor

    return data, indices, indptr, n_docs, l_avg, nonoccurrence
