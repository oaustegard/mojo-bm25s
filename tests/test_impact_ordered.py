"""Tests for impact-ordered postings + anytime retrieval (issue #35).

The contract:

- ``build_impact_ordered_index(corpus_token_ids, n_vocab, ...)`` returns
  the same tuple shape as ``build_index``, but each column
  ``[indptr[t], indptr[t+1])`` is sorted by **descending** ``data[j]``.
  This is a single per-column permutation of an otherwise-identical CSC.

- ``retrieve_batch_anytime(retriever_or_index_dict, queries, k, ...)``
  returns top-k results matching the scan-everything path within
  ``atol=1e-5`` on scores and IDs in the rank-k tie class. Bit-equality
  on indices is **not** required (impact order may use a different
  tie-break path through equal-score rows).

- Persistence: an impact-ordered index round-trips through ``save_index``
  / ``load_index`` with the layout preserved bit-for-bit, including a
  meta-flag distinguishing impact-ordered from doc-id-ordered.

- Early-exit: on a workload where the long tail genuinely cannot move
  top-k, the impact-ordered path visits strictly fewer (data, indices)
  entries than the scan-everything path. We expose a debug iteration
  counter the test reads.

Tests progress: API surface → permutation correctness → parity vs
scan-everything → bm25s end-to-end parity → persistence round-trip →
early-exit accounting → edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest

import bm25s
import mojo_bm25s


# ----------------------------------------------------------------------
# Fixtures and helpers
# ----------------------------------------------------------------------


def _build_vocab_and_ids(corpus_tokens):
    """Deterministic first-occurrence vocab + per-doc int32 id arrays."""
    vocab: dict[str, int] = {}
    ids: list[np.ndarray] = []
    for doc in corpus_tokens:
        arr = np.empty(len(doc), dtype=np.int32)
        for i, tok in enumerate(doc):
            if tok not in vocab:
                vocab[tok] = len(vocab)
            arr[i] = vocab[tok]
        ids.append(arr)
    return vocab, ids


def _make_corpus(n_docs, n_vocab, doc_len_range=(5, 30), seed=0):
    """Random-token-ID corpus with controllable shape."""
    rng = np.random.default_rng(seed)
    docs: list[np.ndarray] = []
    for _ in range(n_docs):
        L = int(rng.integers(doc_len_range[0], doc_len_range[1] + 1))
        docs.append(rng.integers(0, n_vocab, size=L, dtype=np.int32))
    return docs


def _make_queries(n_queries, n_vocab, query_len, seed=1):
    """Per-query int32 arrays of token IDs."""
    rng = np.random.default_rng(seed)
    return [
        rng.integers(0, n_vocab, size=query_len, dtype=np.int32)
        for _ in range(n_queries)
    ]


# ----------------------------------------------------------------------
# API surface
# ----------------------------------------------------------------------


def test_build_impact_ordered_index_is_exported():
    assert hasattr(mojo_bm25s, "build_impact_ordered_index"), (
        "mojo_bm25s.build_impact_ordered_index should be exported per issue #35"
    )
    assert callable(mojo_bm25s.build_impact_ordered_index)


def test_retrieve_batch_anytime_is_exported():
    assert hasattr(mojo_bm25s, "retrieve_batch_anytime"), (
        "mojo_bm25s.retrieve_batch_anytime should be exported per issue #35"
    )
    assert callable(mojo_bm25s.retrieve_batch_anytime)


# ----------------------------------------------------------------------
# Permutation correctness — the structural contract
# ----------------------------------------------------------------------


def test_impact_ordered_index_returns_same_tuple_shape():
    """Same 6-tuple shape as ``build_index`` so downstream consumers
    don't have to learn a new return type."""
    docs = _make_corpus(20, 30, seed=42)
    out = mojo_bm25s.build_impact_ordered_index(
        docs, n_vocab=30, method="lucene"
    )
    assert isinstance(out, tuple)
    assert len(out) == 6
    data, indices, indptr, n_docs, l_avg, nonoccurrence = out
    assert data.dtype == np.float32
    assert indices.dtype == np.int32
    assert indptr.dtype == np.int32
    assert n_docs == 20


def test_impact_ordered_each_column_descending_by_data():
    """The headline structural claim: within each column, ``data[j]`` is
    monotone non-increasing."""
    docs = _make_corpus(50, 25, seed=3)
    data, indices, indptr, n_docs, l_avg, _ = mojo_bm25s.build_impact_ordered_index(
        docs, n_vocab=25, method="lucene"
    )
    for t in range(25):
        col = data[indptr[t]:indptr[t + 1]]
        if col.size < 2:
            continue
        diffs = np.diff(col)
        assert np.all(diffs <= 1e-7), (
            f"column {t} not descending: data={col.tolist()}"
        )


def test_impact_ordered_preserves_data_multiset():
    """Reordering may not add or remove entries — same data values, same
    indices, just permuted within each column."""
    docs = _make_corpus(40, 20, seed=7)
    orig = mojo_bm25s.build_index(docs, n_vocab=20, method="lucene")
    impact = mojo_bm25s.build_impact_ordered_index(docs, n_vocab=20, method="lucene")
    # Same indptr layout (column sizes are unchanged).
    assert np.array_equal(orig[2], impact[2]), "indptr should be unchanged"
    # Same n_docs, l_avg.
    assert orig[3] == impact[3]
    assert orig[4] == pytest.approx(impact[4])
    # Per-column: same multiset of (index, data) pairs.
    for t in range(20):
        lo, hi = int(orig[2][t]), int(orig[2][t + 1])
        orig_pairs = sorted(
            zip(orig[1][lo:hi].tolist(), orig[0][lo:hi].tolist())
        )
        impact_pairs = sorted(
            zip(impact[1][lo:hi].tolist(), impact[0][lo:hi].tolist())
        )
        assert orig_pairs == impact_pairs, (
            f"column {t}: multiset of (index,data) pairs differs"
        )


def test_impact_ordered_hand_built_known_permutation():
    """Two-document corpus with hand-computed expected per-column order.

    Doc 0: token 0 appears 1x, length 3.
    Doc 1: token 0 appears 3x, length 5.
    For lucene, doc-1's token-0 score > doc-0's (higher tf, b-normalization
    factor not enough to flip), so column 0 should be [(doc=1, score_hi),
    (doc=0, score_lo)] after impact ordering — reversed from doc-id order.
    """
    docs = [
        np.array([0, 1, 2], dtype=np.int32),
        np.array([0, 0, 0, 3, 4], dtype=np.int32),
    ]
    orig = mojo_bm25s.build_index(docs, n_vocab=5, method="lucene")
    impact = mojo_bm25s.build_impact_ordered_index(docs, n_vocab=5, method="lucene")

    # Column 0 has 2 entries.
    t0_lo, t0_hi = int(orig[2][0]), int(orig[2][1])
    assert t0_hi - t0_lo == 2

    orig_indices = orig[1][t0_lo:t0_hi].tolist()
    orig_data = orig[0][t0_lo:t0_hi].tolist()
    impact_indices = impact[1][t0_lo:t0_hi].tolist()
    impact_data = impact[0][t0_lo:t0_hi].tolist()

    # Original is doc-id ascending.
    assert orig_indices == [0, 1]
    # Compute which doc has the higher impact.
    doc0_idx = orig_indices.index(0)
    doc1_idx = orig_indices.index(1)
    if orig_data[doc1_idx] > orig_data[doc0_idx]:
        expected = [1, 0]
        expected_data = [orig_data[doc1_idx], orig_data[doc0_idx]]
    else:
        expected = [0, 1]
        expected_data = [orig_data[doc0_idx], orig_data[doc1_idx]]
    assert impact_indices == expected, (
        f"col 0 impact order should be {expected} but got {impact_indices} "
        f"(data: orig={orig_data}, impact={impact_data})"
    )
    assert impact_data == pytest.approx(expected_data)


# ----------------------------------------------------------------------
# A retriever_like dict for retrieve_batch_anytime.
# ----------------------------------------------------------------------


class _MockRetriever:
    """Minimal retriever-like wrapper for use with ``retrieve_batch`` and
    ``retrieve_batch_anytime`` so tests don't have to drag in bm25s.

    Exposes ``.scores`` dict in the bm25s shape (with ``data``,
    ``indices``, ``indptr``, ``num_docs``). ``impact_ordered`` is
    surfaced as both a dict key (for anytime's dict-input path) and an
    attribute (for callers that prefer the attribute form).
    """
    def __init__(self, data, indices, indptr, n_docs, impact_ordered):
        self.scores = {
            "data": data,
            "indices": indices,
            "indptr": indptr,
            "num_docs": int(n_docs),
            "impact_ordered": impact_ordered,
        }
        self.impact_ordered = impact_ordered


def _index_dict_from_build(
    docs, n_vocab, method="lucene", impact_ordered=True
):
    """Build a retriever-like wrapper that both ``retrieve_batch`` and
    ``retrieve_batch_anytime`` can consume — exposes a bm25s-shaped
    ``.scores`` dict.
    """
    if impact_ordered:
        data, indices, indptr, n_docs, l_avg, _ = mojo_bm25s.build_impact_ordered_index(
            docs, n_vocab=n_vocab, method=method
        )
    else:
        data, indices, indptr, n_docs, l_avg, _ = mojo_bm25s.build_index(
            docs, n_vocab=n_vocab, method=method
        )
    return _MockRetriever(data, indices, indptr, n_docs, impact_ordered)


# ----------------------------------------------------------------------
# Parity vs scan-everything (the core correctness claim)
# ----------------------------------------------------------------------


def _scan_everything_scores_ids(
    docs, n_vocab, query_ids, k, method="lucene"
):
    """Reference scan-everything path: build a regular (doc-id-ordered)
    index, scatter through every entry, top-k."""
    data, indices, indptr, n_docs, _, _ = mojo_bm25s.build_index(
        docs, n_vocab=n_vocab, method=method
    )
    scores = mojo_bm25s.csc_score(
        data, indptr, indices, query_ids, n_docs=n_docs
    )
    return mojo_bm25s.topk(scores, k=min(k, n_docs))


def _ids_in_rank_k_tie_class(
    impact_ids, impact_scores, ref_ids, ref_scores,
    full_scores=None, atol=1e-5,
):
    """Returns True iff every id returned by impact-ordered has a true
    score at least as high as the rank-k boundary from the scan-
    everything reference. Tolerant of swaps within a tie.

    If ``full_scores`` (length n_docs) is supplied, picked ids are
    looked up there; otherwise we use the impact-side score (still
    valid: the score values are correct by the multiset assertion).
    """
    boundary = float(np.min(ref_scores))
    for i, picked_id in enumerate(impact_ids.tolist()):
        if full_scores is not None:
            picked_score = float(full_scores[picked_id])
        else:
            picked_score = float(impact_scores[i])
        if picked_score + atol < boundary:
            return False
    return True


@pytest.mark.parametrize("n_docs,n_vocab", [(50, 100), (500, 200)])
@pytest.mark.parametrize("query_len", [1, 2, 5, 20])
@pytest.mark.parametrize("k", [1, 10])
@pytest.mark.parametrize("method", ["lucene", "atire", "bm25l", "bm25+"])
def test_anytime_parity_with_scan_everything(
    n_docs, n_vocab, query_len, k, method
):
    """For every (corpus, query, k, method) combo: scores within atol,
    IDs within the rank-k tie class of the scan-everything reference."""
    if query_len > n_vocab:
        pytest.skip("query_len > n_vocab not meaningful")
    docs = _make_corpus(n_docs, n_vocab, seed=11)
    queries = _make_queries(8, n_vocab, query_len, seed=12)

    idx = _index_dict_from_build(
        docs, n_vocab, method=method, impact_ordered=True
    )
    anytime_scores, anytime_ids = mojo_bm25s.retrieve_batch_anytime(
        idx, queries, k=k
    )

    for qi, q in enumerate(queries):
        # Reference: scan-everything on the doc-id-ordered index.
        ref_data, ref_indices, ref_indptr, ref_n_docs, _, _ = (
            mojo_bm25s.build_index(docs, n_vocab=n_vocab, method=method)
        )
        ref_scores_vec = mojo_bm25s.csc_score(
            ref_data, ref_indptr, ref_indices, np.asarray(q, dtype=np.int32),
            n_docs=ref_n_docs,
        )
        kk = min(k, ref_n_docs)
        ref_scores_arr, ref_ids_arr = mojo_bm25s.topk(ref_scores_vec, k=kk)

        # Score-set must match within tolerance (multiset comparison).
        np.testing.assert_allclose(
            np.sort(anytime_scores[qi, :kk])[::-1],
            np.sort(ref_scores_arr)[::-1],
            atol=1e-5,
            err_msg=(
                f"method={method} n_docs={n_docs} n_vocab={n_vocab} "
                f"query_len={query_len} k={k} qi={qi}"
            ),
        )

        # IDs in the rank-k tie class — measured against the full per-doc
        # score vector (so we tolerate any swap among score-tie ids).
        assert _ids_in_rank_k_tie_class(
            anytime_ids[qi, :kk], anytime_scores[qi, :kk],
            ref_ids_arr, ref_scores_arr,
            full_scores=ref_scores_vec,
        ), (
            f"method={method} qi={qi}: ids {anytime_ids[qi, :kk].tolist()} "
            f"not in rank-k tie class of {ref_ids_arr.tolist()}"
        )


# ----------------------------------------------------------------------
# Larger sparse: 5K x 1K — proxy for medium-size realism.
# ----------------------------------------------------------------------


@pytest.mark.parametrize("query_len", [2, 5])
@pytest.mark.parametrize("k", [1, 10, 100])
def test_anytime_parity_medium_sparse(query_len, k):
    docs = _make_corpus(5000, 1000, doc_len_range=(20, 100), seed=21)
    queries = _make_queries(5, 1000, query_len, seed=22)

    idx_impact = _index_dict_from_build(docs, 1000, "lucene", impact_ordered=True)
    idx_doc = _index_dict_from_build(docs, 1000, "lucene", impact_ordered=False)

    s_any, i_any = mojo_bm25s.retrieve_batch_anytime(idx_impact, queries, k=k)
    s_ref, i_ref = mojo_bm25s.retrieve_batch(idx_doc, queries, k=k)

    for qi in range(len(queries)):
        np.testing.assert_allclose(
            np.sort(s_any[qi])[::-1],
            np.sort(s_ref[qi])[::-1],
            atol=1e-5,
            err_msg=f"query_len={query_len} k={k} qi={qi}",
        )


# ----------------------------------------------------------------------
# Synthetic large-sparse: 100K x 5K (#34 workload).
# ----------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("query_len,k", [(2, 10), (5, 10), (5, 100)])
def test_anytime_parity_large_sparse(query_len, k):
    docs = _make_corpus(100_000, 5000, doc_len_range=(20, 80), seed=31)
    queries = _make_queries(3, 5000, query_len, seed=32)

    idx_impact = _index_dict_from_build(docs, 5000, "lucene", impact_ordered=True)
    idx_doc = _index_dict_from_build(docs, 5000, "lucene", impact_ordered=False)

    s_any, i_any = mojo_bm25s.retrieve_batch_anytime(idx_impact, queries, k=k)
    s_ref, i_ref = mojo_bm25s.retrieve_batch(idx_doc, queries, k=k)

    for qi in range(len(queries)):
        np.testing.assert_allclose(
            np.sort(s_any[qi])[::-1],
            np.sort(s_ref[qi])[::-1],
            atol=1e-5,
            err_msg=f"large_sparse query_len={query_len} k={k} qi={qi}",
        )


# ----------------------------------------------------------------------
# End-to-end vs bm25s reference.
# ----------------------------------------------------------------------


def test_anytime_matches_bm25s_reference_small_fixture():
    """Compare against bm25s.BM25.retrieve() on a small fixture."""
    corpus_tokens = [
        ["the", "quick", "brown", "fox"],
        ["the", "lazy", "dog"],
        ["a", "quick", "fox", "jumps"],
        ["brown", "fox", "fox", "fox"],
        ["fast", "brown", "horse"],
        ["a", "lazy", "cat", "sleeps"],
    ]
    vocab_dict, corpus_ids = _build_vocab_and_ids(corpus_tokens)
    n_vocab = len(vocab_dict)

    # bm25s reference.
    retriever = bm25s.BM25(method="lucene")
    retriever.index(
        ([ids.tolist() for ids in corpus_ids], dict(vocab_dict)),
        create_empty_token=False, show_progress=False,
    )

    queries = [["quick", "fox"], ["brown", "fox"], ["lazy", "dog"]]
    query_ids = [
        np.asarray([vocab_dict[t] for t in q if t in vocab_dict], dtype=np.int32)
        for q in queries
    ]

    k = 3

    # Get bm25s reference scores by querying.
    ref_ids, ref_scores = retriever.retrieve(queries, k=k, show_progress=False)
    ref_ids = np.asarray(ref_ids)
    ref_scores = np.asarray(ref_scores)

    # Anytime impact-ordered retrieve.
    idx_impact = _index_dict_from_build(
        corpus_ids, n_vocab=n_vocab, method="lucene", impact_ordered=True
    )
    s_any, i_any = mojo_bm25s.retrieve_batch_anytime(idx_impact, query_ids, k=k)

    for qi in range(len(queries)):
        np.testing.assert_allclose(
            np.sort(s_any[qi])[::-1],
            np.sort(ref_scores[qi])[::-1],
            atol=1e-5,
            err_msg=f"bm25s parity qi={qi} q={queries[qi]}",
        )


# ----------------------------------------------------------------------
# Persistence round-trip.
# ----------------------------------------------------------------------


def test_persistence_round_trip_impact_ordered(tmp_path):
    """save → load preserves the impact-ordered layout bit-for-bit AND
    sets the ``impact_ordered`` flag on the loaded result."""
    from mojo_bm25s import save_index, load_index, Vocab

    corpus_tokens = [
        ["the", "quick", "brown", "fox"],
        ["the", "lazy", "dog", "sleeps"],
        ["a", "quick", "fox", "jumps", "over", "the", "lazy", "dog"],
        ["brown", "fox", "fox", "fox"],
    ]
    vocab = Vocab.from_corpus(corpus_tokens)
    ids = [vocab.tokens_to_ids(doc) for doc in corpus_tokens]
    data, indices, indptr, n_docs, l_avg, _ = mojo_bm25s.build_impact_ordered_index(
        ids, n_vocab=len(vocab), method="lucene"
    )
    target = tmp_path / "idx_impact"
    save_index(
        target,
        data=data, indices=indices, indptr=indptr,
        n_docs=n_docs, l_avg=l_avg, vocab=vocab,
        k1=1.5, b=0.75, delta=0.5,
        method="lucene", idf_method="lucene",
        impact_ordered=True,
    )
    loaded = load_index(target)
    # Byte-for-byte the same.
    assert np.array_equal(loaded.data, data)
    assert np.array_equal(loaded.indices, indices)
    assert np.array_equal(loaded.indptr, indptr)
    # The flag round-trips.
    assert getattr(loaded, "impact_ordered", False) is True


def test_persistence_backward_compat_doc_id_ordered_loads_unflagged(tmp_path):
    """An index saved WITHOUT the impact_ordered kwarg (or with False)
    loads as ``impact_ordered=False``. No silent flip."""
    from mojo_bm25s import save_index, load_index, Vocab

    corpus_tokens = [
        ["a", "b", "c"], ["b", "c", "d"], ["a", "d"],
    ]
    vocab = Vocab.from_corpus(corpus_tokens)
    ids = [vocab.tokens_to_ids(doc) for doc in corpus_tokens]
    data, indices, indptr, n_docs, l_avg, _ = mojo_bm25s.build_index(
        ids, n_vocab=len(vocab), method="lucene"
    )
    target = tmp_path / "idx_docid"
    save_index(
        target,
        data=data, indices=indices, indptr=indptr,
        n_docs=n_docs, l_avg=l_avg, vocab=vocab,
        k1=1.5, b=0.75, delta=0.5,
        method="lucene", idf_method="lucene",
    )
    loaded = load_index(target)
    # The default — old indexes load as doc-id-ordered.
    assert getattr(loaded, "impact_ordered", False) is False


# ----------------------------------------------------------------------
# Early-exit accounting.
# ----------------------------------------------------------------------


def test_early_exit_visits_fewer_entries_than_scan_everything():
    """On a corpus where the long tail genuinely cannot move top-k,
    impact-ordered must visit strictly fewer (data, indices) entries
    than scan-everything.

    Construction: token 0 is heavy-tailed — one doc dominates, the rest
    contribute trivially. k=1 + a single-term query gives a strong
    early-exit signal.
    """
    # 5000 docs (large enough that the in-term CHUNK boundary triggers
    # the threshold refresh; the column has 5000 entries so the
    # impact-ordered tail has plenty of room for early-exit). Token 0
    # appears in all of them. Doc 0 has it 50 times, all others once.
    # -> doc 0 dominates token-0's column; the long tail of 1-instance
    # docs contributes near-zero each.
    docs: list[np.ndarray] = []
    for i in range(5000):
        if i == 0:
            docs.append(np.array([0] * 50, dtype=np.int32))
        else:
            docs.append(np.array([0, 1, 2, 3, 4], dtype=np.int32))
    n_vocab = 10

    idx = _index_dict_from_build(docs, n_vocab, "lucene", impact_ordered=True)
    # Single-term query on token 0 — at k=1, after the first scatter the
    # heap holds doc 0 with the maximum possible score; everything after
    # must be < threshold so the remaining 999 entries should be skipped.
    queries = [np.array([0], dtype=np.int32)]

    counters = {}
    mojo_bm25s.retrieve_batch_anytime(
        idx, queries, k=1, _debug_iteration_counters=counters
    )
    visited = counters.get("entries_visited", None)
    # Without early-exit, scan-everything would visit ALL 1000 token-0
    # entries (sum of expected_touched). With early-exit we should visit
    # strictly fewer.
    assert visited is not None, (
        "retrieve_batch_anytime should populate "
        "_debug_iteration_counters['entries_visited'] when the kwarg is "
        "supplied"
    )
    # We don't know the exact count (depends on the threshold logic), but
    # we DO know it must be strictly less than the full column length.
    indptr = idx.scores["indptr"]
    col0_len = int(indptr[1] - indptr[0])
    assert visited < col0_len, (
        f"Early-exit should kick in: visited={visited} but column 0 has "
        f"{col0_len} entries — full scan would visit all of them."
    )


# ----------------------------------------------------------------------
# Edge cases.
# ----------------------------------------------------------------------


def test_empty_query_returns_zeros():
    docs = _make_corpus(20, 30, seed=51)
    idx = _index_dict_from_build(docs, 30, "lucene", impact_ordered=True)
    queries = [np.zeros(0, dtype=np.int32)]
    scores, ids = mojo_bm25s.retrieve_batch_anytime(idx, queries, k=5)
    assert scores.shape == (1, 5)
    np.testing.assert_allclose(scores[0], 0.0)


def test_single_token_query_safe():
    """A length-1 query — degenerate for the early-exit logic (no "other
    terms" upper bound) — must still produce correct top-k."""
    docs = _make_corpus(50, 20, seed=61)
    idx_impact = _index_dict_from_build(docs, 20, "lucene", impact_ordered=True)
    idx_doc = _index_dict_from_build(docs, 20, "lucene", impact_ordered=False)
    queries = [np.array([3], dtype=np.int32)]
    s_any, _ = mojo_bm25s.retrieve_batch_anytime(idx_impact, queries, k=10)
    s_ref, _ = mojo_bm25s.retrieve_batch(idx_doc, queries, k=10)
    np.testing.assert_allclose(
        np.sort(s_any[0])[::-1], np.sort(s_ref[0])[::-1], atol=1e-5
    )


def test_query_with_all_tied_impacts_is_safe():
    """Construct a corpus where every entry of a column has the same
    data value (all docs of the same length, all with the same tf for
    a token). Sort must be stable / not crash, and parity holds."""
    # All docs same length, every doc has token 0 exactly once.
    docs = [np.array([0, 1, 2, 3], dtype=np.int32) for _ in range(50)]
    n_vocab = 4
    idx_impact = _index_dict_from_build(docs, n_vocab, "lucene", impact_ordered=True)
    idx_doc = _index_dict_from_build(docs, n_vocab, "lucene", impact_ordered=False)
    queries = [np.array([0], dtype=np.int32), np.array([0, 1], dtype=np.int32)]
    s_any, _ = mojo_bm25s.retrieve_batch_anytime(idx_impact, queries, k=5)
    s_ref, _ = mojo_bm25s.retrieve_batch(idx_doc, queries, k=5)
    for qi in range(len(queries)):
        np.testing.assert_allclose(
            np.sort(s_any[qi])[::-1],
            np.sort(s_ref[qi])[::-1],
            atol=1e-5,
        )


def test_k_equals_1_threshold_grows_slowly_still_correct():
    """k=1 is the case where the threshold (= heap min) rises slowest,
    so early-exit happens latest. Must still produce correct results."""
    docs = _make_corpus(200, 50, seed=71)
    idx_impact = _index_dict_from_build(docs, 50, "lucene", impact_ordered=True)
    idx_doc = _index_dict_from_build(docs, 50, "lucene", impact_ordered=False)
    queries = _make_queries(10, 50, query_len=5, seed=72)

    s_any, _ = mojo_bm25s.retrieve_batch_anytime(idx_impact, queries, k=1)
    s_ref, _ = mojo_bm25s.retrieve_batch(idx_doc, queries, k=1)
    np.testing.assert_allclose(
        np.sort(s_any.flatten()),
        np.sort(s_ref.flatten()),
        atol=1e-5,
    )


def test_block_size_does_not_divide_nnz_evenly():
    """Many small columns with varied sizes — covers the "block size
    doesn't divide nnz" case (no special block alignment assumed)."""
    rng = np.random.default_rng(101)
    docs = []
    for _ in range(37):  # prime number of docs
        L = int(rng.integers(3, 13))
        docs.append(rng.integers(0, 17, size=L, dtype=np.int32))
    idx_impact = _index_dict_from_build(docs, 17, "lucene", impact_ordered=True)
    idx_doc = _index_dict_from_build(docs, 17, "lucene", impact_ordered=False)
    queries = [
        np.array([1, 4, 7], dtype=np.int32),
        np.array([0, 16], dtype=np.int32),
    ]
    s_any, _ = mojo_bm25s.retrieve_batch_anytime(idx_impact, queries, k=5)
    s_ref, _ = mojo_bm25s.retrieve_batch(idx_doc, queries, k=5)
    for qi in range(len(queries)):
        np.testing.assert_allclose(
            np.sort(s_any[qi])[::-1],
            np.sort(s_ref[qi])[::-1],
            atol=1e-5,
        )


# ----------------------------------------------------------------------
# Accepts both retriever and dict.
# ----------------------------------------------------------------------


def test_anytime_accepts_bm25s_retriever_directly():
    """The facade should accept a real bm25s.BM25 instance, transparently
    building the impact-ordered index from its CSC arrays."""
    corpus_tokens = [
        ["the", "quick", "brown", "fox"],
        ["a", "lazy", "dog"],
        ["the", "fast", "brown", "fox", "jumps"],
    ]
    retriever = bm25s.BM25(method="lucene")
    retriever.index(corpus_tokens, show_progress=False)

    queries = [["brown", "fox"], ["lazy", "dog"]]
    s_any, i_any = mojo_bm25s.retrieve_batch_anytime(retriever, queries, k=2)
    # Validate against the retriever's own scan-everything.
    s_ref, i_ref = mojo_bm25s.retrieve_batch(retriever, queries, k=2)
    for qi in range(len(queries)):
        np.testing.assert_allclose(
            np.sort(s_any[qi])[::-1],
            np.sort(s_ref[qi])[::-1],
            atol=1e-5,
        )
