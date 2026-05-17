"""Parity tests for the CSC sparse column-slice + dot accumulator kernel.

This is the retrieve hot path: given a CSC matrix (data, indices, indptr)
and a query as a list of token IDs, scatter-accumulate the column
entries into a per-doc score array.

The parity oracle is ``bm25s.scoring._compute_relevance_from_scores_legacy``.
The issue's acceptance section mentions ``_np_csc_python`` but that's the
CSC *construction* function, not the relevance computation — the issue's
Reference spec section correctly identifies ``_compute_relevance_from_scores_legacy``
as the byte-for-byte parity reference.

Both the legacy reference and the Mojo kernel accumulate column entries
in iteration order, so float32 results are bit-identical (no reordering
of additions).
"""

from __future__ import annotations

import numpy as np
import pytest

from bm25s.scoring import _compute_relevance_from_scores_legacy

import mojo_bm25s


def _reference(data, indptr, indices, n_docs, query_token_ids):
    return _compute_relevance_from_scores_legacy(
        data, indptr, indices, n_docs, query_token_ids, dtype=np.float32
    )


# ---------------------------------------------------------------------------
# Small handcrafted fixtures — exact byte equality.
# ---------------------------------------------------------------------------

def test_csc_score_minimal():
    """3 docs, 4 vocab tokens, 2-token query — exact float32 equality."""
    # vocab 0: rows 0,2 -> 1.0, 2.0
    # vocab 1: rows 1   -> 3.0
    # vocab 2: rows 0,1 -> 0.5, 1.5
    # vocab 3: (empty)
    data = np.array([1.0, 2.0, 3.0, 0.5, 1.5], dtype=np.float32)
    indices = np.array([0, 2, 1, 0, 1], dtype=np.int32)
    indptr = np.array([0, 2, 3, 5, 5], dtype=np.int32)

    query = np.array([0, 2], dtype=np.int32)
    n_docs = 3

    expected = _reference(data, indptr, indices, n_docs, query)
    got = mojo_bm25s.csc_score(data, indptr, indices, query, n_docs)

    assert got.dtype == np.float32
    assert got.shape == (n_docs,)
    # bit-equal: same accumulation order on same float32 inputs
    np.testing.assert_array_equal(got, expected)


def test_csc_score_empty_query():
    """No query tokens -> all-zero score vector."""
    data = np.array([1.0, 2.0], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    indptr = np.array([0, 2], dtype=np.int32)
    query = np.array([], dtype=np.int32)
    n_docs = 2

    got = mojo_bm25s.csc_score(data, indptr, indices, query, n_docs)
    expected = _reference(data, indptr, indices, n_docs, query)
    np.testing.assert_array_equal(got, expected)
    assert got.shape == (n_docs,)
    assert (got == 0).all()


def test_csc_score_query_hits_empty_column():
    """Query token whose column has zero entries contributes nothing."""
    # vocab 0: 1 entry; vocab 1: empty
    data = np.array([1.0], dtype=np.float32)
    indices = np.array([0], dtype=np.int32)
    indptr = np.array([0, 1, 1], dtype=np.int32)
    query = np.array([1, 0], dtype=np.int32)
    n_docs = 2

    expected = _reference(data, indptr, indices, n_docs, query)
    got = mojo_bm25s.csc_score(data, indptr, indices, query, n_docs)
    np.testing.assert_array_equal(got, expected)


def test_csc_score_repeated_query_token():
    """Same token twice in query accumulates twice."""
    data = np.array([2.5], dtype=np.float32)
    indices = np.array([0], dtype=np.int32)
    indptr = np.array([0, 1], dtype=np.int32)
    query = np.array([0, 0], dtype=np.int32)
    n_docs = 1

    expected = _reference(data, indptr, indices, n_docs, query)
    got = mojo_bm25s.csc_score(data, indptr, indices, query, n_docs)
    np.testing.assert_array_equal(got, expected)
    assert got[0] == np.float32(5.0)


def test_csc_score_single_token_query():
    """Length-1 query, len(indices) larger than typical simd_width."""
    n_docs = 11  # not aligned to 4/8
    data = np.arange(1, n_docs + 1, dtype=np.float32)
    indices = np.arange(n_docs, dtype=np.int32)
    indptr = np.array([0, n_docs], dtype=np.int32)
    query = np.array([0], dtype=np.int32)

    expected = _reference(data, indptr, indices, n_docs, query)
    got = mojo_bm25s.csc_score(data, indptr, indices, query, n_docs)
    np.testing.assert_array_equal(got, expected)


# ---------------------------------------------------------------------------
# Shape coverage — non-power-of-two doc/nnz counts.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_docs", [1, 7, 8, 9, 17, 33])
def test_csc_score_doc_counts(n_docs):
    """n_docs values that are below, equal-to, and not-divisible-by SIMD width."""
    rng = np.random.default_rng(42)
    n_vocab = 5
    # Every (vocab, doc) is filled — densest case, easy to reason about.
    data = rng.random((n_vocab * n_docs,), dtype=np.float32)
    indices = np.tile(np.arange(n_docs, dtype=np.int32), n_vocab)
    indptr = np.arange(n_vocab + 1, dtype=np.int32) * n_docs
    query = np.array([0, 2, 4], dtype=np.int32)

    expected = _reference(data, indptr, indices, n_docs, query)
    got = mojo_bm25s.csc_score(data, indptr, indices, query, n_docs)
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("nnz_per_col", [1, 3, 7, 8, 15, 16, 17])
def test_csc_score_col_lengths(nnz_per_col):
    """Column lengths around SIMD-block boundaries."""
    rng = np.random.default_rng(7)
    n_vocab = 4
    n_docs = 32
    nnz = n_vocab * nnz_per_col
    data = rng.random((nnz,), dtype=np.float32)
    # Distinct doc indices per column so accumulation order is deterministic
    indices = np.tile(
        rng.choice(n_docs, size=nnz_per_col, replace=False).astype(np.int32),
        n_vocab,
    )
    indptr = np.arange(n_vocab + 1, dtype=np.int32) * nnz_per_col
    query = np.arange(n_vocab, dtype=np.int32)

    expected = _reference(data, indptr, indices, n_docs, query)
    got = mojo_bm25s.csc_score(data, indptr, indices, query, n_docs)
    np.testing.assert_array_equal(got, expected)


# ---------------------------------------------------------------------------
# Realistic-scale parity: synthetic stand-in for the NQ subset the issue
# mentions. ≥1k docs, ≥10 queries — large enough to exercise the kernel
# without taking seconds in CI. The NQ-scale (10k docs, 100 queries)
# version lives in the microbenchmark, not the parity test suite.
# ---------------------------------------------------------------------------

def _synth_csc(n_docs: int, n_vocab: int, avg_df: int, seed: int):
    """Synthesize a CSC matrix shaped like a BM25 inverted index.

    Each token gets ``avg_df`` random doc IDs (no duplicates within a
    column); values are float32 in (0, 5).
    """
    rng = np.random.default_rng(seed)
    cols_indices = []
    cols_data = []
    indptr = [0]
    for _ in range(n_vocab):
        ndocs_this = max(1, int(rng.poisson(avg_df)))
        ndocs_this = min(ndocs_this, n_docs)
        idx = rng.choice(n_docs, size=ndocs_this, replace=False).astype(np.int32)
        idx.sort()  # CSC convention: row indices sorted within column
        vals = rng.uniform(0.01, 5.0, size=ndocs_this).astype(np.float32)
        cols_indices.append(idx)
        cols_data.append(vals)
        indptr.append(indptr[-1] + ndocs_this)
    return (
        np.concatenate(cols_data),
        np.concatenate(cols_indices),
        np.array(indptr, dtype=np.int32),
    )


def test_csc_score_realistic_scale():
    """1k docs, 500 vocab, ~50 entries/col, 20-token queries — bit-equal."""
    n_docs, n_vocab, avg_df = 1000, 500, 50
    data, indices, indptr = _synth_csc(n_docs, n_vocab, avg_df, seed=0)

    rng = np.random.default_rng(1)
    for _ in range(5):
        query = rng.choice(n_vocab, size=20, replace=True).astype(np.int32)
        expected = _reference(data, indptr, indices, n_docs, query)
        got = mojo_bm25s.csc_score(data, indptr, indices, query, n_docs)
        np.testing.assert_array_equal(got, expected)
