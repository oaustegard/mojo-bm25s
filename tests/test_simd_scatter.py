"""Tests for issue #19 — SIMD-W=8 CSC scatter loads in retrieve_batch_into.

The contract is correctness-preserving: lifting the dense-path inner
scatter loop from scalar to "SIMD load + scalar 8-lane store" must
produce bitwise-identical outputs to the pre-#19 scalar implementation.

The failure modes we test for:
1. Off-by-one on the scalar tail when `(col_end - col_start) % 8 != 0`.
   Exhaustively: column lengths {1, 7, 8, 9, 15, 16, 17, 31, 32, 33,
   64, 127, 128, 129}.
2. Last-column overrun: a `(indices + j).load[width=8]()` issued at the
   last column whose length is not a multiple of 8 must NOT read past
   the end of the indices buffer. If the buffer ends at `indptr[-1]`
   and the last column starts within 8 elements of that end, the SIMD
   load would over-read — but only the tail loop should be active in
   that case.
3. Duplicate-row scatter: a column whose `indices` array contains the
   same row twice must accumulate both contributions. Naive vectorized
   *writes* (scatter) without conflict detection would drop one of
   them; staying scalar on writes preserves correctness.
4. Both sparse and dense paths inside a single batch.

Strategy: hand-build CSC matrices with controlled column lengths,
inject into a `bm25s.BM25` retriever's `scores` dict, then assert
bitwise equality against a pure-numpy scatter+topk oracle.
"""

from __future__ import annotations

import numpy as np
import pytest

import bm25s

import mojo_bm25s


# ---------------------------------------------------------------------------
# Numpy reference oracle — full-zero scratch + scatter, no SIMD.
# Same operation order as the mojo kernel: iterate query tokens, for each
# walk the CSC column in indptr order, scatter-accumulate into scratch.
# ---------------------------------------------------------------------------


def _numpy_retrieve_batch(data, indptr, indices, n_docs, queries, k):
    scores_out = np.zeros((len(queries), k), dtype=np.float32)
    ids_out = np.zeros((len(queries), k), dtype=np.int32)
    for q_idx, q in enumerate(queries):
        scratch = np.zeros(n_docs, dtype=np.float32)
        for t in q:
            t = int(t)
            col_start = int(indptr[t])
            col_end = int(indptr[t + 1])
            for j in range(col_start, col_end):
                row = int(indices[j])
                scratch[row] += float(data[j])
        order = np.argsort(-scratch, kind="stable")[:k]
        scores_out[q_idx, : len(order)] = scratch[order]
        ids_out[q_idx, : len(order)] = order.astype(np.int32)
    return scores_out, ids_out


# ---------------------------------------------------------------------------
# Helpers to build a `bm25s.BM25` with hand-crafted CSC scores. We need
# a real retriever object so the Python facade in `retrieve_batch` can
# walk it; the only thing the kernel reads is `retriever.scores`, so we
# overwrite that dict after indexing a throwaway corpus.
# ---------------------------------------------------------------------------


def _make_retriever_with_csc(data, indptr, indices, n_docs, n_vocab):
    """Build a retriever whose `scores` dict is exactly our hand-built CSC."""
    # Index a throwaway corpus large enough to give us a vocab table the
    # facade can use; we only care that retrieve_batch can pass through.
    # The facade in __init__.py converts string queries via get_tokens_ids,
    # but we'll always pass int-id queries directly, so the vocab need
    # only have >= n_vocab entries.
    placeholder = [[f"v{i}"] for i in range(n_vocab)]
    r = bm25s.BM25()
    r.index(placeholder, show_progress=False)
    # Overwrite the scores with our hand-built CSC.
    r.scores = {
        "data": np.asarray(data, dtype=np.float32),
        "indptr": np.asarray(indptr, dtype=np.int32),
        "indices": np.asarray(indices, dtype=np.int32),
        "num_docs": int(n_docs),
    }
    return r


def _build_dense_csc(col_lengths, n_docs, seed=0):
    """CSC with one column per entry in `col_lengths`. Each column k holds
    `col_lengths[k]` entries with random rows in [0, n_docs) and random
    data values. Returns (data, indptr, indices, n_vocab)."""
    rng = np.random.default_rng(seed)
    n_vocab = len(col_lengths)
    indptr = np.zeros(n_vocab + 1, dtype=np.int32)
    indptr[1:] = np.cumsum(col_lengths)
    total = int(indptr[-1])
    # Random rows but each column's rows are unique-ish; for sparse vs
    # dense tests we want some duplicates so leave plain random.
    indices = rng.integers(0, n_docs, size=total, dtype=np.int32)
    # Use distinct, easily-spotted float values (1.0, 2.0, ...) for
    # bitwise comparisons that stay exact in float32.
    data = (rng.integers(1, 1000, size=total).astype(np.float32) / 100.0)
    return data, indptr, indices, n_vocab


# ---------------------------------------------------------------------------
# Tail-length sweep: exhaustively exercise every `(col_len % 8)` case.
# This is where off-by-one bugs in the scalar tail manifest. Forces the
# DENSE path by picking n_docs just larger than the column length so
# `expected_touched >= n_docs // 8`.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "col_len",
    [1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 127, 128, 129],
)
def test_dense_path_tail_lengths(col_len):
    """For each column length, build a single-vocab-token corpus where
    that column has exactly `col_len` entries, run retrieve_batch, and
    assert bitwise equality with the numpy reference."""
    n_docs = max(col_len + 4, 16)  # ensure n_docs >= col_len so rows fit
    # Build CSC with just one vocab token of length `col_len`.
    data, indptr, indices, n_vocab = _build_dense_csc([col_len], n_docs, seed=42)
    r = _make_retriever_with_csc(data, indptr, indices, n_docs, n_vocab)

    queries = [np.asarray([0], dtype=np.int32)]
    # k <= n_docs always.
    k = min(8, n_docs)

    mojo_scores, _ = mojo_bm25s.retrieve_batch(r, queries, k=k, num_workers=1)
    ref_scores, _ = _numpy_retrieve_batch(
        data, indptr, indices, n_docs, queries, k=k,
    )
    np.testing.assert_array_equal(
        mojo_scores, ref_scores,
        err_msg=f"col_len={col_len} mismatch on dense path tail",
    )


@pytest.mark.parametrize(
    "col_len",
    [1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 127, 128, 129],
)
def test_dense_path_tail_lengths_parallel(col_len):
    """Same sweep on the parallel path. Use multiple queries so workers
    actually get work."""
    n_docs = max(col_len + 4, 16)
    data, indptr, indices, n_vocab = _build_dense_csc([col_len], n_docs, seed=99)
    r = _make_retriever_with_csc(data, indptr, indices, n_docs, n_vocab)

    queries = [np.asarray([0], dtype=np.int32) for _ in range(4)]
    k = min(8, n_docs)

    mojo_scores, _ = mojo_bm25s.retrieve_batch(r, queries, k=k, num_workers=4)
    ref_scores, _ = _numpy_retrieve_batch(
        data, indptr, indices, n_docs, queries, k=k,
    )
    np.testing.assert_array_equal(
        mojo_scores, ref_scores,
        err_msg=f"col_len={col_len} parallel-path tail mismatch",
    )


# ---------------------------------------------------------------------------
# Last-column overrun: the buffer ends at indptr[-1]. If the SIMD load
# at the last column tries to read 8 ints starting at indptr[last] when
# `indptr[-1] - indptr[last] < 8`, it reads past the end of indices/data.
# The loop must use the `while j + W <= col_end` guard, not `while j < col_end`.
#
# Adversarial structure: a corpus whose only column has length 7 (so
# the tail-only path is hit) AND that column starts at offset 0
# (the only column). If the SIMD load runs, it overruns.
# We make this stronger by adding columns *before* the last one of
# exact length 8 (so the SIMD path IS taken there) and the last column
# of length not-multiple-of-8.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "last_col_len",
    [1, 2, 3, 4, 5, 6, 7],
)
def test_last_column_no_overrun(last_col_len):
    """Two-column CSC: first column 8 wide (full SIMD load), last
    column 1..7 wide (tail only). If the SIMD path runs on the last
    column it would over-read past indptr[-1]; if any other tail
    handling is wrong we get an out-of-bounds read or wrong sum."""
    n_docs = 32
    col_lengths = [8, last_col_len]
    data, indptr, indices, n_vocab = _build_dense_csc(
        col_lengths, n_docs, seed=7,
    )
    r = _make_retriever_with_csc(data, indptr, indices, n_docs, n_vocab)

    # Query token 1 = the short last column. Force dense path with k small.
    queries = [np.asarray([1], dtype=np.int32)]
    k = 5

    mojo_scores, _ = mojo_bm25s.retrieve_batch(r, queries, k=k, num_workers=1)
    ref_scores, _ = _numpy_retrieve_batch(
        data, indptr, indices, n_docs, queries, k=k,
    )
    np.testing.assert_array_equal(
        mojo_scores, ref_scores,
        err_msg=f"last-column len {last_col_len} mismatch / possible overrun",
    )


# ---------------------------------------------------------------------------
# Duplicate-row scatter: same row appears multiple times in one column,
# so the scatter must accumulate. If writes were vectorized (and they
# must NOT be in this PR), the conflicting writes would coalesce and
# drop contributions.
# ---------------------------------------------------------------------------


def test_duplicate_row_scatter_in_dense_path():
    """One column, 16 entries, all targeting row 0. Sum should be the
    full sum of `data[0..16]`. If any contribution is dropped, the test
    fails immediately."""
    n_docs = 8  # forces dense path (col_len 16 > 8//8 = 1)
    col_len = 16
    indptr = np.array([0, col_len], dtype=np.int32)
    indices = np.zeros(col_len, dtype=np.int32)  # ALL target row 0
    # Use small distinct values so float32 sum is exact.
    data = np.arange(1, col_len + 1, dtype=np.float32) / 4.0
    expected_sum = float(data.sum())

    r = _make_retriever_with_csc(data, indptr, indices, n_docs, n_vocab=1)
    queries = [np.asarray([0], dtype=np.int32)]
    mojo_scores, mojo_ids = mojo_bm25s.retrieve_batch(
        r, queries, k=n_docs, num_workers=1,
    )

    # Top score should be exactly the full sum on row 0.
    assert mojo_scores[0, 0] == pytest.approx(expected_sum, abs=0.0)
    assert mojo_ids[0, 0] == 0
    # Other rows should be 0.
    assert (mojo_scores[0, 1:] == 0.0).all()


def test_partial_duplicate_row_scatter_dense():
    """A column where the same row appears multiple times AND other
    rows appear once each — exercises the mixed case. 24 entries,
    row 0 appears 8 times, rows 1..16 once each."""
    n_docs = 24
    # Row 0 first 8 entries (1.0..8.0), then rows 1..16 (1.0 each).
    indices = np.concatenate([
        np.zeros(8, dtype=np.int32),
        np.arange(1, 17, dtype=np.int32),
    ])
    data = np.concatenate([
        np.arange(1, 9, dtype=np.float32),  # 1+2+...+8 = 36 on row 0
        np.ones(16, dtype=np.float32),       # 1 each on rows 1..16
    ])
    col_len = len(indices)
    indptr = np.array([0, col_len], dtype=np.int32)

    r = _make_retriever_with_csc(data, indptr, indices, n_docs, n_vocab=1)
    queries = [np.asarray([0], dtype=np.int32)]

    mojo_scores, _ = mojo_bm25s.retrieve_batch(
        r, queries, k=n_docs, num_workers=1,
    )
    ref_scores, _ = _numpy_retrieve_batch(
        data, indptr, indices, n_docs, queries, k=n_docs,
    )
    np.testing.assert_array_equal(
        mojo_scores, ref_scores,
        err_msg="duplicate-row mixed with unique-row scatter diverged",
    )
    # Sanity: top score is row 0 with sum = 36.
    assert mojo_scores[0, 0] == 36.0


# ---------------------------------------------------------------------------
# Mixed sparse + dense paths in one batch — make sure both code paths
# get exercised in the SAME retrieve_batch call.
# ---------------------------------------------------------------------------


def test_mixed_sparse_and_dense_paths_in_one_batch():
    """Build a corpus where:
      - vocab token 0 has a tiny column (forces sparse path)
      - vocab token 1 has a large column (forces dense path)
    Two queries in one batch: [0] hits sparse, [1] hits dense.
    """
    n_docs = 256  # dense threshold = 32
    # Token 0: 5 entries (5 < 32 -> sparse)
    # Token 1: 200 entries (200 > 32 -> dense)
    col_lengths = [5, 200]
    data, indptr, indices, n_vocab = _build_dense_csc(
        col_lengths, n_docs, seed=11,
    )
    r = _make_retriever_with_csc(data, indptr, indices, n_docs, n_vocab)

    queries = [
        np.asarray([0], dtype=np.int32),    # sparse
        np.asarray([1], dtype=np.int32),    # dense
        np.asarray([0, 1], dtype=np.int32), # 5+200=205 -> dense
    ]
    k = 10

    mojo_scores, _ = mojo_bm25s.retrieve_batch(r, queries, k=k, num_workers=1)
    ref_scores, _ = _numpy_retrieve_batch(
        data, indptr, indices, n_docs, queries, k=k,
    )
    np.testing.assert_array_equal(
        mojo_scores, ref_scores,
        err_msg="mixed sparse+dense batch diverged from numpy reference",
    )

    # Same on parallel path.
    mojo_scores_p, _ = mojo_bm25s.retrieve_batch(
        r, queries, k=k, num_workers=3,
    )
    np.testing.assert_array_equal(
        mojo_scores_p, ref_scores,
        err_msg="mixed sparse+dense parallel batch diverged",
    )


# ---------------------------------------------------------------------------
# Multi-column dense scatter — exercises the SIMD path running multiple
# times per query. Each query touches several long columns, so the SIMD
# inner loop runs many iterations, each with its own tail.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_multi_column_dense_scatter(seed):
    """Random query touching multiple long columns — bitwise parity
    with numpy reference."""
    n_docs = 512
    # 10 columns of varying lengths, all dense.
    col_lengths = [73, 64, 100, 65, 128, 33, 17, 95, 81, 129]
    data, indptr, indices, n_vocab = _build_dense_csc(
        col_lengths, n_docs, seed=seed,
    )
    r = _make_retriever_with_csc(data, indptr, indices, n_docs, n_vocab)

    queries = [
        np.asarray(list(range(n_vocab)), dtype=np.int32),
        np.asarray([0, 2, 4, 6, 8], dtype=np.int32),
        np.asarray([1, 3, 5, 7, 9], dtype=np.int32),
    ]
    k = 20

    mojo_scores, _ = mojo_bm25s.retrieve_batch(r, queries, k=k, num_workers=1)
    ref_scores, _ = _numpy_retrieve_batch(
        data, indptr, indices, n_docs, queries, k=k,
    )
    np.testing.assert_array_equal(
        mojo_scores, ref_scores,
        err_msg=f"multi-column dense (seed={seed}) diverged",
    )


# ---------------------------------------------------------------------------
# End-to-end bm25s parity sanity — re-assert against a real bm25s
# corpus that the kernel still matches bm25s after the change. This
# is also covered by tests/parity/test_vs_bm25s.py; pinning here makes
# the test file self-contained.
# ---------------------------------------------------------------------------


def test_e2e_bm25s_parity_smoke():
    """A small bm25s-indexed corpus; mojo retrieve_batch must match a
    numpy CSC scatter+topk on the same data."""
    corpus = [
        ["the", "quick", "brown", "fox"],
        ["jumps", "over", "the", "lazy", "dog"],
        ["pack", "my", "box", "with", "five", "dozen", "liquor", "jugs"],
        ["how", "vexingly", "quick", "daft", "zebras", "jump"],
        ["the", "five", "boxing", "wizards", "jump", "quickly"],
    ] * 20  # 100 docs

    r = bm25s.BM25()
    r.index(corpus, show_progress=False)

    queries = [
        ["quick", "fox"],
        ["jump", "lazy"],
        ["the"],
        ["zebras", "wizards"],
    ]
    token_id_queries = [
        np.asarray(r.get_tokens_ids(q), dtype=np.int32) for q in queries
    ]

    mojo_scores, _ = mojo_bm25s.retrieve_batch(r, queries, k=5, num_workers=1)
    ref_scores, _ = _numpy_retrieve_batch(
        r.scores["data"],
        r.scores["indptr"],
        r.scores["indices"],
        int(r.scores["num_docs"]),
        token_id_queries,
        k=5,
    )
    np.testing.assert_array_equal(
        mojo_scores, ref_scores,
        err_msg="bm25s-indexed corpus diverged from numpy reference",
    )
