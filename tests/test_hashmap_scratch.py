"""Tests for issue #34 — Hash-map scratch for very-sparse queries.

Contract: replace the dense ``scratch[n_docs]`` buffer with an
open-addressed hash table keyed by doc-id when ``touched_rows ≪ n_docs``.
The hash-map path must produce **bitwise-identical** results to the
existing dense / sparse-reset paths (since BM25 score values are
identical sums regardless of accumulator data structure — addition
order is preserved per doc-id because all writes to a given key
land in the same hash slot).

The new path is gated by a heuristic on ``upper_bound = Σ col_lens``.
Below threshold → hashmap (cheap topk over populated entries),
above → existing dense / sparse-reset paths.

Two hidden debug kwargs let tests pin the path:
  - ``force_hashmap=True`` — always use the hashmap path (skip heuristic).
  - ``force_dense=True`` — always use the dense / sparse-reset paths
    (i.e. the pre-#34 behavior). Used as a parity oracle.

Failure modes the tests cover:

1. **Bitwise identity** between hashmap and dense across many
   (corpus, query) shapes — including k larger than the populated
   set, ties at the rank-k boundary, single-doc, single-token-query.
2. **Heuristic crossover** picks the right path automatically.
3. **Hash collisions** — engineered doc-id distribution that
   maximally collides under the Fibonacci hash. Correctness must
   survive long probe chains.
4. **Sizing edge case** — query touching MORE rows than the
   initial table capacity. Implementation must resize or fall back.
5. **Mixed-batch** — short and long queries interleaved; each query
   picks its own path.
6. **Parallel path** — same bitwise invariant on multi-worker dispatch.
7. **bm25s end-to-end parity** — full corpus + hashmap matches numpy.
"""

from __future__ import annotations

import numpy as np
import pytest

import bm25s

import mojo_bm25s


# ---------------------------------------------------------------------------
# Numpy reference oracle — full-zero scratch + scatter, no SIMD, no hash.
# Verbatim from test_simd_scatter.py. Same operation order as the mojo
# kernel: iterate query tokens, walk CSC column in indptr order,
# scatter-accumulate into scratch.
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
# Build a `bm25s.BM25` retriever whose `scores` dict is a hand-crafted CSC.
# ---------------------------------------------------------------------------


def _make_retriever_with_csc(data, indptr, indices, n_docs, n_vocab):
    placeholder = [[f"v{i}"] for i in range(n_vocab)]
    r = bm25s.BM25()
    r.index(placeholder, show_progress=False)
    r.scores = {
        "data": np.asarray(data, dtype=np.float32),
        "indptr": np.asarray(indptr, dtype=np.int32),
        "indices": np.asarray(indices, dtype=np.int32),
        "num_docs": int(n_docs),
    }
    return r


def _build_sparse_csc(col_lengths, n_docs, seed=0, unique_rows=True):
    """CSC with columns of given lengths. Rows drawn from [0, n_docs).
    If `unique_rows`, draw without replacement within a column so the
    scatter touches distinct rows (the canonical sparse case)."""
    rng = np.random.default_rng(seed)
    n_vocab = len(col_lengths)
    indptr = np.zeros(n_vocab + 1, dtype=np.int32)
    indptr[1:] = np.cumsum(col_lengths)
    total = int(indptr[-1])
    if unique_rows:
        # Each column gets distinct rows.
        rows = []
        for cl in col_lengths:
            if cl > n_docs:
                raise ValueError(
                    f"col_len {cl} > n_docs {n_docs}; can't draw unique"
                )
            rows.append(rng.choice(n_docs, size=cl, replace=False))
        indices = np.concatenate(rows).astype(np.int32)
    else:
        indices = rng.integers(0, n_docs, size=total, dtype=np.int32)
    data = (rng.integers(1, 1000, size=total).astype(np.float32) / 100.0)
    return data, indptr, indices, n_vocab


# ---------------------------------------------------------------------------
# 1. Bitwise identity: hashmap path == dense path across many shapes.
#
# Each parametrize entry is (col_lengths, n_docs, k, seed). All chosen so
# the upper-bound is < n_docs / 8 (hashmap regime). force_hashmap and
# force_dense are toggled to ensure we exercise both paths and compare
# them to the numpy oracle as a third witness.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "col_lengths,n_docs,k,seed",
    [
        # Single very-sparse query: 5 rows in a 10K-doc corpus.
        ([5], 10_000, 5, 0),
        ([5], 10_000, 10, 1),
        # Three short columns, ~50 touched total in 10K docs.
        ([20, 15, 18], 10_000, 10, 2),
        # Boundary: exactly 1 entry per column.
        ([1, 1, 1, 1], 1_000, 4, 3),
        # k > touched_rows — most slots fill with zero-scored entries.
        ([3, 4], 5_000, 50, 4),
        # k larger than n_docs is not legal (facade rejects), so cap k.
        # Wider but still under n_docs/8 = 125.
        ([100], 1_000, 20, 5),
        # Many short columns (15 cols × 8 = 120 < 125).
        ([8] * 15, 1_000, 10, 6),
    ],
)
def test_hashmap_path_bitwise_matches_dense(col_lengths, n_docs, k, seed):
    """Force-hashmap and force-dense must produce identical (scores, ids)
    AND both must match the numpy oracle."""
    data, indptr, indices, n_vocab = _build_sparse_csc(
        col_lengths, n_docs, seed=seed, unique_rows=True,
    )
    r = _make_retriever_with_csc(data, indptr, indices, n_docs, n_vocab)
    queries = [np.arange(n_vocab, dtype=np.int32)]

    scores_hm, ids_hm = mojo_bm25s.retrieve_batch(
        r, queries, k=k, num_workers=1, force_hashmap=True,
    )
    scores_d, ids_d = mojo_bm25s.retrieve_batch(
        r, queries, k=k, num_workers=1, force_dense=True,
    )
    ref_scores, _ = _numpy_retrieve_batch(
        data, indptr, indices, n_docs, queries, k=k,
    )

    np.testing.assert_array_equal(
        scores_hm, scores_d,
        err_msg="hashmap and dense paths diverged on scores",
    )
    np.testing.assert_array_equal(
        scores_hm, ref_scores,
        err_msg="hashmap path diverged from numpy oracle",
    )
    # IDs must agree only on positions with non-zero score (the rank-k
    # zero-tier ordering is implementation defined; tested separately).
    nz = scores_hm[0] > 0
    np.testing.assert_array_equal(
        ids_hm[0, nz], ids_d[0, nz],
        err_msg="hashmap and dense paths diverged on non-zero IDs",
    )


# ---------------------------------------------------------------------------
# 2. Heuristic crossover — make sure auto-pick chooses the right path.
#
# We can't directly observe path selection (it's internal). What we CAN
# do is exercise the heuristic boundary: a corpus where the heuristic
# clearly says "hashmap" vs "dense", and assert the auto-pick result
# matches force_hashmap (resp. force_dense). Bitwise identity across
# all three is the strongest invariant.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "col_lengths,n_docs,k,seed,expected_path",
    [
        # Very sparse: upper_bound = 5 << n_docs/8 = 1250. Hashmap.
        ([5], 10_000, 10, 10, "hashmap"),
        # Medium-sparse: upper_bound = 50 << 1250. Hashmap.
        ([50], 10_000, 10, 11, "hashmap"),
        # Dense: upper_bound = 2000 > 1250. Dense / sparse-reset.
        ([2000], 10_000, 10, 12, "dense"),
        # Very dense: upper_bound = 8000 > 1250. Dense.
        ([8000], 10_000, 10, 13, "dense"),
    ],
)
def test_heuristic_crossover_matches_forced(
    col_lengths, n_docs, k, seed, expected_path,
):
    """Auto-pick path on this corpus must match the manually-forced
    path's output — i.e. the heuristic should pick the right one."""
    # For "dense" cases we need duplicates allowed (col_len can exceed n_docs).
    unique_rows = max(col_lengths) <= n_docs
    data, indptr, indices, n_vocab = _build_sparse_csc(
        col_lengths, n_docs, seed=seed, unique_rows=unique_rows,
    )
    r = _make_retriever_with_csc(data, indptr, indices, n_docs, n_vocab)
    queries = [np.arange(n_vocab, dtype=np.int32)]

    # Auto-pick (no force kwarg).
    scores_auto, ids_auto = mojo_bm25s.retrieve_batch(
        r, queries, k=k, num_workers=1,
    )

    # Force the expected path.
    if expected_path == "hashmap":
        scores_forced, ids_forced = mojo_bm25s.retrieve_batch(
            r, queries, k=k, num_workers=1, force_hashmap=True,
        )
    else:
        scores_forced, ids_forced = mojo_bm25s.retrieve_batch(
            r, queries, k=k, num_workers=1, force_dense=True,
        )

    np.testing.assert_array_equal(
        scores_auto, scores_forced,
        err_msg=f"auto-pick diverged from force-{expected_path}",
    )


# ---------------------------------------------------------------------------
# 3. Hash collisions — engineer doc-ids that all hash to the same slot
# under Fibonacci hashing `(d * 0x9E3779B9) >> shift`. With `shift = 32 - log2(cap)`,
# all multiples of `2^shift / gcd` collide. Simplest: pick rows that are
# multiples of the table size so they all hash to slot 0 / `cap - cap/gcd`.
#
# Concrete construction: cap = next_pow2(touched * 2) for touched=50 is 128
# (shift=25). So rows whose `(row * 0x9E3779B9) & ((1<<32)-1) >> 25` all
# equal the same bucket. Easiest: rows that are multiples of 2^25 / 0x9E3779B9
# work approximately; in practice we just pick adversarial spacing and
# *measure* that probing happens (which we can't observe directly).
#
# What we CAN test: correctness under heavy probing. Trick: pick rows that
# stress collision behavior — sequential rows (0, 1, 2, ..., 49) hash well,
# but rows with a common factor cluster. We use rows that are all multiples
# of the table size, which guarantees collisions regardless of hash function.
# ---------------------------------------------------------------------------


def test_hashmap_correctness_under_engineered_collisions():
    """Pick doc-ids designed to all hash to the same bucket. The table
    must still produce correct sums via linear probing."""
    n_docs = 50_000
    # 50 rows, all multiples of 256 — under Fibonacci hash with cap 128
    # (shift=25), `(d << 8) * F >> 25` cycles through few buckets.
    # Any open-addressed table that's correct will sum them right.
    rows = np.array([i * 256 for i in range(50)], dtype=np.int32)
    data = np.array([1.0 + i * 0.5 for i in range(50)], dtype=np.float32)
    indptr = np.array([0, 50], dtype=np.int32)
    r = _make_retriever_with_csc(data, indptr, rows, n_docs, n_vocab=1)
    queries = [np.asarray([0], dtype=np.int32)]
    k = 50

    scores_hm, ids_hm = mojo_bm25s.retrieve_batch(
        r, queries, k=k, num_workers=1, force_hashmap=True,
    )
    ref_scores, _ = _numpy_retrieve_batch(
        data, indptr, rows, n_docs, queries, k=k,
    )
    np.testing.assert_array_equal(
        scores_hm, ref_scores,
        err_msg="hashmap collision case diverged from numpy oracle",
    )


def test_hashmap_correctness_with_repeated_rows():
    """Duplicate-row scatter in the hashmap: the same doc-id is written
    multiple times; the table must accumulate (insert OR update) correctly."""
    n_docs = 10_000
    # Row 7 appears 10 times with large values — must dominate.
    rows = np.array([7] * 10 + [42] * 3 + [99], dtype=np.int32)
    # Row 7 gets 10 * 5.0 = 50.0; row 42 gets 1.0 + 1.0 + 1.0 = 3.0;
    # row 99 gets 2.0. So row 7 is unambiguously top.
    data = np.array(
        [5.0] * 10 + [1.0] * 3 + [2.0], dtype=np.float32,
    )
    indptr = np.array([0, 14], dtype=np.int32)
    r = _make_retriever_with_csc(data, indptr, rows, n_docs, n_vocab=1)
    queries = [np.asarray([0], dtype=np.int32)]
    k = 5

    scores_hm, ids_hm = mojo_bm25s.retrieve_batch(
        r, queries, k=k, num_workers=1, force_hashmap=True,
    )
    ref_scores, ref_ids = _numpy_retrieve_batch(
        data, indptr, rows, n_docs, queries, k=k,
    )
    np.testing.assert_array_equal(
        scores_hm, ref_scores,
        err_msg="hashmap diverged on repeated-row scatter",
    )
    # Row 7 should be top (sum of 10 values = 50.0).
    assert ids_hm[0, 0] == 7
    assert scores_hm[0, 0] == 50.0


# ---------------------------------------------------------------------------
# 4. Sizing edge case — query touches MORE rows than the initial table
# capacity could comfortably hold. Implementation must either resize or
# the heuristic must never route such queries to the hashmap.
# We test by forcing the hashmap path on a query that touches a LOT of
# distinct rows; correctness must hold regardless of table sizing decisions.
# ---------------------------------------------------------------------------


def test_hashmap_handles_more_touched_rows_than_expected():
    """Force-hashmap on a query touching ~1000 distinct rows in a 10K
    corpus. Implementation must size the table correctly OR resize."""
    n_docs = 10_000
    col_lengths = [1000]
    data, indptr, indices, n_vocab = _build_sparse_csc(
        col_lengths, n_docs, seed=99, unique_rows=True,
    )
    r = _make_retriever_with_csc(data, indptr, indices, n_docs, n_vocab)
    queries = [np.asarray([0], dtype=np.int32)]
    k = 20

    scores_hm, _ = mojo_bm25s.retrieve_batch(
        r, queries, k=k, num_workers=1, force_hashmap=True,
    )
    ref_scores, _ = _numpy_retrieve_batch(
        data, indptr, indices, n_docs, queries, k=k,
    )
    np.testing.assert_array_equal(
        scores_hm, ref_scores,
        err_msg="hashmap diverged on large-touched query",
    )


# ---------------------------------------------------------------------------
# 5. Mixed batch — short and long queries interleaved.
# ---------------------------------------------------------------------------


def test_hashmap_mixed_batch():
    """Batch with one tiny (hashmap-candidate) and one large (dense)
    query. Auto-pick should route each correctly; force-hashmap on all
    must still produce correct results."""
    n_docs = 10_000
    # Token 0: 5 entries (hashmap regime — upper_bound=5 < 1250).
    # Token 1: 2000 entries (dense regime — upper_bound=2000 > 1250).
    col_lengths = [5, 2000]
    data, indptr, indices, n_vocab = _build_sparse_csc(
        col_lengths, n_docs, seed=23, unique_rows=False,
    )
    r = _make_retriever_with_csc(data, indptr, indices, n_docs, n_vocab)

    queries = [
        np.asarray([0], dtype=np.int32),       # hashmap
        np.asarray([1], dtype=np.int32),       # dense
        np.asarray([0, 1], dtype=np.int32),    # dense (2005 > 1250)
        np.asarray([0], dtype=np.int32),       # hashmap again
    ]
    k = 10

    scores_auto, _ = mojo_bm25s.retrieve_batch(
        r, queries, k=k, num_workers=1,
    )
    ref_scores, _ = _numpy_retrieve_batch(
        data, indptr, indices, n_docs, queries, k=k,
    )
    np.testing.assert_array_equal(
        scores_auto, ref_scores,
        err_msg="mixed batch (auto-pick) diverged from numpy oracle",
    )


# ---------------------------------------------------------------------------
# 6. Parallel path — same invariant must hold under multi-worker dispatch.
# ---------------------------------------------------------------------------


def test_parallel_matches_serial_bitwise_hashmap_path():
    """force_hashmap on parallel path must match serial bitwise."""
    n_docs = 10_000
    col_lengths = [10, 8, 6, 12]
    data, indptr, indices, n_vocab = _build_sparse_csc(
        col_lengths, n_docs, seed=31, unique_rows=True,
    )
    r = _make_retriever_with_csc(data, indptr, indices, n_docs, n_vocab)

    queries = [
        np.asarray([0, 1], dtype=np.int32),
        np.asarray([1, 2], dtype=np.int32),
        np.asarray([2, 3], dtype=np.int32),
        np.asarray([0, 3], dtype=np.int32),
        np.asarray([0], dtype=np.int32),
        np.asarray([3], dtype=np.int32),
    ]
    k = 8

    scores_serial, _ = mojo_bm25s.retrieve_batch(
        r, queries, k=k, num_workers=1, force_hashmap=True,
    )
    scores_parallel, _ = mojo_bm25s.retrieve_batch(
        r, queries, k=k, num_workers=4, force_hashmap=True,
    )
    np.testing.assert_array_equal(
        scores_serial, scores_parallel,
        err_msg="parallel hashmap path diverged from serial",
    )


# ---------------------------------------------------------------------------
# 7. bm25s end-to-end parity — full real bm25s indexing pipeline, then
# hashmap retrieval, then compare to numpy oracle.
# ---------------------------------------------------------------------------


def test_e2e_bm25s_parity_hashmap_path():
    """Real bm25s-indexed corpus; force_hashmap must match numpy oracle."""
    # Make a corpus large enough that single-token queries are sparse.
    base = [
        ["the", "quick", "brown", "fox"],
        ["jumps", "over", "the", "lazy", "dog"],
        ["pack", "my", "box", "with", "five", "dozen", "liquor", "jugs"],
        ["how", "vexingly", "quick", "daft", "zebras", "jump"],
        ["the", "five", "boxing", "wizards", "jump", "quickly"],
    ]
    # 500 docs — single-rare-term queries should be sparse.
    corpus = base * 100

    r = bm25s.BM25()
    r.index(corpus, show_progress=False)

    queries = [
        ["zebras"],     # rare, hashmap candidate
        ["wizards"],    # rare
        ["quick", "fox"],
        ["jump", "lazy"],
    ]
    token_id_queries = [
        np.asarray(r.get_tokens_ids(q), dtype=np.int32) for q in queries
    ]

    scores_hm, _ = mojo_bm25s.retrieve_batch(
        r, queries, k=5, num_workers=1, force_hashmap=True,
    )
    ref_scores, _ = _numpy_retrieve_batch(
        r.scores["data"],
        r.scores["indptr"],
        r.scores["indices"],
        int(r.scores["num_docs"]),
        token_id_queries,
        k=5,
    )
    np.testing.assert_array_equal(
        scores_hm, ref_scores,
        err_msg="bm25s-indexed corpus diverged on hashmap path",
    )


# ---------------------------------------------------------------------------
# 8. Argument validation — force_hashmap and force_dense are mutually
# exclusive; passing both should raise.
# ---------------------------------------------------------------------------


def test_force_kwargs_mutually_exclusive():
    """Passing both force_hashmap=True and force_dense=True is a
    programming error and should fail loudly."""
    n_docs = 100
    data, indptr, indices, n_vocab = _build_sparse_csc(
        [5], n_docs, seed=0, unique_rows=True,
    )
    r = _make_retriever_with_csc(data, indptr, indices, n_docs, n_vocab)
    queries = [np.asarray([0], dtype=np.int32)]

    with pytest.raises(ValueError, match="mutually exclusive"):
        mojo_bm25s.retrieve_batch(
            r, queries, k=3, num_workers=1,
            force_hashmap=True, force_dense=True,
        )
