"""Tests for the touched-rows sparse scratch reset (issue #21).

Contract: replacing the full ``n_docs``-wide scratch zero with a
"reset only the rows we touched" pass MUST preserve bitwise output
parity with the full-zero implementation that existed before. These
tests are the gate — they're written so they fail only when the
sparse reset misses a row (i.e. leftover state bleeds into the next
query or the next batch).

The sparse-reset specific failure mode is:
    query A scatters into row R; we forget to reset row R; query B
    that does NOT touch row R sees A's residual score on row R.

So every test below sets up that adversarial pattern: queries that
deliberately do not share any vocabulary, so the only way a non-zero
score can appear on a row in query B's scratch is via leftover from
query A.
"""

from __future__ import annotations

import numpy as np
import pytest

import bm25s

import mojo_bm25s


# ---------------------------------------------------------------------------
# Reference path: numpy CSC scatter + topk. Used as the "ground truth"
# bitwise oracle for the mojo retrieve_batch output.
#
# This is intentionally *not* via mojo's per-query path — we want a
# computation that doesn't share code with the implementation we're
# changing, so a bug in the new sparse-reset can't accidentally pass.
# ---------------------------------------------------------------------------


def _numpy_retrieve_batch(retriever, queries_token_ids, k):
    """Reference: numpy full-zero CSC scatter + topk per query."""
    data = np.asarray(retriever.scores["data"], dtype=np.float32)
    indptr = np.asarray(retriever.scores["indptr"], dtype=np.int32)
    indices = np.asarray(retriever.scores["indices"], dtype=np.int32)
    n_docs = int(retriever.scores["num_docs"])

    scores_out = np.zeros((len(queries_token_ids), k), dtype=np.float32)
    ids_out = np.zeros((len(queries_token_ids), k), dtype=np.int32)

    for q_idx, q in enumerate(queries_token_ids):
        scratch = np.zeros(n_docs, dtype=np.float32)
        for t in q:
            t = int(t)
            col_start = int(indptr[t])
            col_end = int(indptr[t + 1])
            for j in range(col_start, col_end):
                row = int(indices[j])
                scratch[row] += float(data[j])
        # Heap-based topk that matches the mojo behavior on the rank-k
        # boundary tie class — sort by (-score, index) ascending so we
        # break ties by lower index. Note: bitwise equality is asserted
        # on *scores only* below; ID equality is tested separately and
        # tolerates the rank-k tie class.
        order = np.argsort(-scratch, kind="stable")[:k]
        scores_out[q_idx, : len(order)] = scratch[order]
        ids_out[q_idx, : len(order)] = order.astype(np.int32)
    return scores_out, ids_out


# ---------------------------------------------------------------------------
# Adversarial corpus + query construction.
# ---------------------------------------------------------------------------


def _build_disjoint_corpus(n_docs: int = 200, seed: int = 0):
    """Build a corpus where docs partition into two halves, each with
    its own private vocabulary. Then queries on the "left" vocab touch
    only left-half doc rows; queries on the "right" vocab touch only
    right-half doc rows. If the sparse reset is buggy, a left query
    followed by a right query will see left scores on left-half rows
    (which the right query should never touch)."""
    rng = np.random.default_rng(seed)
    left_vocab = [f"L{i}" for i in range(20)]
    right_vocab = [f"R{i}" for i in range(20)]

    corpus = []
    for i in range(n_docs):
        if i < n_docs // 2:
            size = int(rng.integers(3, 8))
            doc = [left_vocab[int(j)] for j in rng.integers(0, len(left_vocab), size=size)]
        else:
            size = int(rng.integers(3, 8))
            doc = [right_vocab[int(j)] for j in rng.integers(0, len(right_vocab), size=size)]
        corpus.append(doc)
    r = bm25s.BM25()
    r.index(corpus, show_progress=False)
    return r, left_vocab, right_vocab


def _build_large_sparse(n_docs: int = 5000, vocab_size: int = 400, seed: int = 7):
    """Larger corpus to exercise the case where most rows are NOT
    touched by a query — this is where the sparse reset wins."""
    rng = np.random.default_rng(seed)
    vocab = [f"w{i}" for i in range(vocab_size)]
    corpus = []
    for _ in range(n_docs):
        size = int(rng.integers(3, 10))
        doc = [vocab[int(j)] for j in rng.integers(0, len(vocab), size=size)]
        corpus.append(doc)
    r = bm25s.BM25()
    r.index(corpus, show_progress=False)
    return r, vocab


# ---------------------------------------------------------------------------
# The load-bearing test: adversarial leak detection.
# ---------------------------------------------------------------------------


def test_no_score_leak_between_disjoint_queries():
    """The adversarial leak test. Query A touches only left-half rows;
    query B touches only right-half rows. If sparse reset misses any
    of A's touched rows, query B's output for those rows will be
    non-zero — but the numpy reference (which always full-zeros) will
    show zero score on those rows for query B.

    Bitwise equality with the numpy reference catches the leak."""
    r, left_vocab, right_vocab = _build_disjoint_corpus()

    queries = [
        [left_vocab[0], left_vocab[1], left_vocab[2]],   # touches left half
        [right_vocab[0], right_vocab[1]],                 # touches right half
        [left_vocab[5]],                                  # left half again
        [right_vocab[10], right_vocab[11]],               # right half again
    ]
    token_id_queries = [np.asarray(r.get_tokens_ids(q), dtype=np.int32) for q in queries]

    mojo_scores, mojo_ids = mojo_bm25s.retrieve_batch(r, queries, k=10, num_workers=1)
    ref_scores, _ = _numpy_retrieve_batch(r, token_id_queries, k=10)

    np.testing.assert_array_equal(
        mojo_scores, ref_scores,
        err_msg="sparse-reset leaked scores between disjoint queries",
    )


def test_cross_batch_no_state_leak():
    """Two separate retrieve_batch calls on the same retriever, with
    disjoint query vocabularies. If the second batch sees leftover
    state from the first batch's scratch (e.g. allocation persisted
    via an attribute), the second batch's scores will not match the
    numpy reference.

    The current implementation allocates scratch per-call so this
    test passes today — but ANY future caching attempt (which the
    issue's design opens the door to) must respect the same contract.
    """
    r, left_vocab, right_vocab = _build_disjoint_corpus()

    batch_a = [
        [left_vocab[0], left_vocab[1]],
        [left_vocab[2], left_vocab[3]],
    ]
    batch_b = [
        [right_vocab[0], right_vocab[1]],
        [right_vocab[2]],
    ]

    # Run A first.
    mojo_bm25s.retrieve_batch(r, batch_a, k=5, num_workers=1)
    # Then B; compare to numpy reference.
    mojo_scores_b, _ = mojo_bm25s.retrieve_batch(r, batch_b, k=5, num_workers=1)
    ref_scores_b, _ = _numpy_retrieve_batch(
        r,
        [np.asarray(r.get_tokens_ids(q), dtype=np.int32) for q in batch_b],
        k=5,
    )
    np.testing.assert_array_equal(
        mojo_scores_b, ref_scores_b,
        err_msg="second batch contaminated by first batch's scratch",
    )


def test_empty_query_resets_cleanly():
    """An empty query in the middle of a batch must not freeze the
    scratch state — the next non-empty query must see a clean scratch.

    Adversarial setup: first query touches left-half rows, second
    query is empty, third query touches right-half rows. If the empty
    query somehow skipped the reset, the third query would see the
    first's residuals on left-half rows.
    """
    r, left_vocab, right_vocab = _build_disjoint_corpus()

    queries = [
        [left_vocab[0], left_vocab[1], left_vocab[2]],   # touches left
        [],                                               # no-op
        [right_vocab[0], right_vocab[1]],                 # touches right
    ]
    token_id_queries = [np.asarray(r.get_tokens_ids(q), dtype=np.int32) for q in queries]

    mojo_scores, _ = mojo_bm25s.retrieve_batch(r, queries, k=10, num_workers=1)
    ref_scores, _ = _numpy_retrieve_batch(r, token_id_queries, k=10)

    np.testing.assert_array_equal(
        mojo_scores, ref_scores,
        err_msg="empty query did not reset scratch cleanly",
    )


# ---------------------------------------------------------------------------
# Mixed sparsity in one batch — exercises the sparse + dense paths
# together (useful if a heuristic threshold is added later).
# ---------------------------------------------------------------------------


def test_mixed_sparsity_batch_matches_reference():
    """Mixed batch: some queries touch ~1% of rows, some touch ~50%."""
    r, vocab = _build_large_sparse(n_docs=5000, vocab_size=400, seed=2)

    # Pick a rare-ish token (touches few rows) and a token that's
    # likely to be very common — but bm25s drops the doc on the rare
    # side, so we just use len-1 vs len-many queries.
    sparse_query = [vocab[3]]
    medium_query = [vocab[10], vocab[20], vocab[30]]
    long_query = list(vocab[:50])  # touches many rows

    queries = [sparse_query, medium_query, long_query, sparse_query, long_query]
    token_id_queries = [np.asarray(r.get_tokens_ids(q), dtype=np.int32) for q in queries]

    mojo_scores, _ = mojo_bm25s.retrieve_batch(r, queries, k=20, num_workers=1)
    ref_scores, _ = _numpy_retrieve_batch(r, token_id_queries, k=20)
    np.testing.assert_array_equal(
        mojo_scores, ref_scores,
        err_msg="mixed-sparsity batch diverged from numpy reference",
    )


# ---------------------------------------------------------------------------
# Parallel path: per-worker touched/was_touched arrays must each be
# self-contained. Each worker handles a contiguous chunk of queries,
# so each worker's scratch can see leak between *its* queries — same
# adversarial pattern applies per worker.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_workers", [2, 3, 4])
def test_parallel_no_leak_between_disjoint_queries(num_workers):
    """Each worker owns its own scratch + touched/was_touched arrays.
    Within a single worker's chunk, the same adversarial leak pattern
    must not happen. Use enough queries that every worker gets at
    least 2 disjoint ones in its chunk."""
    r, left_vocab, right_vocab = _build_disjoint_corpus()

    # Build alternating left/right queries so every worker chunk
    # straddles the disjoint vocabulary boundary.
    queries = []
    for i in range(num_workers * 4):
        if i % 2 == 0:
            queries.append([left_vocab[i % len(left_vocab)]])
        else:
            queries.append([right_vocab[i % len(right_vocab)]])
    token_id_queries = [np.asarray(r.get_tokens_ids(q), dtype=np.int32) for q in queries]

    mojo_scores, _ = mojo_bm25s.retrieve_batch(
        r, queries, k=10, num_workers=num_workers,
    )
    ref_scores, _ = _numpy_retrieve_batch(r, token_id_queries, k=10)

    np.testing.assert_array_equal(
        mojo_scores, ref_scores,
        err_msg=f"parallel num_workers={num_workers}: score leak between queries",
    )


def test_parallel_matches_serial_bitwise_under_touched_rows():
    """Belt-and-braces: the serial vs parallel bitwise identity that
    PR #31 established must keep holding under the new reset logic.

    Picks a workload where both paths' reset logic (sparse or full-
    zero) gets exercised: medium n_docs, queries of mixed lengths.
    """
    r, vocab = _build_large_sparse(n_docs=1000, vocab_size=200, seed=4)
    rng = np.random.default_rng(11)
    queries = []
    for _ in range(30):
        size = int(rng.integers(1, 6))
        q = [vocab[int(j)] for j in rng.integers(0, len(vocab), size=size)]
        queries.append(q)

    for k in [1, 5, 10]:
        scores_s, ids_s = mojo_bm25s.retrieve_batch(r, queries, k=k, num_workers=1)
        scores_p, ids_p = mojo_bm25s.retrieve_batch(r, queries, k=k, num_workers=4)
        np.testing.assert_array_equal(
            scores_p, scores_s,
            err_msg=f"parallel != serial under touched-rows reset (k={k})",
        )
        np.testing.assert_array_equal(
            ids_p, ids_s,
            err_msg=f"parallel != serial IDs under touched-rows reset (k={k})",
        )


# ---------------------------------------------------------------------------
# Single-query test — pin the degenerate batch_size=1 path which
# bypasses the parallel split. Same adversarial setup, but only one
# query, so the reset that matters is the *cross-batch* one above.
# ---------------------------------------------------------------------------


def test_single_query_batch_matches_reference():
    """Trivial sanity test — batch_size=1 just to pin the serial
    fallback path under the new reset logic."""
    r, left_vocab, _ = _build_disjoint_corpus()
    queries = [[left_vocab[0], left_vocab[1]]]
    token_id_queries = [np.asarray(r.get_tokens_ids(q), dtype=np.int32) for q in queries]

    mojo_scores, _ = mojo_bm25s.retrieve_batch(r, queries, k=10, num_workers=1)
    ref_scores, _ = _numpy_retrieve_batch(r, token_id_queries, k=10)

    np.testing.assert_array_equal(mojo_scores, ref_scores)
