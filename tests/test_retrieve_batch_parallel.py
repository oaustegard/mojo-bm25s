"""Multithreaded `retrieve_batch` — correctness and parity tests.

The parallel path partitions the batch into `num_workers` contiguous
chunks; each worker owns its own scratch and writes a disjoint slice
of the output. Queries are independent — no floating-point reorder
across queries — so parallel output must be bitwise-identical to
serial.

These tests pin that identity across the chunking edge cases
(`batch_size % num_workers != 0`, `num_workers > batch_size`,
`batch_size == 1`, empty queries inside a batch).
"""

from __future__ import annotations

import numpy as np
import pytest

import bm25s

import mojo_bm25s


# ---------------------------------------------------------------------------
# Shared fixture — small synthetic corpus that produces non-trivial
# overlap between queries.
# ---------------------------------------------------------------------------


@pytest.fixture
def indexed_corpus():
    rng = np.random.default_rng(0)
    vocab = ["apple", "berry", "cherry", "date", "elder", "fig", "grape", "hazelnut"]
    corpus = []
    for _ in range(50):
        size = int(rng.integers(2, 6))
        doc = [vocab[int(i)] for i in rng.integers(0, len(vocab), size=size)]
        corpus.append(doc)
    r = bm25s.BM25()
    r.index(corpus, show_progress=False)

    queries = [
        ["apple", "berry"],
        ["cherry"],
        ["date", "elder", "fig"],
        ["grape", "hazelnut"],
        ["apple"],
        ["berry", "cherry", "date"],
        ["elder"],
        ["fig", "grape"],
        ["hazelnut", "apple", "berry"],
        ["cherry", "date"],
        ["elder", "fig"],
        ["grape", "hazelnut", "apple"],
        ["berry"],
    ]
    return r, queries


# ---------------------------------------------------------------------------
# Parallel/serial bitwise identity — the load-bearing property.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_workers", [2, 3, 4, 8])
@pytest.mark.parametrize("k", [1, 5, 10])
def test_parallel_matches_serial_bitwise(indexed_corpus, num_workers, k):
    """For every (num_workers, k) combo, parallel output must be
    byte-identical to the serial baseline. Queries are independent
    so no FP reorder is possible."""
    r, queries = indexed_corpus
    scores_serial, ids_serial = mojo_bm25s.retrieve_batch(
        r, queries, k=k, num_workers=1
    )
    scores_par, ids_par = mojo_bm25s.retrieve_batch(
        r, queries, k=k, num_workers=num_workers
    )
    np.testing.assert_array_equal(scores_par, scores_serial)
    np.testing.assert_array_equal(ids_par, ids_serial)


# ---------------------------------------------------------------------------
# Chunking edge cases.
# ---------------------------------------------------------------------------


def test_num_workers_larger_than_batch(indexed_corpus):
    """Worker count > batch size: idle workers must not write OOB."""
    r, queries = indexed_corpus
    short_batch = queries[:3]
    scores_s, ids_s = mojo_bm25s.retrieve_batch(r, short_batch, k=5, num_workers=1)
    scores_p, ids_p = mojo_bm25s.retrieve_batch(r, short_batch, k=5, num_workers=16)
    np.testing.assert_array_equal(scores_p, scores_s)
    np.testing.assert_array_equal(ids_p, ids_s)


def test_batch_size_one_uses_serial_fallback(indexed_corpus):
    """batch_size=1 must take the serial path even with num_workers>1
    (parallel chunking would degenerate to a single worker anyway)."""
    r, queries = indexed_corpus
    scores_s, ids_s = mojo_bm25s.retrieve_batch(r, [queries[0]], k=3, num_workers=1)
    scores_p, ids_p = mojo_bm25s.retrieve_batch(r, [queries[0]], k=3, num_workers=4)
    np.testing.assert_array_equal(scores_p, scores_s)
    np.testing.assert_array_equal(ids_p, ids_s)


def test_batch_uneven_chunks_handled(indexed_corpus):
    """batch_size not divisible by num_workers — the last chunk shrinks."""
    r, queries = indexed_corpus
    # 13 queries split across 4 workers: chunks of 4, 4, 4, 1.
    scores_s, ids_s = mojo_bm25s.retrieve_batch(r, queries, k=4, num_workers=1)
    scores_p, ids_p = mojo_bm25s.retrieve_batch(r, queries, k=4, num_workers=4)
    np.testing.assert_array_equal(scores_p, scores_s)
    np.testing.assert_array_equal(ids_p, ids_s)


def test_empty_queries_in_batch_parallel(indexed_corpus):
    """Batches with mixed empty + non-empty queries must work."""
    r, queries = indexed_corpus
    mixed = [queries[0], [], queries[1], [], queries[2]]
    scores_s, ids_s = mojo_bm25s.retrieve_batch(r, mixed, k=3, num_workers=1)
    scores_p, ids_p = mojo_bm25s.retrieve_batch(r, mixed, k=3, num_workers=4)
    np.testing.assert_array_equal(scores_p, scores_s)
    np.testing.assert_array_equal(ids_p, ids_s)


def test_default_num_workers_auto_detect(indexed_corpus):
    """num_workers=0 must auto-detect cores and still produce the right
    answer (the count is implementation-defined; the result is not)."""
    r, queries = indexed_corpus
    scores_default, ids_default = mojo_bm25s.retrieve_batch(r, queries, k=5)  # num_workers=0
    scores_s, ids_s = mojo_bm25s.retrieve_batch(r, queries, k=5, num_workers=1)
    np.testing.assert_array_equal(scores_default, scores_s)
    np.testing.assert_array_equal(ids_default, ids_s)


def test_num_workers_negative_raises():
    r = bm25s.BM25()
    r.index([["a"], ["b"]], show_progress=False)
    with pytest.raises(ValueError, match=r"num_workers"):
        mojo_bm25s.retrieve_batch(r, [["a"]], k=1, num_workers=-1)


# ---------------------------------------------------------------------------
# Parity vs bm25s reference — parallel path must match the oracle, not
# just the serial mojo path. (Catches a hypothetical bug where serial &
# parallel both reproduce the same wrong answer.)
# ---------------------------------------------------------------------------


def test_parallel_matches_bm25s_reference(indexed_corpus):
    r, queries = indexed_corpus
    ref_results, ref_scores = r.retrieve(
        [r.get_tokens_ids(q) if q else np.array([], dtype=np.int32) for q in queries],
        k=5, show_progress=False,
    )
    mojo_scores, mojo_ids = mojo_bm25s.retrieve_batch(
        r, queries, k=5, num_workers=4,
    )
    np.testing.assert_allclose(mojo_scores, ref_scores, atol=1e-6)
    # IDs may differ at tie-class boundaries; assert per-rank score
    # match instead.
    for row in range(len(queries)):
        for rank in range(5):
            assert mojo_scores[row, rank] == pytest.approx(
                ref_scores[row, rank], abs=1e-6
            ), f"row={row} rank={rank}"
