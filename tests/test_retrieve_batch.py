"""Tests for `mojo_bm25s.retrieve_batch` — the Path A batched entry point.

The contract: same scores as the per-query patch path within float32
tolerance, IDs in the rank-k tie class.
"""

from __future__ import annotations

import numpy as np
import pytest

import bm25s

import mojo_bm25s


VOCAB = [
    "cat", "dog", "fish", "bird", "horse",
    "fast", "slow", "loud", "quiet", "small",
    "river", "ocean", "mountain", "forest", "city",
    "the", "and", "of", "to", "with",
]


def _corpus(n: int = 100, seed: int = 0) -> list[list[str]]:
    rng = np.random.default_rng(seed)
    return [
        list(rng.choice(VOCAB, size=int(rng.integers(4, 12)), replace=True))
        for _ in range(n)
    ]


def _queries(n: int = 12, seed: int = 1) -> list[list[str]]:
    rng = np.random.default_rng(seed)
    return [
        list(rng.choice(VOCAB, size=int(rng.integers(1, 4)), replace=False))
        for _ in range(n)
    ]


@pytest.fixture(scope="module")
def indexed():
    corpus = _corpus()
    r = bm25s.BM25()
    r.index(corpus, show_progress=False)
    rp = bm25s.BM25()
    rp.index(corpus, show_progress=False)
    mojo_bm25s.patch_bm25s(rp)
    return r, rp, _queries()


def test_returns_well_typed_arrays(indexed):
    r, _, queries = indexed
    scores, ids = mojo_bm25s.retrieve_batch(r, queries, k=10)
    assert scores.shape == (len(queries), 10)
    assert ids.shape == (len(queries), 10)
    assert scores.dtype == np.float32
    assert ids.dtype == np.int32


def test_scores_sorted_descending_per_row(indexed):
    r, _, queries = indexed
    scores, _ = mojo_bm25s.retrieve_batch(r, queries, k=10)
    diffs = np.diff(scores, axis=1)
    assert (diffs <= 1e-7).all(), "per-row scores must be sorted descending"


@pytest.mark.parametrize("k", [1, 3, 10])
def test_parity_with_per_query_patch(indexed, k):
    r, rp, queries = indexed
    batch_scores, batch_ids = mojo_bm25s.retrieve_batch(r, queries, k=k)
    for i, q in enumerate(queries):
        pq_ids, pq_scores = rp.retrieve([q], k=k, show_progress=False)
        # Scores must match within float32 tolerance, independent of
        # how ties at the rank-k boundary are broken.
        np.testing.assert_allclose(
            np.sort(batch_scores[i])[::-1],
            np.sort(pq_scores[0])[::-1],
            atol=1e-5,
            err_msg=f"query[{i}]={q!r} k={k}",
        )


@pytest.mark.parametrize("k", [1, 3, 10])
def test_ids_in_rank_k_tie_class(indexed, k):
    """Every ID returned must have a score at least as high as the
    rank-k boundary from the per-query path. Tolerates tie-swap."""
    r, rp, queries = indexed
    batch_scores, batch_ids = mojo_bm25s.retrieve_batch(r, queries, k=k)
    for i, q in enumerate(queries):
        full_scores = r.get_scores(q)
        _, pq_scores = rp.retrieve([q], k=k, show_progress=False)
        boundary = float(pq_scores[0, -1])
        for picked_id in batch_ids[i].tolist():
            picked_score = float(full_scores[picked_id])
            assert picked_score + 1e-5 >= boundary, (
                f"q={q!r} k={k}: id={picked_id} score={picked_score:.6f} "
                f"below boundary {boundary:.6f}"
            )


def test_token_id_input_works(indexed):
    """Pre-tokenized int input must work the same as string input."""
    r, _, queries = indexed
    id_queries = [
        np.asarray(r.get_tokens_ids(q), dtype=np.int32) for q in queries
    ]
    s_str, _ = mojo_bm25s.retrieve_batch(r, queries, k=5)
    s_int, _ = mojo_bm25s.retrieve_batch(r, id_queries, k=5)
    np.testing.assert_array_equal(s_str, s_int)


def test_empty_query_returns_zeros(indexed):
    r, _, _ = indexed
    scores, ids = mojo_bm25s.retrieve_batch(r, [[]], k=5)
    assert scores.shape == (1, 5)
    np.testing.assert_allclose(scores[0], 0.0)


def test_k_larger_than_corpus_pads_with_zeros():
    """When k exceeds corpus size, the extra row slots stay at the
    zero-init from the Python facade."""
    corpus = [["a", "b"], ["a", "c"]]  # 2 docs
    r = bm25s.BM25()
    r.index(corpus, show_progress=False)
    scores, ids = mojo_bm25s.retrieve_batch(r, [["a"]], k=5)
    assert scores.shape == (1, 5)
    # Position 0 and 1 should have real scores; 2-4 stay at init.
    assert np.all(scores[0, 2:] == 0.0)


def test_rejects_invalid_k(indexed):
    r, _, queries = indexed
    with pytest.raises(ValueError):
        mojo_bm25s.retrieve_batch(r, queries, k=0)
    with pytest.raises(ValueError):
        mojo_bm25s.retrieve_batch(r, queries, k=-1)
