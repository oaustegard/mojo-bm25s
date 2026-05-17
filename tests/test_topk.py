"""Parity tests for top-k selection vs bm25s reference.

Mojo top-k must match `bm25s.selection.topk(backend='numpy', sorted=True)`
modulo tie-breaking: scores at every rank must match within atol=1e-6,
and indices must match when the scores at rank k-1 and rank k differ
(no boundary tie). Two algorithms are exposed for benchmarking — heap
(O(N log k)) and quickselect (O(N) average) — both must satisfy
parity independently.
"""

from __future__ import annotations

import numpy as np
import pytest

from bm25s.selection import topk as bm25s_topk

import mojo_bm25s


ATOL = 1e-6
ALGORITHMS = ["heap", "quickselect"]


def _sorted_pairs(scores: np.ndarray, indices: np.ndarray):
    """Return (scores, indices) sorted by descending score, ties broken by
    ascending index. Used to canonicalize outputs from algorithms whose
    tie-breaking differs."""
    order = np.lexsort((indices, -scores))
    return scores[order], indices[order]


# ---------------------------------------------------------------------------
# Parity: distinct scores → both scores and indices match exactly.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("algo", ALGORITHMS)
@pytest.mark.parametrize(
    "n,k",
    [(100, 10), (1000, 50), (10, 1), (10, 10), (257, 16)],
)
def test_topk_parity_random_scores(algo, n, k):
    rng = np.random.default_rng(seed=0xC0FFEE + n * 31 + k)
    scores = rng.random(n, dtype=np.float32)

    ref_scores, ref_indices = bm25s_topk(scores, k, backend="numpy", sorted=True)
    got_scores, got_indices = mojo_bm25s.topk(scores, k, algorithm=algo)

    assert got_scores.dtype == np.float32
    assert got_indices.dtype == np.int32
    assert got_scores.shape == (k,)
    assert got_indices.shape == (k,)
    np.testing.assert_allclose(got_scores, ref_scores, atol=ATOL)
    np.testing.assert_array_equal(got_indices, ref_indices.astype(np.int32))


# ---------------------------------------------------------------------------
# Descending order invariant: scores[i] >= scores[i+1].
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("algo", ALGORITHMS)
def test_topk_output_is_descending(algo):
    rng = np.random.default_rng(seed=1)
    scores = rng.random(500, dtype=np.float32)
    got_scores, _ = mojo_bm25s.topk(scores, 25, algorithm=algo)
    assert np.all(np.diff(got_scores) <= 0), "top-k scores must be non-increasing"


# ---------------------------------------------------------------------------
# Indices must point at valid input positions, no duplicates.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("algo", ALGORITHMS)
def test_topk_indices_unique_and_in_range(algo):
    rng = np.random.default_rng(seed=2)
    n, k = 300, 30
    scores = rng.random(n, dtype=np.float32)
    _, got_indices = mojo_bm25s.topk(scores, k, algorithm=algo)
    assert len(set(int(i) for i in got_indices)) == k
    assert int(got_indices.min()) >= 0
    assert int(got_indices.max()) < n


# ---------------------------------------------------------------------------
# Returned scores must equal the input scores at the returned indices.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("algo", ALGORITHMS)
def test_topk_scores_match_input_at_indices(algo):
    rng = np.random.default_rng(seed=3)
    scores = rng.random(200, dtype=np.float32)
    got_scores, got_indices = mojo_bm25s.topk(scores, 15, algorithm=algo)
    np.testing.assert_array_equal(got_scores, scores[got_indices])


# ---------------------------------------------------------------------------
# Tie-breaking edge case from the issue acceptance criteria:
# when scores at rank k-1 and rank k are equal, the implementation may
# return either index. We assert *score* equality across rank k-1.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("algo", ALGORITHMS)
def test_topk_tie_at_boundary(algo):
    # 10 scores, indices 4 and 5 both have score 0.5 — the (k=5) boundary.
    scores = np.array(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9],
        dtype=np.float32,
    )
    got_scores, _ = mojo_bm25s.topk(scores, 5, algorithm=algo)
    # The set of top-5 scores must be {0.9, 0.8, 0.7, 0.6, 0.5}.
    expected = np.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)
    np.testing.assert_allclose(np.sort(got_scores)[::-1], expected, atol=ATOL)


# ---------------------------------------------------------------------------
# All-equal scores: every output score must equal the input value; indices
# can be any k-subset of [0, n).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("algo", ALGORITHMS)
def test_topk_all_equal(algo):
    scores = np.full(50, 0.3, dtype=np.float32)
    got_scores, got_indices = mojo_bm25s.topk(scores, 7, algorithm=algo)
    np.testing.assert_allclose(got_scores, np.full(7, 0.3), atol=ATOL)
    assert len(set(int(i) for i in got_indices)) == 7


# ---------------------------------------------------------------------------
# k == 1 and k == n boundary sizes.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("algo", ALGORITHMS)
def test_topk_k_equals_one(algo):
    rng = np.random.default_rng(seed=4)
    scores = rng.random(100, dtype=np.float32)
    got_scores, got_indices = mojo_bm25s.topk(scores, 1, algorithm=algo)
    assert got_scores.shape == (1,)
    assert got_indices.shape == (1,)
    assert int(got_indices[0]) == int(scores.argmax())
    assert float(got_scores[0]) == pytest.approx(float(scores.max()), abs=ATOL)


@pytest.mark.parametrize("algo", ALGORITHMS)
def test_topk_k_equals_n(algo):
    rng = np.random.default_rng(seed=5)
    n = 40
    scores = rng.random(n, dtype=np.float32)
    got_scores, got_indices = mojo_bm25s.topk(scores, n, algorithm=algo)
    # Should be a complete sorted permutation.
    expected_order = np.argsort(scores)[::-1].astype(np.int32)
    np.testing.assert_array_equal(got_indices, expected_order)
    np.testing.assert_allclose(got_scores, scores[expected_order], atol=ATOL)


# ---------------------------------------------------------------------------
# Unknown algorithm should fail loud.
# ---------------------------------------------------------------------------

def test_topk_unknown_algorithm_raises():
    scores = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    with pytest.raises((ValueError, RuntimeError, Exception)):
        mojo_bm25s.topk(scores, 2, algorithm="bogus")


# ---------------------------------------------------------------------------
# Heap and quickselect must agree with each other on every score.
# ---------------------------------------------------------------------------

def test_heap_and_quickselect_agree():
    rng = np.random.default_rng(seed=0xABCD)
    scores = rng.random(800, dtype=np.float32)
    s_h, i_h = mojo_bm25s.topk(scores, 40, algorithm="heap")
    s_q, i_q = mojo_bm25s.topk(scores, 40, algorithm="quickselect")
    np.testing.assert_allclose(s_h, s_q, atol=ATOL)
    # With distinct random scores both should also pick the same indices.
    np.testing.assert_array_equal(i_h, i_q)
