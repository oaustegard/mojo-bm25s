"""Interop-shim tests for the Python facade.

Covers what issue #5 calls out explicitly:
- shape/dtype validation (and coercion where defined)
- zero-copy roundtrip on the one API that contracts a caller-supplied
  output buffer (`csc_score_into`)
- the one-liner smoke example from the issue acceptance
- the vectorized `score_idf_array` for index-time IDF over a vocab
"""

from __future__ import annotations

import math
import numpy as np
import pytest

import mojo_bm25s

# Re-use a couple of the bm25s reference functions as the parity oracle
# for `score_idf_array`. Already an installed test-extra.
from bm25s.scoring import (
    _score_idf_robertson, _score_idf_lucene, _score_idf_atire,
    _score_idf_bm25l, _score_idf_bm25plus,
)


# ---------------------------------------------------------------------------
# One-liner smoke from issue acceptance.
# ---------------------------------------------------------------------------

def test_topk_oneliner_smoke():
    """The exact example in the issue must run end-to-end."""
    rng = np.random.default_rng(0)
    arr = rng.random(1000).astype(np.float32)
    scores, idx = mojo_bm25s.topk(arr, k=10)
    assert scores.shape == (10,)
    assert idx.shape == (10,)
    assert scores.dtype == np.float32
    assert idx.dtype == np.int32
    # Sorted descending.
    assert np.all(np.diff(scores) <= 0)
    # Returned indices actually point at the returned scores.
    np.testing.assert_array_equal(arr[idx], scores)


# ---------------------------------------------------------------------------
# Zero-copy roundtrip — caller-supplied output buffer mutates in place.
# This is the meaningful "no allocations in hot paths" test: the kernel
# writes into the buffer it was handed, not a fresh one.
# ---------------------------------------------------------------------------

def _tiny_csc():
    """Two-doc, three-token CSC, hand-computed.

    Vocabulary: tokens 0, 1, 2. Docs: 0, 1.
    Column 0 (token 0): {doc 0: 0.5, doc 1: 0.25}
    Column 1 (token 1): {doc 0: 0.10}
    Column 2 (token 2): {doc 1: 1.00}
    """
    data = np.array([0.5, 0.25, 0.10, 1.00], dtype=np.float32)
    indices = np.array([0, 1, 0, 1], dtype=np.int32)
    indptr = np.array([0, 2, 3, 4], dtype=np.int32)
    return data, indices, indptr


def test_csc_score_into_writes_caller_buffer():
    data, indices, indptr = _tiny_csc()
    query = np.array([0, 2], dtype=np.int32)
    scores_out = np.zeros(2, dtype=np.float32)

    # Returns None — output is the caller's buffer, mutated in place.
    rv = mojo_bm25s.csc_score_into(data, indptr, indices, query, scores_out)
    assert rv is None
    np.testing.assert_allclose(scores_out, [0.5, 1.25], atol=1e-7)


def test_csc_score_into_accumulates_into_nonzero_buffer():
    """The kernel adds; it does not zero `scores_out` first.

    Lets the caller preload doc-side priors / boosts before retrieve.
    """
    data, indices, indptr = _tiny_csc()
    query = np.array([0], dtype=np.int32)
    scores_out = np.array([10.0, 100.0], dtype=np.float32)

    mojo_bm25s.csc_score_into(data, indptr, indices, query, scores_out)
    np.testing.assert_allclose(scores_out, [10.5, 100.25], atol=1e-7)


def test_csc_score_into_buffer_identity_preserved():
    """Same numpy array object before and after — we did not realloc."""
    data, indices, indptr = _tiny_csc()
    query = np.array([1], dtype=np.int32)
    scores_out = np.zeros(2, dtype=np.float32)
    buffer_id = scores_out.__array_interface__["data"][0]

    mojo_bm25s.csc_score_into(data, indptr, indices, query, scores_out)
    assert scores_out.__array_interface__["data"][0] == buffer_id


def test_csc_score_into_dtype_strictness():
    """Caller-provided `scores_out` must already be float32 contiguous;
    we won't silently coerce because that would defeat zero-copy."""
    data, indices, indptr = _tiny_csc()
    query = np.array([0], dtype=np.int32)
    bad = np.zeros(2, dtype=np.float64)
    with pytest.raises((TypeError, ValueError)):
        mojo_bm25s.csc_score_into(data, indptr, indices, query, bad)


# ---------------------------------------------------------------------------
# Vectorized score_idf_array — vocab-wide IDF lookup table.
# ---------------------------------------------------------------------------

IDF_METHODS = ["robertson", "lucene", "atire", "bm25l", "bm25+"]
REFERENCE_IDF = {
    "robertson": _score_idf_robertson,
    "lucene": _score_idf_lucene,
    "atire": _score_idf_atire,
    "bm25l": _score_idf_bm25l,
    "bm25+": _score_idf_bm25plus,
}


@pytest.mark.parametrize("method", IDF_METHODS)
def test_score_idf_array_parity(method):
    df = np.array([1, 3, 5, 10, 100], dtype=np.int32)
    n_docs = 100
    if method == "robertson":
        expected = np.asarray(
            [REFERENCE_IDF[method](int(d), n_docs, allow_negative=False) for d in df],
            dtype=np.float32,
        )
    else:
        expected = np.asarray(
            [REFERENCE_IDF[method](int(d), n_docs) for d in df],
            dtype=np.float32,
        )
    got = mojo_bm25s.score_idf_array(method, df, n_docs)
    assert got.dtype == np.float32
    assert got.shape == df.shape
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_score_idf_array_matches_scalar():
    """Vectorized form must agree element-wise with the scalar score_idf."""
    df = np.array([2, 7, 42, 99], dtype=np.int32)
    n = 1000
    vec = mojo_bm25s.score_idf_array("lucene", df, n)
    scalar = np.asarray(
        [mojo_bm25s.score_idf("lucene", int(d), n) for d in df],
        dtype=np.float32,
    )
    np.testing.assert_allclose(vec, scalar, atol=1e-7)


def test_score_idf_array_empty():
    out = mojo_bm25s.score_idf_array("lucene", np.array([], dtype=np.int32), 10)
    assert out.shape == (0,)
    assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# Shape / dtype coercion on the input-allocating APIs (`score_tfc`,
# `csc_score`, `topk`). These coerce where it's safe; they refuse where
# it would mask a caller bug.
# ---------------------------------------------------------------------------

def test_score_tfc_coerces_input_dtype():
    """float64 input is silently cast to float32 (the kernel's only dtype)."""
    tf_f64 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    tf_f32 = tf_f64.astype(np.float32)
    out_f64 = mojo_bm25s.score_tfc("robertson", tf_f64, 20.0, 15.0, 1.5, 0.75, 0.0)
    out_f32 = mojo_bm25s.score_tfc("robertson", tf_f32, 20.0, 15.0, 1.5, 0.75, 0.0)
    np.testing.assert_allclose(out_f64, out_f32, atol=1e-7)
    assert out_f64.dtype == np.float32


def test_score_tfc_handles_noncontiguous_input():
    """Strided slices should work — we coerce via ascontiguousarray."""
    full = np.arange(20, dtype=np.float32)
    strided = full[::2]  # length 10, non-contiguous
    assert not strided.flags["C_CONTIGUOUS"]
    out = mojo_bm25s.score_tfc("robertson", strided, 20.0, 15.0, 1.5, 0.75, 0.0)
    assert out.shape == (10,)


def test_topk_coerces_input_dtype():
    arr64 = np.array([3.0, 1.0, 4.0, 1.5, 9.0, 2.6], dtype=np.float64)
    scores, idx = mojo_bm25s.topk(arr64, k=3)
    assert scores.dtype == np.float32
    np.testing.assert_array_equal(idx, np.array([4, 2, 0], dtype=np.int32))


def test_topk_rejects_invalid_k():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    with pytest.raises(ValueError):
        mojo_bm25s.topk(arr, k=0)
    with pytest.raises(ValueError):
        mojo_bm25s.topk(arr, k=-1)
    with pytest.raises(ValueError):
        mojo_bm25s.topk(arr, k=4)


def test_csc_score_coerces_input_dtypes():
    """`csc_score` (the allocating variant) coerces inputs; only its
    `_into` sibling is strict about `scores_out`."""
    data = np.array([0.5, 0.25], dtype=np.float64)        # → coerce to f32
    indices = np.array([0, 1], dtype=np.int64)            # → coerce to i32
    indptr = np.array([0, 2], dtype=np.int64)
    query = np.array([0], dtype=np.int64)
    out = mojo_bm25s.csc_score(data, indptr, indices, query, n_docs=2)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [0.5, 0.25], atol=1e-7)
