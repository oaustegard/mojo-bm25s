"""Parity tests for BM25 scoring kernels against bm25s reference.

Every (tfc_method, idf_method) combination is asserted bit-for-bit
within atol=1e-6 of the bm25s reference. Edge cases (df=0, N=1,
tf_array length > simd_width) come from the issue acceptance criteria.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from bm25s.scoring import (
    _score_tfc_robertson, _score_tfc_lucene, _score_tfc_atire,
    _score_tfc_bm25l, _score_tfc_bm25plus,
    _score_idf_robertson, _score_idf_lucene, _score_idf_atire,
    _score_idf_bm25l, _score_idf_bm25plus,
)

import mojo_bm25s

ATOL = 1e-6

TFC_METHODS = ["robertson", "lucene", "atire", "bm25l", "bm25+"]
IDF_METHODS = ["robertson", "lucene", "atire", "bm25l", "bm25+"]

REFERENCE_TFC = {
    "robertson": _score_tfc_robertson,
    "lucene": _score_tfc_lucene,
    "atire": _score_tfc_atire,
    "bm25l": _score_tfc_bm25l,
    "bm25+": _score_tfc_bm25plus,
}
REFERENCE_IDF = {
    "robertson": _score_idf_robertson,
    "lucene": _score_idf_lucene,
    "atire": _score_idf_atire,
    "bm25l": _score_idf_bm25l,
    "bm25+": _score_idf_bm25plus,
}

# Fixed synthetic corpus parameters.
K1 = 1.5
B = 0.75
DELTA = 1.0
L_AVG = 15.0
L_D = 20.0
N_DOCS = 100

# tf_array length 9 — not divisible by simd_width 4 or 8.
TF_ARRAY = np.array([0, 1, 2, 3, 5, 7, 10, 15, 20], dtype=np.float32)
# tf_array length 17 — explicitly tests tail handling past one full simd block.
TF_ARRAY_LONG = np.arange(17, dtype=np.float32)

# df values bm25s computes IDF on; df=0 is filtered out here because
# atire/bm25+ would log(0). df=0 handled in its own test below.
DF_VALUES = [1, 3, 5, 10, 100]


# ---------------------------------------------------------------------------
# TFC parity — per-method, vectorized over tf_array.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", TFC_METHODS)
def test_tfc_parity(method):
    ref_fn = REFERENCE_TFC[method]
    expected = np.asarray(
        ref_fn(TF_ARRAY, l_d=L_D, l_avg=L_AVG, k1=K1, b=B, delta=DELTA),
        dtype=np.float32,
    )
    got = mojo_bm25s.score_tfc(method, TF_ARRAY, L_D, L_AVG, K1, B, DELTA)
    assert got.dtype == np.float32
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, atol=ATOL)


@pytest.mark.parametrize("method", TFC_METHODS)
def test_tfc_length_exceeds_simd_width(method):
    """tf_array of length 17 exercises full SIMD blocks + scalar tail."""
    ref_fn = REFERENCE_TFC[method]
    expected = np.asarray(
        ref_fn(TF_ARRAY_LONG, l_d=L_D, l_avg=L_AVG, k1=K1, b=B, delta=DELTA),
        dtype=np.float32,
    )
    got = mojo_bm25s.score_tfc(method, TF_ARRAY_LONG, L_D, L_AVG, K1, B, DELTA)
    np.testing.assert_allclose(got, expected, atol=ATOL)


@pytest.mark.parametrize("method", TFC_METHODS)
def test_tfc_empty_array(method):
    empty = np.array([], dtype=np.float32)
    got = mojo_bm25s.score_tfc(method, empty, L_D, L_AVG, K1, B, DELTA)
    assert got.shape == (0,)


# ---------------------------------------------------------------------------
# IDF parity — per-method × per-df.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", IDF_METHODS)
@pytest.mark.parametrize("df", DF_VALUES)
def test_idf_parity(method, df):
    ref_fn = REFERENCE_IDF[method]
    if method == "robertson":
        expected = float(ref_fn(df, N_DOCS, allow_negative=False))
        got = mojo_bm25s.score_idf(method, df, N_DOCS, allow_negative=False)
    else:
        expected = float(ref_fn(df, N_DOCS))
        got = mojo_bm25s.score_idf(method, df, N_DOCS)
    assert math.isclose(got, expected, abs_tol=ATOL)


def test_idf_robertson_allow_negative_branch():
    """Robertson with allow_negative=True; high df where inner < 1."""
    df, n = 80, 100
    expected = float(_score_idf_robertson(df, n, allow_negative=True))
    got = mojo_bm25s.score_idf("robertson", df, n, allow_negative=True)
    assert math.isclose(got, expected, abs_tol=ATOL)
    # And the clamped version should be 0 (log(1)).
    clamped = mojo_bm25s.score_idf("robertson", df, n, allow_negative=False)
    assert math.isclose(clamped, 0.0, abs_tol=ATOL)


# ---------------------------------------------------------------------------
# Combined (tfc, idf) parity — the full 25-combo matrix.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tfc_method", TFC_METHODS)
@pytest.mark.parametrize("idf_method", IDF_METHODS)
def test_combined_score_parity(tfc_method, idf_method):
    """For each of the 25 (tfc, idf) combos, tfc * idf must match bm25s."""
    df = 5
    ref_tfc = np.asarray(
        REFERENCE_TFC[tfc_method](
            TF_ARRAY, l_d=L_D, l_avg=L_AVG, k1=K1, b=B, delta=DELTA
        ),
        dtype=np.float32,
    )
    if idf_method == "robertson":
        ref_idf = float(REFERENCE_IDF[idf_method](df, N_DOCS, allow_negative=False))
    else:
        ref_idf = float(REFERENCE_IDF[idf_method](df, N_DOCS))
    expected = ref_tfc * np.float32(ref_idf)

    got_tfc = mojo_bm25s.score_tfc(tfc_method, TF_ARRAY, L_D, L_AVG, K1, B, DELTA)
    got_idf = mojo_bm25s.score_idf(idf_method, df, N_DOCS)
    got = got_tfc * np.float32(got_idf)

    np.testing.assert_allclose(got, expected, atol=ATOL)


# ---------------------------------------------------------------------------
# Edge cases from the issue.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["robertson", "lucene", "bm25l"])
def test_idf_df_zero(method):
    """df=0 is well-defined for variants that don't divide by df."""
    expected = float(REFERENCE_IDF[method](0, N_DOCS))
    got = mojo_bm25s.score_idf(method, 0, N_DOCS)
    assert math.isclose(got, expected, abs_tol=ATOL)


@pytest.mark.parametrize("idf_method", IDF_METHODS)
def test_single_doc_corpus(idf_method):
    """N=1, df=1: the smallest non-degenerate corpus."""
    if idf_method == "robertson":
        expected = float(REFERENCE_IDF[idf_method](1, 1, allow_negative=False))
        got = mojo_bm25s.score_idf(idf_method, 1, 1, allow_negative=False)
    else:
        expected = float(REFERENCE_IDF[idf_method](1, 1))
        got = mojo_bm25s.score_idf(idf_method, 1, 1)
    assert math.isclose(got, expected, abs_tol=ATOL)


def test_unknown_method_raises():
    with pytest.raises((ValueError, RuntimeError, Exception)):
        mojo_bm25s.score_tfc("nonexistent", TF_ARRAY, L_D, L_AVG, K1, B, DELTA)
    with pytest.raises((ValueError, RuntimeError, Exception)):
        mojo_bm25s.score_idf("nonexistent", 1, N_DOCS)
