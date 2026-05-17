"""Input-validation tests for the Python facade.

The Mojo kernels operate on raw int32/float32 pointers with no in-kernel
bounds or overflow checks. The Python shim is the only place where
adversarial / careless callers can be stopped before the kernel walks
arbitrary memory or silently truncates 64-bit indices.

Findings covered (from the 2026-05-17 hardening review):

- #2 int32 silent truncation on indptr / indices
- #7 int32 wrap on `lengths.cumsum()` in `retrieve_batch`
- #11 OOB query token IDs walking past `indptr.shape[0]-1`
"""

from __future__ import annotations

import numpy as np
import pytest

import mojo_bm25s


INT32_MAX = np.iinfo(np.int32).max


# ---------------------------------------------------------------------------
# Finding #2 — int32 silent truncation on indptr / indices.
# ---------------------------------------------------------------------------


def test_csc_score_rejects_indptr_value_overflowing_int32():
    """int64 indptr whose largest entry exceeds INT32_MAX must raise.

    Today's behavior: np.ascontiguousarray(indptr, dtype=np.int32)
    silently wraps to negative — the kernel then walks arbitrary memory.
    """
    n_vocab = 3
    data = np.array([0.5], dtype=np.float32)
    indices = np.array([0], dtype=np.int32)
    # indptr final-entry > INT32_MAX. Use int64 so the value exists.
    indptr = np.array([0, 1, 1, INT32_MAX + 1], dtype=np.int64)
    query = np.array([0], dtype=np.int32)

    with pytest.raises(OverflowError, match=r"INT32_MAX|overflow|int32"):
        mojo_bm25s.csc_score(data, indptr, indices, query, n_docs=1)


def test_csc_score_rejects_indices_value_overflowing_int32():
    n_vocab = 1
    data = np.array([0.5], dtype=np.float32)
    indices = np.array([INT32_MAX + 1], dtype=np.int64)
    indptr = np.array([0, 1], dtype=np.int32)
    query = np.array([0], dtype=np.int32)

    with pytest.raises(OverflowError, match=r"INT32_MAX|overflow|int32"):
        mojo_bm25s.csc_score(data, indptr, indices, query, n_docs=2)


def test_csc_score_into_rejects_indptr_value_overflowing_int32():
    data = np.array([0.5], dtype=np.float32)
    indices = np.array([0], dtype=np.int32)
    indptr = np.array([0, 1, 1, INT32_MAX + 1], dtype=np.int64)
    query = np.array([0], dtype=np.int32)
    scores_out = np.zeros(2, dtype=np.float32)

    with pytest.raises(OverflowError, match=r"INT32_MAX|overflow|int32"):
        mojo_bm25s.csc_score_into(data, indptr, indices, query, scores_out)


def test_csc_score_accepts_native_int32_with_full_range_values():
    """In-range int32 indptr / indices must continue to work — we only
    reject values that would *silently truncate*."""
    data = np.array([0.5, 0.25], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    indptr = np.array([0, 1, 2], dtype=np.int32)
    query = np.array([0, 1], dtype=np.int32)
    out = mojo_bm25s.csc_score(data, indptr, indices, query, n_docs=2)
    np.testing.assert_allclose(out, [0.5, 0.25], atol=1e-7)


def test_csc_score_accepts_int64_inputs_within_int32_range():
    """int64 inputs whose values all fit in int32 should coerce cleanly,
    matching the existing test_interop.test_csc_score_coerces_input_dtypes."""
    data = np.array([0.5, 0.25], dtype=np.float64)
    indices = np.array([0, 1], dtype=np.int64)
    indptr = np.array([0, 2], dtype=np.int64)
    query = np.array([0], dtype=np.int64)
    out = mojo_bm25s.csc_score(data, indptr, indices, query, n_docs=2)
    np.testing.assert_allclose(out, [0.5, 0.25], atol=1e-7)


# ---------------------------------------------------------------------------
# Finding #7 — int32 wrap on `lengths.cumsum()` in `retrieve_batch`.
#
# Rather than allocate a real 2-billion-token batch, we directly construct
# a batch whose total token count claim exceeds INT32_MAX by stubbing
# `len()` on a sequence. The cumsum-overflow check has to fire before any
# allocation happens.
# ---------------------------------------------------------------------------


class _FakeIntTokenList:
    """Mimics a long list of ints without actually allocating one."""

    def __init__(self, length: int, value: int = 1) -> None:
        self._length = length
        self._value = value

    def __len__(self) -> int:
        return self._length

    def __iter__(self):
        return iter([self._value])

    def __getitem__(self, idx):
        return self._value


class _StubRetriever:
    """Minimal stand-in for a bm25s.BM25 retriever — only the attributes
    `retrieve_batch` touches."""

    def __init__(self, n_docs: int = 4) -> None:
        # 1 vocab term, 1 nnz; trivial valid CSC.
        self.scores = {
            "data": np.array([1.0], dtype=np.float32),
            "indptr": np.array([0, 1], dtype=np.int32),
            "indices": np.array([0], dtype=np.int32),
            "num_docs": n_docs,
        }

    def get_tokens_ids(self, q):  # not used in this test
        return [0 for _ in q]


def test_retrieve_batch_rejects_cumulative_token_count_overflowing_int32(
    monkeypatch,
):
    """A batch whose total query tokens claim exceeds INT32_MAX must
    raise OverflowError — silent int32 wrap would let the kernel read
    past the queries_concat buffer."""
    # Build a fake batch of two queries whose declared lengths sum to
    # INT32_MAX + 2 but each query carries one real token (the per-element
    # iteration in retrieve_batch should never touch the bogus length:
    # the overflow guard must fire first).
    retriever = _StubRetriever()
    half = INT32_MAX // 2 + 1
    big_q_a = _FakeIntTokenList(half, value=0)
    big_q_b = _FakeIntTokenList(half, value=0)

    with pytest.raises(OverflowError, match=r"INT32_MAX|overflow|int32"):
        mojo_bm25s.retrieve_batch(retriever, [big_q_a, big_q_b], k=1)


def test_retrieve_batch_accepts_small_batch_within_int32():
    """A normal small batch is unaffected by the new guard."""
    retriever = _StubRetriever(n_docs=2)
    queries = [[0], [0], []]  # mix of populated and empty queries
    scores, ids = mojo_bm25s.retrieve_batch(retriever, queries, k=1)
    assert scores.shape == (3, 1)
    assert ids.shape == (3, 1)


# ---------------------------------------------------------------------------
# Finding #11 — OOB query token IDs walking past indptr.
# ---------------------------------------------------------------------------


def test_csc_score_rejects_query_token_id_at_or_past_vocab_size():
    """A query token id >= n_vocab indexes past indptr; current behavior
    is an arbitrary-memory read. The facade must raise IndexError before
    invoking the kernel."""
    data = np.array([0.5, 0.25], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    indptr = np.array([0, 1, 2], dtype=np.int32)  # n_vocab = 2
    bad_query = np.array([5], dtype=np.int32)  # token 5 doesn't exist

    with pytest.raises(IndexError, match=r"token|vocabulary|indptr"):
        mojo_bm25s.csc_score(data, indptr, indices, bad_query, n_docs=2)


def test_csc_score_rejects_negative_query_token_id():
    data = np.array([0.5], dtype=np.float32)
    indices = np.array([0], dtype=np.int32)
    indptr = np.array([0, 1], dtype=np.int32)
    bad_query = np.array([-1], dtype=np.int32)

    with pytest.raises(IndexError, match=r"token|negative"):
        mojo_bm25s.csc_score(data, indptr, indices, bad_query, n_docs=1)


def test_csc_score_into_rejects_oob_query_token_id():
    data = np.array([0.5], dtype=np.float32)
    indices = np.array([0], dtype=np.int32)
    indptr = np.array([0, 1], dtype=np.int32)
    bad_query = np.array([99], dtype=np.int32)
    scores_out = np.zeros(1, dtype=np.float32)

    with pytest.raises(IndexError):
        mojo_bm25s.csc_score_into(
            data, indptr, indices, bad_query, scores_out
        )


def test_retrieve_batch_rejects_oob_query_token_id():
    """Same guard for the batched path: every query token id must be
    a valid index into the retriever's indptr."""
    retriever = _StubRetriever(n_docs=2)
    # retriever vocab is 1 term (indptr.shape[0] - 1 == 1), so token 7
    # is OOB.
    with pytest.raises(IndexError):
        mojo_bm25s.retrieve_batch(retriever, [[7]], k=1)


def test_csc_score_accepts_query_token_id_at_last_valid_position():
    """Boundary: the last valid token id is n_vocab - 1.

    For indptr of shape (n_vocab + 1,), the largest valid token id is
    indptr.shape[0] - 2. This must NOT raise.
    """
    n_vocab = 3
    data = np.array([0.5, 0.25, 0.1], dtype=np.float32)
    indices = np.array([0, 1, 0], dtype=np.int32)
    indptr = np.array([0, 1, 2, 3], dtype=np.int32)  # n_vocab = 3
    last_valid_query = np.array([n_vocab - 1], dtype=np.int32)
    out = mojo_bm25s.csc_score(data, indptr, indices, last_valid_query, n_docs=2)
    np.testing.assert_allclose(out, [0.1, 0.0], atol=1e-7)


def test_csc_score_accepts_empty_query():
    """Empty query is well-defined: every doc scores 0. The OOB guard
    must short-circuit (max/min of an empty array would raise)."""
    data = np.array([0.5], dtype=np.float32)
    indices = np.array([0], dtype=np.int32)
    indptr = np.array([0, 1], dtype=np.int32)
    empty_query = np.array([], dtype=np.int32)
    out = mojo_bm25s.csc_score(data, indptr, indices, empty_query, n_docs=2)
    np.testing.assert_allclose(out, [0.0, 0.0], atol=1e-7)
