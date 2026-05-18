"""Anytime retrieval — impact-ordered postings with per-term early-exit.

Issue #35. Pair to ``retrieve_batch``: same per-query top-k contract,
different walk order. Where ``retrieve_batch`` scans every
``(data[j], indices[j])`` entry of every column referenced by the
query, this kernel walks each column in **descending-impact** order
and stops walking that column as soon as its remaining entries can no
longer move the top-k boundary.

Index requirement
-----------------

The CSC must already be impact-ordered — each column's ``data[j]`` is
descending. ``build_impact_ordered_index`` produces this layout. Feeding
a doc-id-ordered index to this kernel would still produce correct
results (the early-exit gate never trips) but the speedup disappears.

Threshold sketch (per term ``t``, after each scatter step):

    remaining_max[t] = data[j_next]               # next entry's impact
    upper_bound[d]    ≤ scratch[d] + remaining_max[t]
                                    + Σ remaining_max[u] for u ≠ t

If ``heap.min()`` is set (heap has ``k`` entries) and
``Σ_t remaining_max[t] < heap.min()``, no doc can enter top-k.
Terminate the term loop early.

This file is the **Python facade** — it allocates the output buffers,
canonicalizes the input (accept a ``bm25s.BM25`` retriever OR a CSC dict
OR a ``LoadedIndex``-shaped object), and calls the Mojo kernel
``retrieve_batch_anytime`` via the loaded ``_kernel`` module.

Debug counter
-------------

For test-driven verification of the early-exit branch, pass
``_debug_iteration_counters={}``; on return, the dict will have
``"entries_visited"`` (sum across all queries / all terms of the
``(data, indices)`` entries actually consumed before early-exit).
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np


# Local re-imports keep the cycle clean: `__init__` imports `anytime`,
# `anytime` reaches back to `__init__` for kernel + helpers via late binding.
def _get_kernel():
    from . import _kernel
    return _kernel


def _to_int32_checked(arr: np.ndarray, name: str) -> np.ndarray:
    from . import _to_int32_checked as f
    return f(arr, name)


def _validate_query_token_ids(query: np.ndarray, n_vocab: int, name: str = "query_token_ids") -> None:
    from . import _validate_query_token_ids as f
    return f(query, n_vocab, name)


_INT32_MAX = int(np.iinfo(np.int32).max)


def _extract_csc_from_input(index_or_retriever) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, int, bool,
]:
    """Pull CSC arrays out of the various accepted input shapes.

    Returns ``(data, indices, indptr, n_docs, impact_ordered)``. If the
    input is a doc-id-ordered CSC (``impact_ordered=False``) we
    transparently build the impact-ordered version on the fly so the
    facade is still correct — at the cost of an extra build pass. The
    caller is expected to either (a) hand us an already-impact-ordered
    dict (set ``impact_ordered=True``), or (b) hand us a ``bm25s.BM25``
    and accept the on-the-fly re-permutation overhead.
    """
    # Dict-like (raw CSC). Accept "num_docs" (bm25s scores dict) or
    # "n_docs" (LoadedIndex-shaped) interchangeably.
    if isinstance(index_or_retriever, Mapping):
        data = np.ascontiguousarray(index_or_retriever["data"], dtype=np.float32)
        indices = _to_int32_checked(np.asarray(index_or_retriever["indices"]), "indices")
        indptr = _to_int32_checked(np.asarray(index_or_retriever["indptr"]), "indptr")
        n_docs = int(
            index_or_retriever.get("num_docs", index_or_retriever.get("n_docs", 0))
        )
        impact_ordered = bool(index_or_retriever.get("impact_ordered", False))
        if not impact_ordered:
            data, indices = _impact_permute_columns(data, indices, indptr)
        return data, indices, indptr, n_docs, True

    # LoadedIndex-shaped dataclass (issue #26).
    if hasattr(index_or_retriever, "data") and hasattr(index_or_retriever, "indptr"):
        data = np.ascontiguousarray(index_or_retriever.data, dtype=np.float32)
        indices = _to_int32_checked(np.asarray(index_or_retriever.indices), "indices")
        indptr = _to_int32_checked(np.asarray(index_or_retriever.indptr), "indptr")
        n_docs = int(index_or_retriever.n_docs)
        impact_ordered = bool(getattr(index_or_retriever, "impact_ordered", False))
        if not impact_ordered:
            data, indices = _impact_permute_columns(data, indices, indptr)
        return data, indices, indptr, n_docs, True

    # bm25s.BM25 retriever shape: .scores dict + .get_tokens_ids.
    if hasattr(index_or_retriever, "scores"):
        scores = index_or_retriever.scores
        data = np.ascontiguousarray(scores["data"], dtype=np.float32)
        indices = _to_int32_checked(np.asarray(scores["indices"]), "indices")
        indptr = _to_int32_checked(np.asarray(scores["indptr"]), "indptr")
        n_docs = int(scores["num_docs"])
        # The dict may carry an "impact_ordered" hint (tests / our own
        # build path) — honor it. bm25s indexes lack the key and are
        # always doc-id-ordered, so we permute on the fly.
        impact_ordered = bool(scores.get("impact_ordered", False)) \
            if isinstance(scores, Mapping) else False
        if not impact_ordered:
            data, indices = _impact_permute_columns(data, indices, indptr)
        return data, indices, indptr, n_docs, True

    raise TypeError(
        f"retrieve_batch_anytime: unsupported input type "
        f"{type(index_or_retriever).__name__}"
    )


def _impact_permute_columns(
    data: np.ndarray, indices: np.ndarray, indptr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Permute each column so data[j] is descending. Stable on ties."""
    if data.size == 0:
        return data, indices
    data_out = np.empty_like(data)
    indices_out = np.empty_like(indices)
    n_vocab = indptr.shape[0] - 1
    for t in range(n_vocab):
        lo, hi = int(indptr[t]), int(indptr[t + 1])
        if hi - lo < 2:
            data_out[lo:hi] = data[lo:hi]
            indices_out[lo:hi] = indices[lo:hi]
            continue
        order = np.argsort(-data[lo:hi], kind="stable")
        data_out[lo:hi] = data[lo:hi][order]
        indices_out[lo:hi] = indices[lo:hi][order]
    return data_out, indices_out


def _convert_queries(query_tokens_batch, retriever) -> tuple[
    list[np.ndarray], np.ndarray, np.ndarray, int,
]:
    """Convert the heterogeneous batch into concatenated int32 + offsets."""
    batch_size = len(query_tokens_batch)
    lengths64 = np.fromiter(
        (len(q) for q in query_tokens_batch), dtype=np.int64, count=batch_size,
    )
    total_tokens = int(lengths64.sum())
    if total_tokens > _INT32_MAX:
        raise OverflowError(
            f"total query tokens {total_tokens} exceeds int32 max"
        )

    token_id_batch: list[np.ndarray] = []
    for q in query_tokens_batch:
        if len(q) == 0:
            token_id_batch.append(np.zeros(0, dtype=np.int32))
        elif isinstance(q[0], str):
            if retriever is None or not hasattr(retriever, "get_tokens_ids"):
                raise TypeError(
                    "string-token queries require a retriever with "
                    "get_tokens_ids(); pass int32 arrays instead, or pass "
                    "a bm25s.BM25 instance as the first argument."
                )
            ids = retriever.get_tokens_ids(q)
            token_id_batch.append(np.asarray(ids, dtype=np.int32))
        else:
            token_id_batch.append(np.asarray(q, dtype=np.int32))

    offsets = np.zeros(batch_size + 1, dtype=np.int32)
    np.cumsum(lengths64.astype(np.int32), out=offsets[1:])

    if batch_size > 0:
        queries_concat = np.ascontiguousarray(
            np.concatenate(token_id_batch), dtype=np.int32,
        )
    else:
        queries_concat = np.zeros(0, dtype=np.int32)

    return token_id_batch, queries_concat, offsets, batch_size


def retrieve_batch_anytime(
    index_or_retriever,
    query_tokens_batch,
    k: int = 10,
    num_workers: int = 0,
    *,
    _debug_iteration_counters: Optional[dict] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Anytime impact-ordered top-k. Pairs with ``retrieve_batch``.

    ``index_or_retriever`` is one of:

    - A dict-shaped CSC index (keys ``data``, ``indices``, ``indptr``,
      and either ``num_docs`` or ``n_docs``). Pass ``impact_ordered=True``
      if the column entries are already descending-impact; else we
      re-permute on the fly (correct but adds an O(nnz log nnz) build).
    - A ``LoadedIndex`` (from ``mojo_bm25s.load_index``). The
      ``impact_ordered`` attribute is honored.
    - A ``bm25s.BM25`` retriever. Its CSC is always doc-id-ordered so
      we permute on the fly.

    ``query_tokens_batch`` is the same shape ``retrieve_batch`` accepts:
    a list of per-query token-id arrays (int32) OR token-string lists
    (only the bm25s-retriever input form supports the latter).

    Returns ``(scores: float32[batch, k], ids: int32[batch, k])`` sorted
    descending per row. Same contract as ``retrieve_batch`` modulo
    rank-k tie-class equivalence — bit-equality on indices is **not**
    guaranteed because impact-order walks ties in a different order.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if num_workers < 0:
        raise ValueError(f"num_workers must be >= 0, got {num_workers}")
    if num_workers == 0:
        import os
        num_workers = os.cpu_count() or 1

    data, indices, indptr, n_docs, _ = _extract_csc_from_input(index_or_retriever)

    # Treat str-token queries only when a retriever was passed.
    retriever = index_or_retriever if hasattr(index_or_retriever, "get_tokens_ids") else None
    _, queries_concat, offsets, batch_size = _convert_queries(
        query_tokens_batch, retriever
    )

    if batch_size > 0:
        _validate_query_token_ids(
            queries_concat, n_vocab=indptr.shape[0] - 1,
            name="query_tokens_batch (concatenated)",
        )

    scores_out = np.zeros((batch_size, k), dtype=np.float32)
    ids_out = np.zeros((batch_size, k), dtype=np.int32)

    if batch_size == 0:
        return scores_out, ids_out

    # Single-element scratch for the debug iteration counter — easier to
    # marshal as a length-1 int64 buffer than a Python int across the FFI.
    counter_buf = np.zeros(1, dtype=np.int64)
    counter_ptr = int(counter_buf.__array_interface__["data"][0])
    want_counter = 1 if _debug_iteration_counters is not None else 0

    _kernel = _get_kernel()
    _kernel.retrieve_batch_anytime(
        (
            int(data.__array_interface__["data"][0]),
            int(indptr.__array_interface__["data"][0]),
            int(indices.__array_interface__["data"][0]),
            int(n_docs),
        ),
        (
            int(queries_concat.__array_interface__["data"][0]),
            int(offsets.__array_interface__["data"][0]),
            int(batch_size),
        ),
        (
            int(scores_out.__array_interface__["data"][0]),
            int(ids_out.__array_interface__["data"][0]),
            int(k),
            int(num_workers),
            int(counter_ptr),
            int(want_counter),
        ),
    )

    if _debug_iteration_counters is not None:
        _debug_iteration_counters["entries_visited"] = int(counter_buf[0])

    return scores_out, ids_out
