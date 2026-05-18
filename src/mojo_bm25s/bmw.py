"""Python facade for Block-Max WAND retrieve (issue #33).

The Mojo kernel lives in ``retrieve_bmw.mojo`` and is wired into the
shared library via ``lib.mojo``. This module:

- Computes ``(block_max_impacts, block_offsets)`` from a built CSC index.
- Calls the kernel via ``mojo_bm25s.kernel.retrieve_batch_bmw`` after
  marshaling all numpy arrays to contiguous int32/float32 with checked
  bounds (same discipline as the scan-everything ``retrieve_batch``
  facade in ``__init__.py``).

Index extension is built as a NEW function (``build_block_max_metadata``)
rather than modifying ``build_index``'s return tuple. Old callers of
``build_index`` get exactly what they got before. BMW is opt-in: callers
who want it call ``build_block_max_metadata`` separately on the CSC
arrays. The metadata is cheap to compute (single pass over ``data``).

The kernel is intentionally a separate entry point — the existing
``retrieve_batch`` (PR #31) stays as the always-available scan-everything
fallback. Per the orchestrator brief, we ship BMW as opt-in and report
its perf honestly; if it doesn't pay off we recommend close-as-tried.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


_INT32_MAX = int(np.iinfo(np.int32).max)


def build_block_max_metadata(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    block_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Partition each term's CSC posting list into ``block_size``-sized
    blocks and return per-block max-impact metadata for Block-Max WAND.

    Parameters
    ----------
    data:
        Float32 CSC ``data`` array, shape ``(nnz,)``. Element ``data[j]``
        is the impact of doc ``indices[j]`` for the term whose column
        ``j`` falls into per ``indptr``.
    indices:
        Int32 CSC ``indices`` array (unused for the max computation —
        we only need ``indptr`` to identify column boundaries — but
        accepted for API symmetry with ``build_index``'s return tuple).
    indptr:
        Int32 CSC ``indptr`` array, shape ``(n_vocab + 1,)``.
    block_size:
        Postings per block. Default 128 (textbook BMW choice from
        Ding & Suel 2011). All blocks are this size except possibly
        the final block of each term, which may be shorter.

    Returns
    -------
    (block_max_impacts, block_offsets):
        - ``block_max_impacts: float32[n_blocks_total]`` — per-block
          max-impact, concatenated across terms in vocab order.
        - ``block_offsets: int32[n_vocab + 1]`` — first block index for
          each term; the term's blocks span
          ``block_max_impacts[block_offsets[t] : block_offsets[t+1]]``.

    Notes
    -----
    Index-size overhead: roughly ``nnz / block_size`` floats per term =
    ``4 * nnz / block_size`` bytes total. At ``B=128`` this is ~3.1% of
    the CSC ``data`` array.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    data = np.ascontiguousarray(data, dtype=np.float32)
    indptr = np.ascontiguousarray(indptr, dtype=np.int32)
    n_vocab = indptr.shape[0] - 1

    # First pass: compute per-term block counts to size the output arrays.
    block_offsets = np.zeros(n_vocab + 1, dtype=np.int32)
    for t in range(n_vocab):
        n_t = int(indptr[t + 1]) - int(indptr[t])
        n_blocks = (n_t + block_size - 1) // block_size  # ceil
        block_offsets[t + 1] = block_offsets[t] + n_blocks

    total_blocks = int(block_offsets[-1])
    block_max_impacts = np.zeros(total_blocks, dtype=np.float32)

    # Second pass: fill per-block maxes via np.maximum.reduceat-style
    # explicit loop (cleaner than reduceat with its sentinel handling).
    for t in range(n_vocab):
        cs = int(indptr[t])
        ce = int(indptr[t + 1])
        if cs == ce:
            continue
        bo = int(block_offsets[t])
        n_blocks = int(block_offsets[t + 1]) - bo
        for b in range(n_blocks):
            b_start = cs + b * block_size
            b_end = min(cs + (b + 1) * block_size, ce)
            block_max_impacts[bo + b] = float(np.max(data[b_start:b_end]))

    return block_max_impacts, block_offsets


def _to_int32_checked(arr: np.ndarray, name: str) -> np.ndarray:
    arr = np.ascontiguousarray(arr)
    if arr.dtype != np.int32 and arr.size:
        amax = int(arr.max())
        amin = int(arr.min())
        if amax > _INT32_MAX or amin < -_INT32_MAX - 1:
            raise OverflowError(
                f"{name} contains values outside int32 range "
                f"(got min={amin}, max={amax})"
            )
    return np.ascontiguousarray(arr, dtype=np.int32)


def _validate_query_token_ids(
    query: np.ndarray, n_vocab: int, name: str = "query_token_ids"
) -> None:
    if query.size == 0:
        return
    qmax = int(query.max())
    qmin = int(query.min())
    if qmin < 0:
        raise IndexError(f"{name} contains negative token id {qmin}")
    if qmax >= n_vocab:
        raise IndexError(
            f"{name} contains token id {qmax} but vocabulary size is "
            f"{n_vocab} (valid range: [0, {n_vocab - 1}])"
        )


def retrieve_batch_bmw(
    retriever,
    query_tokens_batch: Sequence,
    k: int = 10,
    num_workers: int = 0,
    block_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Block-Max WAND batched retrieve.

    Same shape as ``retrieve_batch``. Pre-computes the block-max metadata
    on the fly if the retriever doesn't carry it (cached by attaching to
    the retriever object on first call to amortize across batches).

    Parameters
    ----------
    retriever:
        Either a ``bm25s.BM25`` (after ``.index()``) or an index dict
        with the keys ``data, indptr, indices, num_docs``. If a dict is
        passed it may also carry ``block_max_impacts, block_offsets``
        keys; otherwise we compute them.
    query_tokens_batch:
        Batch of queries. Each query is a list of token strings (we
        convert via ``retriever.get_tokens_ids``) or pre-tokenized
        int32 IDs.
    k:
        Number of top results per query.
    num_workers:
        0 = auto (``os.cpu_count()``), 1 = serial, >1 = parallel chunks.
    block_size:
        Must match the block size used to build the metadata. Default 128.

    Returns
    -------
    (scores, ids): ``(float32[batch, k], int32[batch, k])``, descending
    per row.
    """
    # Lazy import to keep the kernel-load deferred to first use.
    from . import _kernel  # noqa: F401  (loaded via mojo_bm25s/__init__)
    import mojo_bm25s as _mod

    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if num_workers < 0:
        raise ValueError(f"num_workers must be >= 0, got {num_workers}")
    if num_workers == 0:
        import os
        num_workers = os.cpu_count() or 1

    # Extract CSC arrays + retriever-style metadata cache.
    if hasattr(retriever, "scores"):
        scores_dict = retriever.scores
        n_docs = int(scores_dict["num_docs"])
        data = np.ascontiguousarray(scores_dict["data"], dtype=np.float32)
        indptr = _to_int32_checked(scores_dict["indptr"], "indptr")
        indices = _to_int32_checked(scores_dict["indices"], "indices")
    elif isinstance(retriever, dict):
        n_docs = int(retriever["num_docs"])
        data = np.ascontiguousarray(retriever["data"], dtype=np.float32)
        indptr = _to_int32_checked(retriever["indptr"], "indptr")
        indices = _to_int32_checked(retriever["indices"], "indices")
    else:
        raise TypeError(
            f"retriever must be a bm25s.BM25 or index dict, got "
            f"{type(retriever).__name__}"
        )

    # Look for cached metadata on the retriever; cache if absent so
    # repeated retrievals don't recompute.
    cached = (
        getattr(retriever, "_bmw_metadata", None)
        if not isinstance(retriever, dict)
        else retriever.get("_bmw_metadata", None)
    )
    block_max_impacts: np.ndarray
    block_offsets: np.ndarray
    if cached is not None and cached.get("block_size") == block_size:
        block_max_impacts = cached["block_max_impacts"]
        block_offsets = cached["block_offsets"]
    else:
        block_max_impacts, block_offsets = build_block_max_metadata(
            data, indices, indptr, block_size=block_size,
        )
        cache_entry = dict(
            block_max_impacts=block_max_impacts,
            block_offsets=block_offsets,
            block_size=block_size,
        )
        if isinstance(retriever, dict):
            retriever["_bmw_metadata"] = cache_entry
        else:
            try:
                retriever._bmw_metadata = cache_entry
            except (AttributeError, TypeError):
                pass  # frozen or otherwise read-only — recompute each call

    # Marshal queries — same pattern as retrieve_batch.
    batch_size = len(query_tokens_batch)
    lengths64 = np.fromiter(
        (len(q) for q in query_tokens_batch),
        dtype=np.int64, count=batch_size,
    )
    total_tokens = int(lengths64.sum())
    if total_tokens > _INT32_MAX:
        raise OverflowError(
            f"total query tokens {total_tokens} exceeds INT32_MAX"
        )

    token_id_batch: list[np.ndarray] = []
    for q in query_tokens_batch:
        if len(q) == 0:
            token_id_batch.append(np.zeros(0, dtype=np.int32))
        elif isinstance(q[0], str):
            if not hasattr(retriever, "get_tokens_ids"):
                raise TypeError(
                    "string queries require a retriever with "
                    "get_tokens_ids; use int32 IDs for dict-style index"
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

    _validate_query_token_ids(
        queries_concat, n_vocab=indptr.shape[0] - 1,
        name="query_tokens_batch (concatenated)",
    )

    scores_out = np.zeros((batch_size, k), dtype=np.float32)
    ids_out = np.zeros((batch_size, k), dtype=np.int32)

    _mod._kernel.retrieve_batch_bmw(
        (
            int(data.__array_interface__["data"][0]),
            int(indptr.__array_interface__["data"][0]),
            int(indices.__array_interface__["data"][0]),
            int(n_docs),
        ),
        (
            int(block_max_impacts.__array_interface__["data"][0]),
            int(block_offsets.__array_interface__["data"][0]),
            int(block_size),
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
        ),
    )
    return scores_out, ids_out
