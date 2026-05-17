"""Backend factories + retrieve callers for the bench harness.

Three backends, each exposing two calls:

- ``retrieve_one(retriever, query_tokens, k)`` — per-query latency
- ``retrieve_batch(retriever, queries_list, k)`` — batched throughput

The numba backend already batches internally via
``_retrieve_numba_functional``. The mojo backend, as of Path A
(PHASE2.md §4), uses ``mojo_bm25s.retrieve_batch`` for the batched
form; the per-query form goes through the same entry point with
``batch_size=1``. Numpy stays per-query in both forms (bm25s's numpy
backend has no batched native path).
"""

from __future__ import annotations

import bm25s

import mojo_bm25s


def build_retriever(backend: str, corpus_tokens, **bm25_kwargs):
    """Return an indexed retriever for the named backend.

    Note: the mojo backend builds with ``backend="numpy"`` on the bm25s
    side. We don't call ``patch_bm25s`` here because the bench routes
    through ``retrieve_batch`` directly — the monkey-patch is for the
    drop-in API, not the bench fast path.
    """
    if backend == "numpy":
        r = bm25s.BM25(backend="numpy", **bm25_kwargs)
        r.index(corpus_tokens, show_progress=False)
        return r
    if backend == "numba":
        r = bm25s.BM25(backend="numba", **bm25_kwargs)
        r.index(corpus_tokens, show_progress=False)
        return r
    if backend == "mojo":
        r = bm25s.BM25(backend="numpy", **bm25_kwargs)
        r.index(corpus_tokens, show_progress=False)
        return r
    raise ValueError(
        f"unknown backend {backend!r}; choose from numpy, numba, mojo"
    )


def backend_selection(backend: str) -> str:
    """Top-k path bm25s should use; only meaningful for numpy/numba."""
    return "numba" if backend == "numba" else "numpy"


def retrieve_one(backend: str, retriever, query_tokens, k: int):
    """Retrieve top-k for a single query. Used for latency measurement."""
    if backend == "mojo":
        return mojo_bm25s.retrieve_batch(retriever, [query_tokens], k=k)
    return retriever.retrieve(
        [query_tokens], k=k,
        backend_selection=backend_selection(backend),
        show_progress=False,
    )


def retrieve_batch(backend: str, retriever, queries_list, k: int):
    """Retrieve top-k for every query in the batch. Used for throughput."""
    if backend == "mojo":
        return mojo_bm25s.retrieve_batch(retriever, queries_list, k=k)
    return retriever.retrieve(
        queries_list, k=k,
        backend_selection=backend_selection(backend),
        show_progress=False,
    )
