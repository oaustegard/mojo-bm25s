"""Backend factories for the bench harness.

Three backends, same interface — each returns an indexed `bm25s.BM25`
retriever ready to receive `.retrieve(...)` calls:

- ``"numpy"``: stock bm25s with its default numpy scorer + topk
- ``"numba"``: bm25s with the numba-JIT'd hot paths
- ``"mojo"``: bm25s with `mojo_bm25s.patch_bm25s` monkey-patched in

The numba JIT compiles on first query (one-time ~7s), so the harness
warms up before timing.
"""

from __future__ import annotations

import bm25s

import mojo_bm25s


def build_retriever(backend: str, corpus_tokens, **bm25_kwargs):
    """Return an indexed retriever for the named backend.

    ``bm25_kwargs`` is forwarded to ``bm25s.BM25(...)``.
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
        mojo_bm25s.patch_bm25s(r)
        return r
    raise ValueError(
        f"unknown backend {backend!r}; choose from numpy, numba, mojo"
    )


def backend_selection(backend: str) -> str:
    """The kwarg ``bm25s.retrieve`` wants for the top-k path."""
    return "numba" if backend == "numba" else "numpy"
