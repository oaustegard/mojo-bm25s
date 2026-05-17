"""bm25s backend-swap glue.

Monkey-patches a `bm25s.BM25` retriever in place so that its retrieve
hot paths route through the Mojo kernels — no fork, no subclass.

Two methods get reassigned on the instance:

- ``_compute_relevance_from_scores``: replaced with a closure that
  forwards to `mojo_bm25s.csc_score`. The downstream `weight_mask`
  and `nonoccurrence_array` handling in `get_scores_from_ids`
  continues to run on the Mojo-produced score vector — we only swap
  the inner scatter, not the surrounding logic.

- ``_get_top_k_results``: replaced with a closure that calls the
  retriever's own `get_scores` (so the patched scorer above runs),
  then routes to `mojo_bm25s.topk`. The ``backend`` and ``sorted``
  kwargs from the bm25s signature are accepted-and-ignored because
  Mojo's top-k always uses the heap kernel and always returns
  descending order.
"""

from __future__ import annotations

import numpy as np

from . import csc_score, topk


def patch_bm25s(retriever):
    """Redirect a `bm25s.BM25` retriever's hot paths to Mojo kernels.

    The retriever must already be indexed (this patch reads
    ``retriever.dtype``; it does not touch the index). Returns the same
    retriever for chaining.

    Raises ``ValueError`` if the retriever was built with a non-float32
    score dtype — the Mojo kernels are float32-only and silent
    downcast would distort retrieval scores.
    """
    dtype = np.dtype(getattr(retriever, "dtype", "float32"))
    if dtype != np.float32:
        raise ValueError(
            f"mojo_bm25s.patch_bm25s requires the BM25 index dtype to be "
            f"float32; got {dtype}. Rebuild the retriever with "
            f"`bm25s.BM25(dtype='float32')`."
        )

    def _mojo_compute_relevance_from_scores(
        data: np.ndarray,
        indptr: np.ndarray,
        indices: np.ndarray,
        num_docs: int,
        query_tokens_ids: np.ndarray,
        dtype: np.dtype,
    ) -> np.ndarray:
        if np.dtype(dtype) != np.float32:
            raise ValueError(
                f"mojo_bm25s.patch_bm25s expected float32 retrieve, got "
                f"dtype={dtype}"
            )
        return csc_score(
            data=data,
            indptr=indptr,
            indices=indices,
            query_token_ids=query_tokens_ids,
            n_docs=num_docs,
        )

    retriever._compute_relevance_from_scores = _mojo_compute_relevance_from_scores

    # Capture the bound method so the downstream weight_mask /
    # nonoccurrence handling in get_scores_from_ids still runs on the
    # Mojo-scored vector.
    original_get_scores = retriever.get_scores

    def _mojo_get_top_k_results(
        query_tokens_single,
        k: int = 1000,
        backend: str = "auto",     # accepted, ignored
        sorted: bool = False,      # accepted, ignored (Mojo topk always sorted)
        weight_mask=None,
    ):
        if len(query_tokens_single) == 0:
            scores_q = np.zeros(
                retriever.scores["num_docs"], dtype=np.float32
            )
        else:
            scores_q = original_get_scores(
                query_tokens_single, weight_mask=weight_mask
            )
        return topk(scores_q, k=k)

    retriever._get_top_k_results = _mojo_get_top_k_results

    return retriever
