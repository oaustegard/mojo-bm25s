"""Microbench for issue #35: anytime impact-ordered retrieve vs scan-everything.

Synthetic 100K-doc corpus over a 5K vocabulary (same shape as the #34
hashmap microbench). Runs both ``retrieve_batch`` (scan-everything,
hashmap/dense-auto) and ``retrieve_batch_anytime`` (impact-ordered with
early-exit) over query-length × k grids; reports per-call median ms.

Acceptance signal (from the issue's spec, adapted for CCotw): a
synthetic-bench speedup of ≥ 1.5x at (query_len=5, k=10) would be the
positive signal. Less than that and the per-query overhead doesn't
justify the new index format + kernel. See PR description for the
results.

Usage::

    python benchmarks/microbench_anytime.py
"""

from __future__ import annotations

import time

import numpy as np

import mojo_bm25s


def _make_corpus(n_docs, n_vocab, doc_len_range=(20, 80), seed=0):
    rng = np.random.default_rng(seed)
    return [
        rng.integers(0, n_vocab, size=int(rng.integers(*doc_len_range)),
                     dtype=np.int32)
        for _ in range(n_docs)
    ]


def _make_queries(n_queries, n_vocab, query_len, seed=1):
    rng = np.random.default_rng(seed)
    return [
        rng.integers(0, n_vocab, size=query_len, dtype=np.int32)
        for _ in range(n_queries)
    ]


class _MockRetriever:
    def __init__(self, data, indices, indptr, n_docs, impact_ordered):
        self.scores = {
            "data": data, "indices": indices, "indptr": indptr,
            "num_docs": int(n_docs), "impact_ordered": impact_ordered,
        }


def _time(fn, *args, n_warmup=2, n_runs=5, **kwargs):
    for _ in range(n_warmup):
        fn(*args, **kwargs)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2]


def main():
    n_docs = 100_000
    n_vocab = 5_000
    batch = 100
    print(f"Building corpus n_docs={n_docs} n_vocab={n_vocab} ...")
    docs = _make_corpus(n_docs, n_vocab, seed=42)
    print("Building doc-id-ordered index ...")
    data_doc, idx_doc, indptr_doc, n_docs_b, _, _ = mojo_bm25s.build_index(
        docs, n_vocab=n_vocab, method="lucene"
    )
    print("Building impact-ordered index ...")
    data_imp, idx_imp, indptr_imp, _, _, _ = mojo_bm25s.build_impact_ordered_index(
        docs, n_vocab=n_vocab, method="lucene"
    )
    print()

    r_doc = _MockRetriever(data_doc, idx_doc, indptr_doc, n_docs_b, False)
    r_imp = _MockRetriever(data_imp, idx_imp, indptr_imp, n_docs_b, True)

    print(f"{'query_len':>10} {'k':>5} {'scan ms':>10} {'anytime ms':>12} {'speedup':>10}")
    print("-" * 50)
    for query_len in [2, 5, 20]:
        for k in [10, 100]:
            queries = _make_queries(batch, n_vocab, query_len, seed=99)

            t_scan = _time(
                mojo_bm25s.retrieve_batch, r_doc, queries, k=k, num_workers=1,
            )
            t_any = _time(
                mojo_bm25s.retrieve_batch_anytime, r_imp, queries, k=k, num_workers=1,
            )
            speedup = t_scan / t_any if t_any > 0 else float("inf")
            print(
                f"{query_len:>10d} {k:>5d} "
                f"{t_scan*1000:>10.2f} {t_any*1000:>12.2f} {speedup:>10.2f}x"
            )


if __name__ == "__main__":
    main()
