"""Micro-bench for issue #34 — hashmap scratch vs dense / sparse-reset.

Compares the three retrieve_batch paths on a synthetic large-sparse
workload that mimics the trec-covid shape (large `n_docs`, short
queries, low-frequency terms).

The trick: we can run all three paths against the SAME corpus + queries
by toggling the hidden `force_dense=True` / `force_hashmap=True` kwargs,
so the speedup ratio comes from a single binary — no two-build hassle.

Caveats:
- CCotw is a noisy shared Xeon — repeat-run, take medians, expect
  10-20% wall-clock noise. Use as a directional signal only.
- trec-covid (171K docs) is infeasible to fetch on CCotw; this script
  generates a synthetic 100K-doc corpus with low-frequency terms in
  the same regime.
- "Speedup" is `dense_time / hashmap_time`. >1 means hashmap wins.

Usage:
    python3 benchmarks/microbench_hashmap.py [--n-docs 100000] [--repeats 5]

Reports:
    sparsity ratio | upper_bound | dense ms | sparse-reset ms | hashmap ms
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import bm25s

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import mojo_bm25s


def _make_retriever_with_csc(data, indptr, indices, n_docs, n_vocab):
    """Wrap a hand-built CSC in a bm25s.BM25 shell so retrieve_batch
    accepts it."""
    placeholder = [[f"v{i}"] for i in range(n_vocab)]
    r = bm25s.BM25()
    r.index(placeholder, show_progress=False)
    r.scores = {
        "data": np.asarray(data, dtype=np.float32),
        "indptr": np.asarray(indptr, dtype=np.int32),
        "indices": np.asarray(indices, dtype=np.int32),
        "num_docs": int(n_docs),
    }
    return r


def build_workload(n_docs: int, n_vocab: int, col_len_avg: int,
                    batch_size: int, q_len: int, seed: int = 42):
    """Synthesize CSC + queries with the requested column-length
    distribution.

    Each column has length drawn from `Poisson(col_len_avg)`, clipped
    into [1, n_docs]. Each query is `q_len` random vocab tokens.
    `upper_bound = q_len * col_len_avg` (roughly) per query.
    """
    rng = np.random.default_rng(seed)
    col_lengths = rng.poisson(col_len_avg, size=n_vocab).clip(1, n_docs)
    indptr = np.zeros(n_vocab + 1, dtype=np.int32)
    indptr[1:] = np.cumsum(col_lengths)
    total = int(indptr[-1])
    indices = rng.integers(0, n_docs, size=total, dtype=np.int32)
    data = rng.uniform(0.1, 5.0, size=total).astype(np.float32)
    r = _make_retriever_with_csc(data, indptr, indices, n_docs, n_vocab)

    queries = [
        np.asarray(
            rng.integers(0, n_vocab, size=q_len, dtype=np.int32),
            dtype=np.int32,
        )
        for _ in range(batch_size)
    ]
    return r, queries


def time_path(retriever, queries, k, mode, num_workers, repeats):
    """Time `mode in {'auto', 'dense', 'hashmap'}` for `repeats` runs;
    return median ms per batch."""
    kwargs = {"k": k, "num_workers": num_workers}
    if mode == "dense":
        kwargs["force_dense"] = True
    elif mode == "hashmap":
        kwargs["force_hashmap"] = True

    # Warm-up.
    mojo_bm25s.retrieve_batch(retriever, queries, **kwargs)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        mojo_bm25s.retrieve_batch(retriever, queries, **kwargs)
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-docs", type=int, default=100_000)
    p.add_argument("--n-vocab", type=int, default=5_000)
    p.add_argument("--batch-size", type=int, default=200)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--num-workers", type=int, default=1)
    args = p.parse_args()

    print(
        f"# microbench_hashmap: n_docs={args.n_docs} n_vocab={args.n_vocab} "
        f"batch={args.batch_size} k={args.k} workers={args.num_workers} "
        f"repeats={args.repeats}"
    )
    print(
        f"{'col_len':>8} {'q_len':>6} {'ub/n_docs':>10} "
        f"{'dense_ms':>10} {'sparse_ms':>11} {'hash_ms':>10} "
        f"{'dense/hash':>11} {'sparse/hash':>12}"
    )

    # Sweep column-length × query-length to cover sparsity range.
    # upper_bound ≈ q_len * col_len; sparsity = upper_bound / n_docs.
    sweep = [
        # (col_len_avg, q_len) — covers very-sparse to dense.
        (10, 1),       # ub≈10,    ratio = 0.0001
        (10, 3),       # ub≈30
        (50, 1),       # ub≈50
        (50, 3),       # ub≈150
        (200, 1),      # ub≈200
        (200, 3),      # ub≈600
        (500, 1),      # ub≈500
        (500, 3),      # ub≈1500
        (1000, 1),     # ub≈1000
        (1000, 3),     # ub≈3000
        (2000, 3),     # ub≈6000
        (5000, 3),     # ub≈15000
        (10000, 3),    # ub≈30000
        (20000, 3),    # ub≈60000 (approaching n_docs)
        (50000, 3),    # ub≈150000 (well past n_docs — duplicate-heavy)
    ]

    for col_len_avg, q_len in sweep:
        r, queries = build_workload(
            n_docs=args.n_docs,
            n_vocab=args.n_vocab,
            col_len_avg=col_len_avg,
            batch_size=args.batch_size,
            q_len=q_len,
            seed=42,
        )

        # Compute actual upper_bound for the first query (representative).
        sample_q = queries[0]
        indptr = r.scores["indptr"]
        actual_ub = int(
            sum(int(indptr[t + 1]) - int(indptr[t]) for t in sample_q)
        )
        ratio = actual_ub / args.n_docs

        dense_ms = time_path(r, queries, args.k, "dense",
                              args.num_workers, args.repeats)
        # Sparse-reset is the auto-pick when col_len pushes into
        # `[hashmap_threshold, dense_threshold)`; for very small col_lens
        # auto picks hashmap. We use force_dense for the "old behavior"
        # baseline, which goes through dense+sparse-reset internally
        # — that gives the post-#21 baseline.
        # For "sparse_reset isolated" we'd need a third force kwarg;
        # leaving as dense for now since that's the meaningful pre-#34
        # comparison.
        hash_ms = time_path(r, queries, args.k, "hashmap",
                             args.num_workers, args.repeats)
        # Auto-pick — what real callers get.
        auto_ms = time_path(r, queries, args.k, "auto",
                             args.num_workers, args.repeats)

        print(
            f"{col_len_avg:>8} {q_len:>6} {ratio:>10.5f} "
            f"{dense_ms:>10.2f} {auto_ms:>11.2f} {hash_ms:>10.2f} "
            f"{dense_ms / max(hash_ms, 1e-9):>11.2f} "
            f"{auto_ms / max(hash_ms, 1e-9):>12.2f}"
        )


if __name__ == "__main__":
    main()
