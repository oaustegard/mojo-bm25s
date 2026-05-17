"""Micro-bench for issue #19 — SIMD-W=8 scatter on the dense path.

Compares two builds of `mojo_bm25s.retrieve_batch` on a synthetic
workload that forces the DENSE inner path (expected_touched >= n_docs/8).

The way to use this:
1. Build the baseline `.so` (revert retrieve.mojo dense-path lift), copy
   to `build/mojo_bm25s_baseline.so`.
2. Build the SIMD `.so` (current head), keep as `build/mojo_bm25s.so`.
3. This script wraps both into separate sys.modules and times each.

Simplified single-build version: just times the current `.so` against
itself with a warm-up, reports QPS. Use with two builds to compute the
delta manually.

CCotw caveat (see PHASE2.md): the container shares CPU with other
workloads, so wall-clock noise can be 5-15%. Run multiple times and
take the median.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import bm25s

# Make src/ importable.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import mojo_bm25s


def build_dense_workload(n_docs: int, n_vocab: int, avg_col_len: int,
                          batch_size: int, q_len: int, seed: int = 42):
    """Synthesize a CSC + queries that force the dense path.

    `expected_touched` per query == q_len * avg_col_len, must exceed
    `n_docs / 8`. We pick parameters that easily satisfy this.
    """
    rng = np.random.default_rng(seed)
    # Build CSC by column.
    col_lengths = rng.poisson(avg_col_len, size=n_vocab).clip(1, n_docs)
    indptr = np.zeros(n_vocab + 1, dtype=np.int32)
    indptr[1:] = np.cumsum(col_lengths).astype(np.int32)
    total = int(indptr[-1])
    indices = rng.integers(0, n_docs, size=total, dtype=np.int32)
    data = (rng.random(total).astype(np.float32) + 0.1) * 5.0

    # Build a retriever with these scores.
    placeholder = [[f"v{i}"] for i in range(n_vocab)]
    r = bm25s.BM25()
    r.index(placeholder, show_progress=False)
    r.scores = {
        "data": data, "indptr": indptr, "indices": indices,
        "num_docs": int(n_docs),
    }

    # Queries: random tokens from vocab, each of length q_len.
    queries = [
        np.asarray(rng.choice(n_vocab, size=q_len, replace=False), dtype=np.int32)
        for _ in range(batch_size)
    ]

    return r, queries


def time_one(r, queries, k, num_workers, n_iters):
    # Warm-up
    mojo_bm25s.retrieve_batch(r, queries, k=k, num_workers=num_workers)
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        mojo_bm25s.retrieve_batch(r, queries, k=k, num_workers=num_workers)
        times.append(time.perf_counter() - t0)
    return times


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-docs", type=int, default=50000)
    p.add_argument("--n-vocab", type=int, default=5000)
    p.add_argument("--avg-col-len", type=int, default=400)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--q-len", type=int, default=8)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--iters", type=int, default=20)
    args = p.parse_args()

    print(
        f"Dense-path bench: n_docs={args.n_docs} n_vocab={args.n_vocab} "
        f"avg_col_len={args.avg_col_len} batch={args.batch} q_len={args.q_len} "
        f"k={args.k} workers={args.workers} iters={args.iters}"
    )
    expected_touched = args.q_len * args.avg_col_len
    threshold = args.n_docs // 8
    print(
        f"  expected_touched ~= {expected_touched}, "
        f"dense_threshold = {threshold} "
        f"({'DENSE' if expected_touched >= threshold else 'SPARSE'} path)"
    )

    r, queries = build_dense_workload(
        args.n_docs, args.n_vocab, args.avg_col_len,
        args.batch, args.q_len,
    )

    times = time_one(r, queries, args.k, args.workers, args.iters)
    median = statistics.median(times)
    mean = statistics.mean(times)
    stdev = statistics.stdev(times) if len(times) > 1 else 0.0
    p10 = sorted(times)[max(0, len(times)//10)]
    qps = args.batch / median

    print(f"  per-batch median: {median*1000:.3f} ms")
    print(f"  per-batch p10:    {p10*1000:.3f} ms")
    print(f"  per-batch mean:   {mean*1000:.3f} ms (std {stdev*1000:.3f})")
    print(f"  QPS (median):     {qps:.1f}")


if __name__ == "__main__":
    main()
