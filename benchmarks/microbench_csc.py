"""Wall-clock microbench: Mojo CSC kernel vs bm25s legacy numpy path.

Measures only the retrieve hot path — given an already-built CSC index,
how long does it take to score a query? Both implementations get the
same float32 inputs and produce the same float32 output (parity tests
in ``tests/test_csc.py`` verify bit-equality).

Usage::

    pixi run python benchmarks/microbench_csc.py
    pixi run python benchmarks/microbench_csc.py --n-docs 100000 --n-queries 500

Headline numbers print to stdout. No regression bar — this is a
diagnostic, not a CI gate.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

# Make the in-tree src/ package importable when run from the repo root.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from bm25s.scoring import _compute_relevance_from_scores_legacy

import mojo_bm25s


def synth_csc(n_docs: int, n_vocab: int, avg_df: int, seed: int = 0):
    """Synthesize a CSC matrix shaped like a BM25 inverted index.

    Each token has Poisson(avg_df) document postings (clamped to
    n_docs). Mirrors the sparsity an NQ-subset index would have at
    these dimensions without needing the dataset on disk.
    """
    rng = np.random.default_rng(seed)
    cols_indices, cols_data = [], []
    indptr = [0]
    for _ in range(n_vocab):
        ndocs_this = max(1, int(rng.poisson(avg_df)))
        ndocs_this = min(ndocs_this, n_docs)
        idx = rng.choice(n_docs, size=ndocs_this, replace=False).astype(np.int32)
        idx.sort()
        vals = rng.uniform(0.01, 5.0, size=ndocs_this).astype(np.float32)
        cols_indices.append(idx)
        cols_data.append(vals)
        indptr.append(indptr[-1] + ndocs_this)
    return (
        np.concatenate(cols_data),
        np.concatenate(cols_indices),
        np.array(indptr, dtype=np.int32),
    )


def time_fn(fn: Callable[[], None], warmup: int = 3, reps: int = 20) -> float:
    """Return median wall-clock seconds across ``reps`` after warmup."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-docs", type=int, default=10_000)
    ap.add_argument("--n-vocab", type=int, default=20_000)
    ap.add_argument("--avg-df", type=int, default=40)
    ap.add_argument("--n-queries", type=int, default=100)
    ap.add_argument("--query-len", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(
        f"Building CSC: n_docs={args.n_docs}, n_vocab={args.n_vocab}, "
        f"avg_df={args.avg_df}"
    )
    data, indices, indptr = synth_csc(
        args.n_docs, args.n_vocab, args.avg_df, seed=args.seed
    )
    nnz = data.shape[0]
    print(f"  nnz={nnz:,}  ({nnz / args.n_vocab:.1f} entries/col avg)")

    rng = np.random.default_rng(args.seed + 1)
    queries = [
        rng.choice(args.n_vocab, size=args.query_len, replace=True).astype(np.int32)
        for _ in range(args.n_queries)
    ]

    # Parity spot-check on first query — fail loudly if the kernel drifted.
    ref0 = _compute_relevance_from_scores_legacy(
        data, indptr, indices, args.n_docs, queries[0], dtype=np.float32
    )
    got0 = mojo_bm25s.csc_score(data, indptr, indices, queries[0], args.n_docs)
    if not np.array_equal(ref0, got0):
        raise SystemExit("PARITY FAILURE — Mojo kernel diverged from legacy reference")

    def run_legacy():
        for q in queries:
            _compute_relevance_from_scores_legacy(
                data, indptr, indices, args.n_docs, q, dtype=np.float32
            )

    def run_mojo():
        for q in queries:
            mojo_bm25s.csc_score(data, indptr, indices, q, args.n_docs)

    print(f"Running {args.n_queries} queries × {args.query_len} tokens each...")
    t_legacy = time_fn(run_legacy)
    t_mojo = time_fn(run_mojo)

    print()
    print(f"  bm25s legacy (numpy add.at): {t_legacy * 1000:7.2f} ms  total")
    print(f"  mojo_bm25s.csc_score:        {t_mojo * 1000:7.2f} ms  total")
    print(f"  per-query (legacy):          {t_legacy / args.n_queries * 1e6:7.1f} us")
    print(f"  per-query (mojo):            {t_mojo / args.n_queries * 1e6:7.1f} us")
    print(f"  speedup (legacy / mojo):     {t_legacy / t_mojo:7.2f}x")


if __name__ == "__main__":
    main()
