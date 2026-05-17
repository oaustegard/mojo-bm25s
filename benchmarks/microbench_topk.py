"""Microbenchmark: Mojo top-k vs Numba (closest perf target) vs numpy.

The Numba implementation in `bm25s.numba.selection` is the closest perf
target to beat — it JIT-compiles the same heap algorithm to native code.
The numpy path uses `argpartition + argsort`, which is well-vectorized
but allocates more.

Two Mojo variants are timed: `heap` (O(N log k)) and `quickselect`
(O(N) average), to confirm which to keep as the default. Numba is
warmed up before timing to exclude JIT compilation cost.

Run: ``pixi run bench`` or ``python benchmarks/microbench_topk.py``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from statistics import median
from typing import Callable

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np

import mojo_bm25s
from bm25s.selection import topk as numpy_topk
from bm25s.numba.selection import topk as numba_topk


# (N, k) sweep — small/medium/large corpora at typical retrieval k values.
SWEEPS: list[tuple[int, int]] = [
    (1_000, 10),
    (10_000, 10),
    (10_000, 100),
    (100_000, 10),
    (100_000, 100),
    (1_000_000, 10),
    (1_000_000, 100),
]
REPS = 20
WARMUP = 3


def _time(fn: Callable, *args) -> float:
    """Return median wall-clock time per call in milliseconds."""
    for _ in range(WARMUP):
        fn(*args)
    samples = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        fn(*args)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return median(samples)


def _mojo_heap(scores, k):
    return mojo_bm25s.topk(scores, k, algorithm="heap")


def _mojo_quickselect(scores, k):
    return mojo_bm25s.topk(scores, k, algorithm="quickselect")


def _numpy(scores, k):
    return numpy_topk(scores, k, backend="numpy", sorted=True)


def _numba(scores, k):
    return numba_topk(scores, k, backend="numba", sorted=True)


BACKENDS = [
    ("numpy", _numpy),
    ("numba", _numba),
    ("mojo-heap", _mojo_heap),
    ("mojo-qsel", _mojo_quickselect),
]


def main():
    print(f"{'N':>10} {'k':>5}   " + "  ".join(f"{name:>11}" for name, _ in BACKENDS))
    print("-" * (18 + 13 * len(BACKENDS)))

    rng = np.random.default_rng(seed=42)
    for n, k in SWEEPS:
        scores = rng.random(n, dtype=np.float32)
        row = []
        for _, fn in BACKENDS:
            ms = _time(fn, scores, k)
            row.append(f"{ms:>9.3f} ms")
        print(f"{n:>10} {k:>5}   " + "  ".join(row))


if __name__ == "__main__":
    main()
