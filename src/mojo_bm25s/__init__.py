"""Python shim that loads the Mojo-built kernel from ``build/mojo_bm25s.so``.

Re-exports the BM25 scoring kernels with the signatures the test suite
and downstream callers see. The underlying Mojo functions take only
positional ``PythonObject`` args; default-argument handling lives here
on the Python side.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_DIR.parent.parent
_KERNEL_PATH = _REPO_ROOT / "build" / "mojo_bm25s.so"


def _load_kernel():
    if not _KERNEL_PATH.exists():
        raise ImportError(
            f"mojo_bm25s kernel not found at {_KERNEL_PATH}. "
            "Run `pixi run build` first."
        )
    spec = importlib.util.spec_from_file_location(
        "mojo_bm25s.kernel", str(_KERNEL_PATH)
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not create import spec for {_KERNEL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["mojo_bm25s.kernel"] = module
    spec.loader.exec_module(module)
    return module


_kernel = _load_kernel()
hello = _kernel.hello


def score_tfc(
    method: str,
    tf_array: np.ndarray,
    l_d: float,
    l_avg: float,
    k1: float,
    b: float,
    delta: float = 0.0,
) -> np.ndarray:
    """Apply a BM25 term-frequency-component scorer to ``tf_array``.

    ``method`` is one of ``"robertson"``, ``"lucene"``, ``"atire"``,
    ``"bm25l"``, ``"bm25+"``. The input array is coerced to float32
    contiguous before being handed to the Mojo kernel.
    """
    arr = np.ascontiguousarray(tf_array, dtype=np.float32)
    return _kernel.score_tfc(method, arr, (l_d, l_avg, k1, b, delta))


def score_idf(
    method: str, df: float, n: float, allow_negative: bool = False
) -> float:
    """Apply a BM25 inverse-document-frequency scorer to ``(df, n)``.

    ``allow_negative`` is honored only by the ``"robertson"`` variant
    (it's the only one whose bm25s reference accepts the flag); the
    other variants ignore it.
    """
    return _kernel.score_idf(method, df, n, allow_negative)


def topk(
    scores: np.ndarray, k: int, algorithm: str = "heap"
) -> tuple[np.ndarray, np.ndarray]:
    """Return the top-k highest scores and their original indices.

    ``algorithm`` is ``"heap"`` (O(N log k) min-heap, faster for small k)
    or ``"quickselect"`` (O(N) average, faster for large k or large N).
    Output is ``(scores, indices)`` of dtypes ``(float32, int32)``,
    sorted by descending score. Tie-breaking at the rank-k boundary is
    implementation-defined — matches `bm25s.selection.topk(backend='numpy')`
    on scores but may disagree on indices when boundary scores are equal.
    """
    arr = np.ascontiguousarray(scores, dtype=np.float32)
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if k > arr.shape[0]:
        raise ValueError(
            f"k={k} exceeds input length {arr.shape[0]}"
        )
    return _kernel.topk(algorithm, arr, k)


__all__ = ["hello", "score_tfc", "score_idf", "topk"]
