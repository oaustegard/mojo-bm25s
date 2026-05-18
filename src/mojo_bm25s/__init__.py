"""Python shim that loads the Mojo-built kernel.

Re-exports the BM25 scoring kernels with the signatures the test suite
and downstream callers see. The underlying Mojo functions take only
positional ``PythonObject`` args; default-argument handling lives here
on the Python side.

Kernel-loading search order (see ``_KERNEL_SEARCH_PATHS``):

1. ``<package_dir>/_kernel.so`` — bundled by ``pip install``ed wheels
   (staged by ``scripts/build_wheel.py`` before ``python -m build``).
2. ``<repo_root>/build/mojo_bm25s.so`` — produced by ``pixi run build``
   in the in-tree dev workflow.

A wheel install hits path #1 immediately. A fresh `git clone` + `pixi
run build` hits #2. Both work without environment-detection branching.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_DIR.parent.parent

# Kernel search order: bundled-adjacent-to-package first (wheel install),
# then the in-tree `build/` dir (pixi run build, contributor workflow).
# Keeping both paths means the same `__init__.py` works whether the user
# `pip install`'d a wheel or is running from a freshly-cloned dev tree —
# no environment-detection branching, just a list of locations to check.
_KERNEL_SEARCH_PATHS = (
    _PACKAGE_DIR / "_kernel.so",
    _REPO_ROOT / "build" / "mojo_bm25s.so",
)

_INT32_MAX = int(np.iinfo(np.int32).max)
_INT32_MIN = int(np.iinfo(np.int32).min)


def _to_int32_checked(arr: np.ndarray, name: str) -> np.ndarray:
    """Coerce ``arr`` to int32 contiguous, but raise if any value would
    silently truncate. The Mojo kernels read int32 pointers — a wrapped
    indptr / indices entry sends the kernel walking arbitrary memory.
    """
    arr = np.ascontiguousarray(arr)
    if arr.dtype != np.int32 and arr.size:
        amax = int(arr.max())
        amin = int(arr.min())
        if amax > _INT32_MAX or amin < _INT32_MIN:
            raise OverflowError(
                f"{name} contains values outside int32 range "
                f"[{_INT32_MIN}, {_INT32_MAX}] (got min={amin}, max={amax}); "
                f"Mojo kernels are int32-only."
            )
    return np.ascontiguousarray(arr, dtype=np.int32)


def _validate_query_token_ids(
    query: np.ndarray, n_vocab: int, name: str = "query_token_ids"
) -> None:
    """Reject query token IDs that would index past ``indptr``.

    ``indptr`` has shape ``(n_vocab + 1,)`` so the largest valid token id
    is ``n_vocab - 1``. The Mojo kernel does no bounds checking — an OOB
    id sends ``indptr[t + 1]`` past the buffer.
    """
    if query.size == 0:
        return
    qmax = int(query.max())
    qmin = int(query.min())
    if qmin < 0:
        raise IndexError(
            f"{name} contains negative token id {qmin}"
        )
    if qmax >= n_vocab:
        raise IndexError(
            f"{name} contains token id {qmax} but vocabulary size is "
            f"{n_vocab} (valid range: [0, {n_vocab - 1}])"
        )


def _resolve_kernel_path() -> Path:
    """Return the first existing kernel .so from the search-path list.

    Raises ImportError with the full search list if none exist — most
    common cause is a fresh clone without a build (`pixi run build`).
    """
    for candidate in _KERNEL_SEARCH_PATHS:
        if candidate.exists():
            return candidate
    raise ImportError(
        "mojo_bm25s kernel not found. Searched:\n  "
        + "\n  ".join(str(p) for p in _KERNEL_SEARCH_PATHS)
        + "\nRun `pixi run build` for in-tree dev, or `pip install` "
        "the wheel."
    )


def _load_kernel():
    kernel_path = _resolve_kernel_path()
    spec = importlib.util.spec_from_file_location(
        "mojo_bm25s.kernel", str(kernel_path)
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not create import spec for {kernel_path}")
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


def score_idf_array(
    method: str,
    df_array: np.ndarray,
    n_docs: float,
    allow_negative: bool = False,
) -> np.ndarray:
    """Vectorized IDF: apply the named scorer to every entry of ``df_array``.

    For computing the vocab-wide IDF lookup table at index time. Input
    is coerced to contiguous float32; output is a fresh float32 array
    of the same shape.
    """
    df = np.ascontiguousarray(df_array, dtype=np.float32)
    out = np.zeros(df.shape[0], dtype=np.float32)
    _kernel.score_idf_array(
        method,
        int(df.__array_interface__["data"][0]),
        int(out.__array_interface__["data"][0]),
        int(df.shape[0]),
        float(n_docs),
        bool(allow_negative),
    )
    return out


def csc_score(
    data: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    query_token_ids: np.ndarray,
    n_docs: int,
) -> np.ndarray:
    """CSC retrieve hot path: scatter-accumulate columns into a score vector.

    Parameters mirror ``bm25s.scoring._compute_relevance_from_scores_legacy``
    but reordered to put the query last (Mojo kernels take the immutable
    matrix data first). Returns a fresh ``np.zeros(n_docs, dtype=float32)``
    populated by summing ``data[j]`` into ``scores[indices[j]]`` for each
    column referenced by ``query_token_ids``.

    Arrays are coerced to contiguous float32 (data) / int32 (indptr,
    indices, query_token_ids) before being passed to the Mojo kernel via
    raw buffer pointers — no Python-level per-element iteration.
    """
    data = np.ascontiguousarray(data, dtype=np.float32)
    indptr = _to_int32_checked(indptr, "indptr")
    indices = _to_int32_checked(indices, "indices")
    query = _to_int32_checked(query_token_ids, "query_token_ids")
    _validate_query_token_ids(query, n_vocab=indptr.shape[0] - 1)
    scores = np.zeros(int(n_docs), dtype=np.float32)

    _kernel.csc_score(
        int(data.__array_interface__["data"][0]),
        int(indptr.__array_interface__["data"][0]),
        int(indices.__array_interface__["data"][0]),
        int(query.__array_interface__["data"][0]),
        int(query.shape[0]),
        (int(scores.__array_interface__["data"][0]), int(n_docs)),
    )
    return scores


def csc_score_into(
    data: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    query_token_ids: np.ndarray,
    scores_out: np.ndarray,
) -> None:
    """Zero-allocation CSC retrieve: accumulate into the caller's buffer.

    ``scores_out`` is **mutated in place** — must be float32 contiguous
    of length ``n_docs``. The kernel *adds* into the buffer (does not
    zero it first), so the caller chooses whether to preload priors
    before invoking. Returns ``None``.

    Strict on ``scores_out``'s dtype/layout because silent coercion
    would defeat the zero-copy contract; inputs (``data``, ``indptr``,
    ``indices``, ``query_token_ids``) are coerced like in ``csc_score``.
    """
    if scores_out.dtype != np.float32:
        raise TypeError(
            f"scores_out must be float32, got {scores_out.dtype}"
        )
    if not scores_out.flags["C_CONTIGUOUS"]:
        raise ValueError("scores_out must be C-contiguous")
    if scores_out.ndim != 1:
        raise ValueError(f"scores_out must be 1-D, got shape {scores_out.shape}")

    data = np.ascontiguousarray(data, dtype=np.float32)
    indptr = _to_int32_checked(indptr, "indptr")
    indices = _to_int32_checked(indices, "indices")
    query = _to_int32_checked(query_token_ids, "query_token_ids")
    _validate_query_token_ids(query, n_vocab=indptr.shape[0] - 1)

    _kernel.csc_score(
        int(data.__array_interface__["data"][0]),
        int(indptr.__array_interface__["data"][0]),
        int(indices.__array_interface__["data"][0]),
        int(query.__array_interface__["data"][0]),
        int(query.shape[0]),
        (int(scores_out.__array_interface__["data"][0]), int(scores_out.shape[0])),
    )
    return None


def retrieve_batch(
    retriever,
    query_tokens_batch,
    k: int = 10,
    num_workers: int = 0,
    *,
    force_hashmap: bool = False,
    force_dense: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Batched retrieve in a single Mojo call — Path A from PHASE2.md.

    ``retriever`` is an indexed ``bm25s.BM25``. ``query_tokens_batch``
    is a list of queries; each query is either a list of token strings
    (which we convert via ``retriever.get_tokens_ids``) or a numpy/list
    of int token IDs.

    Returns ``(scores: float32[batch, k], ids: int32[batch, k])`` sorted
    descending per row. Same parity guarantees as the per-query
    ``patch_bm25s`` path (identical scores within float32 tolerance, IDs
    in the rank-k tie class).

    The point of this entry point is to amortize the Python ↔ Mojo
    per-call cost: where the per-query patch crosses the boundary
    O(n_queries) times, this crosses exactly once per batch.

    ``num_workers`` selects the parallel-batch dispatch policy:
    - ``0`` (default) → auto-pick = ``os.cpu_count()``, capped at the
      batch size. Set to ``1`` to force the historical serial path.
    - ``1`` → serial; one scratch buffer reused across queries.
    - ``> 1`` → parallel; one scratch per worker, batch chunked into
      ``num_workers`` contiguous slices. Output is bitwise-identical to
      the serial path (queries are independent).

    ``force_hashmap`` / ``force_dense`` (debug-only, mutually exclusive):
    pin the per-query path selector for parity testing. By default the
    kernel auto-picks per query based on ``Σ col_len`` vs ``n_docs / 8``
    (#21 dense/sparse-reset boundary) vs a tighter ``n_docs / 32`` for
    the hashmap path (issue #34). These kwargs override that heuristic.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if num_workers < 0:
        raise ValueError(f"num_workers must be >= 0, got {num_workers}")
    if num_workers == 0:
        import os
        num_workers = os.cpu_count() or 1
    if force_hashmap and force_dense:
        raise ValueError(
            "force_hashmap and force_dense are mutually exclusive"
        )
    # path_mode: 0=auto, 1=force_dense, 2=force_hashmap.
    if force_dense:
        path_mode = 1
    elif force_hashmap:
        path_mode = 2
    else:
        path_mode = 0

    # Sum lengths in int64 BEFORE materializing per-query arrays — the
    # whole point of the guard is to refuse a workload that would overflow
    # int32 without first allocating GB of int32 buffers.
    batch_size = len(query_tokens_batch)
    lengths64 = np.fromiter(
        (len(q) for q in query_tokens_batch), dtype=np.int64, count=batch_size,
    )
    total_tokens = int(lengths64.sum())
    if total_tokens > _INT32_MAX:
        raise OverflowError(
            f"total query tokens {total_tokens} exceeds int32 INT32_MAX "
            f"({_INT32_MAX}); the kernel uses int32 offsets and would wrap."
        )

    token_id_batch: list[np.ndarray] = []
    for q in query_tokens_batch:
        if len(q) == 0:
            token_id_batch.append(np.zeros(0, dtype=np.int32))
        elif isinstance(q[0], str):
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

    data = np.ascontiguousarray(retriever.scores["data"], dtype=np.float32)
    indptr = _to_int32_checked(retriever.scores["indptr"], "indptr")
    indices = _to_int32_checked(retriever.scores["indices"], "indices")
    n_docs = int(retriever.scores["num_docs"])

    _validate_query_token_ids(
        queries_concat, n_vocab=indptr.shape[0] - 1,
        name="query_tokens_batch (concatenated)",
    )

    scores_out = np.zeros((batch_size, k), dtype=np.float32)
    ids_out = np.zeros((batch_size, k), dtype=np.int32)

    _kernel.retrieve_batch(
        (
            int(data.__array_interface__["data"][0]),
            int(indptr.__array_interface__["data"][0]),
            int(indices.__array_interface__["data"][0]),
            int(n_docs),
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
            int(path_mode),
        ),
    )
    return scores_out, ids_out


from .patch import patch_bm25s  # noqa: E402
from .stem import stem, stem_corpus  # noqa: E402
from .tokenize import tokenize, ENGLISH_STOPWORDS  # noqa: E402
from .vocab import Vocab  # noqa: E402
from .index_builder import build_index, build_impact_ordered_index  # noqa: E402
from .io import save_index, load_index  # noqa: E402
from .retriever import Retriever  # noqa: E402
from .anytime import retrieve_batch_anytime  # noqa: E402


__all__ = [
    "hello",
    "score_tfc",
    "score_idf",
    "score_idf_array",
    "topk",
    "csc_score",
    "csc_score_into",
    "retrieve_batch",
    "retrieve_batch_anytime",
    "patch_bm25s",
    "stem",
    "stem_corpus",
    "tokenize",
    "ENGLISH_STOPWORDS",
    "Vocab",
    "build_index",
    "build_impact_ordered_index",
    "save_index",
    "load_index",
    "Retriever",
]
