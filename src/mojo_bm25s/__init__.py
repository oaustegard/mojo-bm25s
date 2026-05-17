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
    indptr = np.ascontiguousarray(indptr, dtype=np.int32)
    indices = np.ascontiguousarray(indices, dtype=np.int32)
    query = np.ascontiguousarray(query_token_ids, dtype=np.int32)
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
    indptr = np.ascontiguousarray(indptr, dtype=np.int32)
    indices = np.ascontiguousarray(indices, dtype=np.int32)
    query = np.ascontiguousarray(query_token_ids, dtype=np.int32)

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
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    token_id_batch: list[np.ndarray] = []
    for q in query_tokens_batch:
        if len(q) == 0:
            token_id_batch.append(np.zeros(0, dtype=np.int32))
        elif isinstance(q[0], str):
            ids = retriever.get_tokens_ids(q)
            token_id_batch.append(np.asarray(ids, dtype=np.int32))
        else:
            token_id_batch.append(np.asarray(q, dtype=np.int32))

    batch_size = len(token_id_batch)
    lengths = np.fromiter(
        (q.shape[0] for q in token_id_batch), dtype=np.int32, count=batch_size,
    )
    offsets = np.zeros(batch_size + 1, dtype=np.int32)
    np.cumsum(lengths, out=offsets[1:])

    if batch_size > 0:
        queries_concat = np.ascontiguousarray(
            np.concatenate(token_id_batch), dtype=np.int32,
        )
    else:
        queries_concat = np.zeros(0, dtype=np.int32)

    data = np.ascontiguousarray(retriever.scores["data"], dtype=np.float32)
    indptr = np.ascontiguousarray(retriever.scores["indptr"], dtype=np.int32)
    indices = np.ascontiguousarray(retriever.scores["indices"], dtype=np.int32)
    n_docs = int(retriever.scores["num_docs"])

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
        ),
    )
    return scores_out, ids_out


from .patch import patch_bm25s  # noqa: E402


__all__ = [
    "hello",
    "score_tfc",
    "score_idf",
    "score_idf_array",
    "topk",
    "csc_score",
    "csc_score_into",
    "retrieve_batch",
    "patch_bm25s",
]
