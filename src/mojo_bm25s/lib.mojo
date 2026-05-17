"""Mojo entry point for mojo_bm25s.

Exposes Python-callable functions wired into a `PythonModuleBuilder`.
The .so is loaded by `mojo_bm25s/__init__.py` and re-exported as
`mojo_bm25s.<name>`.

Numerical kernels live in `scoring.mojo`, `csc.mojo`, and `topk.mojo`;
this file only orchestrates numpy ↔ Mojo marshaling and runtime dispatch
on method-name strings. Every Python-facing function takes only buffer
addresses (read out via `arr.__array_interface__["data"][0]` on the
Python side) — the inner loops do no Python interop.

The 6-arg cap on `PythonModuleBuilder.def_function` forces the BM25
scalar params to ride in a 5-tuple (`score_tfc`) and forces the CSC
output pointer + length to ride together (`csc_score`); the `__init__.py`
shim re-flattens both so callers see normal signatures.
"""

from std.python import PythonObject, Python
from std.python.bindings import PythonModuleBuilder
from std.os import abort
from std.memory import UnsafePointer

from scoring import (
    _tfc_robertson, _tfc_lucene, _tfc_atire, _tfc_bm25l, _tfc_bm25plus,
    _idf_robertson, _idf_lucene, _idf_atire, _idf_bm25l, _idf_bm25plus,
    tfc_scalar, idf_scalar,
)
from topk import topk_heap_impl, topk_quickselect_impl
from csc import csc_score_into
from retrieve import retrieve_batch_into


def hello() raises -> PythonObject:
    var v = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)
    return PythonObject(Float64(v.reduce_add()))


# ---------------------------------------------------------------------------
# score_tfc — vectorized over `tf_array` via raw pointers.
# Dispatch on the method string happens ONCE outside the hot loop, so
# the per-element body is a pointer load + SIMD-generic kernel call +
# pointer store.
# ---------------------------------------------------------------------------

def score_tfc(
    method: PythonObject, tf_array: PythonObject, params: PythonObject
) raises -> PythonObject:
    """Apply the named TFC kernel element-wise across ``tf_array``.

    ``params`` is a Python 5-tuple ``(l_d, l_avg, k1, b, delta)``.
    Returns a new float32 numpy array of the same length as the input.
    """
    var m = String(py=method)
    var l_d = Float32(Float64(py=params[0]))
    var l_avg = Float32(Float64(py=params[1]))
    var k1 = Float32(Float64(py=params[2]))
    var b = Float32(Float64(py=params[3]))
    var delta = Float32(Float64(py=params[4]))

    var np = Python.import_module("numpy")
    var n = Int(py=tf_array.shape[0])
    var result = np.zeros(n, dtype="float32")

    var tf_ptr = UnsafePointer[Float32, MutExternalOrigin](
        unsafe_from_address=Int(py=tf_array.__array_interface__["data"][0])
    )
    var out_ptr = UnsafePointer[Float32, MutExternalOrigin](
        unsafe_from_address=Int(py=result.__array_interface__["data"][0])
    )

    if m == "robertson":
        for i in range(n):
            var v = SIMD[DType.float32, 1](tf_ptr[i])
            out_ptr[i] = _tfc_robertson[1](v, l_d, l_avg, k1, b, delta)[0]
    elif m == "lucene":
        for i in range(n):
            var v = SIMD[DType.float32, 1](tf_ptr[i])
            out_ptr[i] = _tfc_lucene[1](v, l_d, l_avg, k1, b, delta)[0]
    elif m == "atire":
        for i in range(n):
            var v = SIMD[DType.float32, 1](tf_ptr[i])
            out_ptr[i] = _tfc_atire[1](v, l_d, l_avg, k1, b, delta)[0]
    elif m == "bm25l":
        for i in range(n):
            var v = SIMD[DType.float32, 1](tf_ptr[i])
            out_ptr[i] = _tfc_bm25l[1](v, l_d, l_avg, k1, b, delta)[0]
    elif m == "bm25+":
        for i in range(n):
            var v = SIMD[DType.float32, 1](tf_ptr[i])
            out_ptr[i] = _tfc_bm25plus[1](v, l_d, l_avg, k1, b, delta)[0]
    else:
        raise Error(String("unknown TFC method: ", m))

    return result


def score_idf(
    method: PythonObject, df: PythonObject, n: PythonObject,
    allow_negative: PythonObject,
) raises -> PythonObject:
    """Apply the named IDF kernel to a single ``(df, n)`` pair."""
    var m = String(py=method)
    var df_f = Float32(Float64(py=df))
    var n_f = Float32(Float64(py=n))
    var allow_neg = Bool(py=allow_negative)
    var val = idf_scalar(m, df_f, n_f, allow_neg)
    return PythonObject(Float64(val))


# ---------------------------------------------------------------------------
# score_idf_array — vectorized IDF over a vocab-wide df array. Inputs
# are arbitrary numeric numpy arrays on the Python side (the shim
# coerces to float32 contiguous), so the kernel sees one raw pointer.
# ---------------------------------------------------------------------------

def score_idf_array(
    method: PythonObject,
    df_ptr: PythonObject,
    out_ptr: PythonObject,
    n_elem: PythonObject,
    n_docs: PythonObject,
    allow_negative: PythonObject,
) raises -> PythonObject:
    """Compute IDF for an array of doc-frequencies into ``out_ptr``.

    ``df_ptr`` / ``out_ptr`` are integer buffer addresses (caller owns
    both allocations and dtype invariants). Dispatch on ``method``
    happens once; the per-element body is pure Mojo.
    """
    var m = String(py=method)
    var n = Int(py=n_elem)
    var nd = Float32(Float64(py=n_docs))
    var allow_neg = Bool(py=allow_negative)

    var df = UnsafePointer[Float32, MutExternalOrigin](
        unsafe_from_address=Int(py=df_ptr)
    )
    var out = UnsafePointer[Float32, MutExternalOrigin](
        unsafe_from_address=Int(py=out_ptr)
    )

    if m == "robertson":
        for i in range(n):
            var v = SIMD[DType.float32, 1](df[i])
            out[i] = _idf_robertson[1](v, nd, allow_neg)[0]
    elif m == "lucene":
        for i in range(n):
            var v = SIMD[DType.float32, 1](df[i])
            out[i] = _idf_lucene[1](v, nd)[0]
    elif m == "atire":
        for i in range(n):
            var v = SIMD[DType.float32, 1](df[i])
            out[i] = _idf_atire[1](v, nd)[0]
    elif m == "bm25l":
        for i in range(n):
            var v = SIMD[DType.float32, 1](df[i])
            out[i] = _idf_bm25l[1](v, nd)[0]
    elif m == "bm25+":
        for i in range(n):
            var v = SIMD[DType.float32, 1](df[i])
            out[i] = _idf_bm25plus[1](v, nd)[0]
    else:
        raise Error(String("unknown IDF method: ", m))

    return PythonObject(None)


# ---------------------------------------------------------------------------
# topk — pointer-based input copy into the Mojo-owned working buffer
# (the heap/quickselect kernels need List ownership for mutation), then
# pointer-based write into caller-allocated output numpy arrays.
# ---------------------------------------------------------------------------

def topk(
    algorithm: PythonObject, scores_array: PythonObject, k: PythonObject
) raises -> PythonObject:
    """Select the top-k highest scores from a 1-D float32 array.

    Returns a Python 2-tuple ``(scores: float32[k], indices: int32[k])``
    sorted by descending score. ``algorithm`` is ``"heap"`` or
    ``"quickselect"``.
    """
    var algo = String(py=algorithm)
    var n = Int(py=scores_array.shape[0])
    var k_int = Int(py=k)

    var src = UnsafePointer[Float32, MutExternalOrigin](
        unsafe_from_address=Int(py=scores_array.__array_interface__["data"][0])
    )
    var scores = List[Float32](length=n, fill=Float32(0))
    for i in range(n):
        scores[i] = src[i]

    var result_values: List[Float32]
    var result_indices: List[Int32]
    if algo == "heap":
        var pair = topk_heap_impl(scores, k_int)
        result_values = pair[0].copy()
        result_indices = pair[1].copy()
    elif algo == "quickselect":
        var pair = topk_quickselect_impl(scores, k_int)
        result_values = pair[0].copy()
        result_indices = pair[1].copy()
    else:
        raise Error(String("unknown topk algorithm: ", algo))

    var k_out = len(result_values)
    var np = Python.import_module("numpy")
    var scores_out = np.zeros(k_out, dtype="float32")
    var indices_out = np.zeros(k_out, dtype="int32")

    var out_v = UnsafePointer[Float32, MutExternalOrigin](
        unsafe_from_address=Int(py=scores_out.__array_interface__["data"][0])
    )
    var out_i = UnsafePointer[Int32, MutExternalOrigin](
        unsafe_from_address=Int(py=indices_out.__array_interface__["data"][0])
    )
    for i in range(k_out):
        out_v[i] = result_values[i]
        out_i[i] = result_indices[i]

    var builtins = Python.import_module("builtins")
    var lst = builtins.list()
    lst.append(scores_out)
    lst.append(indices_out)
    return builtins.tuple(lst)


def csc_score(
    data_ptr: PythonObject,
    indptr_ptr: PythonObject,
    indices_ptr: PythonObject,
    query_ptr: PythonObject,
    n_query: PythonObject,
    scores_ptr_and_n_docs: PythonObject,
) raises -> PythonObject:
    """Mojo-level CSC retrieve kernel; pointer-based for hot-path speed.

    All array arguments come in as the integer address of the underlying
    buffer (Python: ``arr.__array_interface__["data"][0]``). The Python
    shim in ``__init__.py`` packs the addresses and dispatches; the kernel
    itself does zero Python interop on the inner loop.

    ``scores_ptr_and_n_docs`` is a 2-tuple ``(scores_out_ptr, n_docs)``
    because ``def_function`` caps positional args at 6.
    """
    var data = UnsafePointer[Float32, MutExternalOrigin](
        unsafe_from_address=Int(py=data_ptr)
    )
    var indptr = UnsafePointer[Int32, MutExternalOrigin](
        unsafe_from_address=Int(py=indptr_ptr)
    )
    var indices = UnsafePointer[Int32, MutExternalOrigin](
        unsafe_from_address=Int(py=indices_ptr)
    )
    var query = UnsafePointer[Int32, MutExternalOrigin](
        unsafe_from_address=Int(py=query_ptr)
    )
    var nq = Int(py=n_query)
    var scores = UnsafePointer[Float32, MutExternalOrigin](
        unsafe_from_address=Int(py=scores_ptr_and_n_docs[0])
    )
    var nd = Int(py=scores_ptr_and_n_docs[1])

    csc_score_into(data, indptr, indices, query, nq, scores, nd)
    return PythonObject(None)


def retrieve_batch(
    matrix_args: PythonObject,
    queries_args: PythonObject,
    out_args: PythonObject,
) raises -> PythonObject:
    """Batched retrieve entry point: one Mojo crossing per batch.

    Three tuple-packed argument groups (to fit the 6-arg `def_function`
    cap and to keep related pointers together):

    - ``matrix_args = (data_ptr, indptr_ptr, indices_ptr, n_docs)``
    - ``queries_args = (queries_concat_ptr, queries_offsets_ptr, batch_size)``
    - ``out_args = (scores_out_ptr, ids_out_ptr, k)``

    All pointers are integer addresses (`arr.__array_interface__["data"][0]`).
    The Python shim in `__init__.py` does all the marshaling.
    """
    var data = UnsafePointer[Float32, MutExternalOrigin](
        unsafe_from_address=Int(py=matrix_args[0])
    )
    var indptr = UnsafePointer[Int32, MutExternalOrigin](
        unsafe_from_address=Int(py=matrix_args[1])
    )
    var indices = UnsafePointer[Int32, MutExternalOrigin](
        unsafe_from_address=Int(py=matrix_args[2])
    )
    var n_docs = Int(py=matrix_args[3])

    var queries = UnsafePointer[Int32, MutExternalOrigin](
        unsafe_from_address=Int(py=queries_args[0])
    )
    var offsets = UnsafePointer[Int32, MutExternalOrigin](
        unsafe_from_address=Int(py=queries_args[1])
    )
    var batch = Int(py=queries_args[2])

    var scores_out = UnsafePointer[Float32, MutExternalOrigin](
        unsafe_from_address=Int(py=out_args[0])
    )
    var ids_out = UnsafePointer[Int32, MutExternalOrigin](
        unsafe_from_address=Int(py=out_args[1])
    )
    var k = Int(py=out_args[2])

    retrieve_batch_into(
        data, indptr, indices, n_docs,
        queries, offsets, batch,
        k, scores_out, ids_out,
    )
    return PythonObject(None)


@export
def PyInit_kernel() -> PythonObject:
    try:
        var m = PythonModuleBuilder("mojo_bm25s.kernel")
        m.def_function[hello]("hello")
        m.def_function[score_tfc]("score_tfc")
        m.def_function[score_idf]("score_idf")
        m.def_function[score_idf_array]("score_idf_array")
        m.def_function[topk]("topk")
        m.def_function[csc_score]("csc_score")
        m.def_function[retrieve_batch]("retrieve_batch")
        return m.finalize()
    except e:
        abort(String("failed to create module: ", e))
