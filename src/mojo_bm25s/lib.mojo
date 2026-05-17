"""Mojo entry point for mojo_bm25s.

Exposes Python-callable functions wired into a `PythonModuleBuilder`.
The .so is loaded by `mojo_bm25s/__init__.py` and re-exported as
`mojo_bm25s.<name>`.

Numerical kernels live in `scoring.mojo`; this file only orchestrates
numpy ↔ Mojo marshaling and runtime dispatch on method-name strings.

The Python-facing `score_tfc` takes a 5-tuple of scalar params rather
than five separate args because `PythonModuleBuilder.def_function`
caps at 6 positional args; the `__init__.py` shim re-flattens them so
callers see a normal kwargless signature.
"""

from std.python import PythonObject, Python
from std.python.bindings import PythonModuleBuilder
from std.os import abort

from scoring import tfc_scalar, idf_scalar
from topk import topk_heap_impl, topk_quickselect_impl


def hello() raises -> PythonObject:
    var v = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)
    return PythonObject(Float64(v.reduce_add()))


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
    for i in range(n):
        var tf = Float32(Float64(py=tf_array[i]))
        var val = tfc_scalar(m, tf, l_d, l_avg, k1, b, delta)
        result[i] = PythonObject(Float64(val))
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

    # Marshal numpy → Mojo List once, then run the algorithm in pure Mojo.
    var scores = List[Float32](length=n, fill=Float32(0))
    for i in range(n):
        scores[i] = Float32(Float64(py=scores_array[i]))

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
    for i in range(k_out):
        scores_out[i] = PythonObject(Float64(result_values[i]))
        indices_out[i] = PythonObject(Int(result_indices[i]))

    var builtins = Python.import_module("builtins")
    var lst = builtins.list()
    lst.append(scores_out)
    lst.append(indices_out)
    return builtins.tuple(lst)


@export
def PyInit_kernel() -> PythonObject:
    try:
        var m = PythonModuleBuilder("mojo_bm25s.kernel")
        m.def_function[hello]("hello")
        m.def_function[score_tfc]("score_tfc")
        m.def_function[score_idf]("score_idf")
        m.def_function[topk]("topk")
        return m.finalize()
    except e:
        abort(String("failed to create module: ", e))
