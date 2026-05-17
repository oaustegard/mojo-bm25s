"""Mojo entry point for mojo_bm25s.

Phase 1 stub: exposes a single `hello()` Python-callable function that
constructs a SIMD vector and returns its reduction. Exists only to
exercise the toolchain end-to-end (Mojo compile → shared lib → Python
import) before real BM25 kernels land in subsequent issues.
"""

from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.os import abort


def hello() raises -> PythonObject:
    var v = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)
    return PythonObject(Float64(v.reduce_add()))


@export
def PyInit_kernel() -> PythonObject:
    try:
        var m = PythonModuleBuilder("mojo_bm25s.kernel")
        m.def_function[hello]("hello")
        return m.finalize()
    except e:
        abort(String("failed to create module: ", e))
