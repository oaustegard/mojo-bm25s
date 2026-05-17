"""BM25 scoring kernels.

5 TFC variants × 5 IDF variants from xhluca/bm25s, ported to Mojo.
Each kernel is generic over `SIMD[DType.float32, w]` so callers can
pick a lane count appropriate for the target ISA; the Phase 1 wrapper
in lib.mojo invokes them with `w=1` (per-element), but a vectorized
caller can call the same functions with `w=8` on AVX2 etc.

`tfc_scalar` and `idf_scalar` are the runtime dispatch entry points:
one string comparison resolves the kernel before the hot loop, so the
loop itself contains no branching.
"""

from std.math import log


# ---------------------------------------------------------------------------
# TFC kernels — `delta` is unused by robertson/lucene/atire but accepted
# uniformly so the wrapper has one signature.
# ---------------------------------------------------------------------------

def _tfc_robertson[w: Int](
    tf: SIMD[DType.float32, w],
    l_d: Float32, l_avg: Float32, k1: Float32, b: Float32, delta: Float32
) -> SIMD[DType.float32, w]:
    return tf / (k1 * ((1.0 - b) + b * l_d / l_avg) + tf)


def _tfc_lucene[w: Int](
    tf: SIMD[DType.float32, w],
    l_d: Float32, l_avg: Float32, k1: Float32, b: Float32, delta: Float32
) -> SIMD[DType.float32, w]:
    return _tfc_robertson[w](tf, l_d, l_avg, k1, b, delta)


def _tfc_atire[w: Int](
    tf: SIMD[DType.float32, w],
    l_d: Float32, l_avg: Float32, k1: Float32, b: Float32, delta: Float32
) -> SIMD[DType.float32, w]:
    return (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * l_d / l_avg))


def _tfc_bm25l[w: Int](
    tf: SIMD[DType.float32, w],
    l_d: Float32, l_avg: Float32, k1: Float32, b: Float32, delta: Float32
) -> SIMD[DType.float32, w]:
    var c = tf / (1.0 - b + b * l_d / l_avg)
    return ((k1 + 1.0) * (c + delta)) / (k1 + c + delta)


def _tfc_bm25plus[w: Int](
    tf: SIMD[DType.float32, w],
    l_d: Float32, l_avg: Float32, k1: Float32, b: Float32, delta: Float32
) -> SIMD[DType.float32, w]:
    var num = (k1 + 1.0) * tf
    var den = k1 * (1.0 - b + b * l_d / l_avg) + tf
    return (num / den) + delta


# ---------------------------------------------------------------------------
# IDF kernels. Robertson alone takes `allow_negative` because its
# clamp-to-1 branch is variant-specific in bm25s.
# ---------------------------------------------------------------------------

def _idf_robertson[w: Int](
    df: SIMD[DType.float32, w], n: Float32, allow_negative: Bool
) -> SIMD[DType.float32, w]:
    var inner = (n - df + 0.5) / (df + 0.5)
    if not allow_negative:
        # Per-lane clamp: inner = max(inner, 1.0). bm25s zeroes the
        # IDF when (N-df+0.5)/(df+0.5) < 1 by virtue of log(1) == 0.
        var ones = SIMD[DType.float32, w](1.0)
        inner = max(inner, ones)
    return log(inner)


def _idf_lucene[w: Int](
    df: SIMD[DType.float32, w], n: Float32
) -> SIMD[DType.float32, w]:
    return log(1.0 + (n - df + 0.5) / (df + 0.5))


def _idf_atire[w: Int](
    df: SIMD[DType.float32, w], n: Float32
) -> SIMD[DType.float32, w]:
    return log(n / df)


def _idf_bm25l[w: Int](
    df: SIMD[DType.float32, w], n: Float32
) -> SIMD[DType.float32, w]:
    return log((n + 1.0) / (df + 0.5))


def _idf_bm25plus[w: Int](
    df: SIMD[DType.float32, w], n: Float32
) -> SIMD[DType.float32, w]:
    return log((n + 1.0) / df)


# ---------------------------------------------------------------------------
# Runtime dispatch on method name. One string comparison resolves the
# kernel; the SIMD math then runs without branching. `w=1` here because
# the Phase 1 Python wrapper iterates element-by-element via PythonObject;
# a vectorized caller can write its own loop calling the underlying
# `_tfc_*[w]` kernels directly.
# ---------------------------------------------------------------------------

def tfc_scalar(
    method: String, tf: Float32,
    l_d: Float32, l_avg: Float32, k1: Float32, b: Float32, delta: Float32
) raises -> Float32:
    var v = SIMD[DType.float32, 1](tf)
    if method == "robertson":
        return _tfc_robertson[1](v, l_d, l_avg, k1, b, delta)[0]
    elif method == "lucene":
        return _tfc_lucene[1](v, l_d, l_avg, k1, b, delta)[0]
    elif method == "atire":
        return _tfc_atire[1](v, l_d, l_avg, k1, b, delta)[0]
    elif method == "bm25l":
        return _tfc_bm25l[1](v, l_d, l_avg, k1, b, delta)[0]
    elif method == "bm25+":
        return _tfc_bm25plus[1](v, l_d, l_avg, k1, b, delta)[0]
    raise Error(String("unknown TFC method: ", method))


def idf_scalar(
    method: String, df: Float32, n: Float32, allow_negative: Bool
) raises -> Float32:
    var v = SIMD[DType.float32, 1](df)
    if method == "robertson":
        return _idf_robertson[1](v, n, allow_negative)[0]
    elif method == "lucene":
        return _idf_lucene[1](v, n)[0]
    elif method == "atire":
        return _idf_atire[1](v, n)[0]
    elif method == "bm25l":
        return _idf_bm25l[1](v, n)[0]
    elif method == "bm25+":
        return _idf_bm25plus[1](v, n)[0]
    raise Error(String("unknown IDF method: ", method))
