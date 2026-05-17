"""Smoke test: verifies the Mojo build artifact loads and runs."""

import math


def test_import_kernel():
    import mojo_bm25s

    assert hasattr(mojo_bm25s, "hello")


def test_hello_returns_simd_reduction():
    import mojo_bm25s

    # lib.mojo builds SIMD[float32, 4](1, 2, 3, 4) and returns reduce_add().
    result = mojo_bm25s.hello()
    assert isinstance(result, float)
    assert math.isclose(result, 10.0)
