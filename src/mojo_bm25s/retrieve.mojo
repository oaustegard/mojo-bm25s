"""Batched retrieve: scatter + topk for many queries in one Mojo call.

Path A from PHASE2.md. The Phase 1 monkey-patch produced one Python ↔
Mojo crossing per *kernel* (csc_score + topk) per *query* — three
crossings per retrieve() when you include bm25s's framing. Profiling
showed the boundary cost, not the SIMD math, was what kept Mojo behind
Numba (Numba's JIT inlines the framing too).

This kernel collapses the per-batch crossing count to **one**. The
Python facade allocates the output arrays + scratch metadata once,
then a single `retrieve_batch_into` call runs the entire batch in
Mojo — scatter into a Mojo-owned scratch score buffer, topk on it,
write the top-k scores+ids back to the caller's `(batch, k)` numpy
matrices.

Same parity guarantees as the per-query path: identical scores within
float32 tolerance, IDs in the rank-k tie class. See
`tests/test_retrieve_batch.py` and `tests/parity/test_vs_bm25s.py`.
"""

from std.memory import UnsafePointer

from topk import topk_heap_impl_ptr


def retrieve_batch_into(
    data: UnsafePointer[Float32, MutExternalOrigin],
    indptr: UnsafePointer[Int32, MutExternalOrigin],
    indices: UnsafePointer[Int32, MutExternalOrigin],
    n_docs: Int,
    queries_concat: UnsafePointer[Int32, MutExternalOrigin],
    queries_offsets: UnsafePointer[Int32, MutExternalOrigin],
    batch_size: Int,
    k: Int,
    scores_out: UnsafePointer[Float32, MutExternalOrigin],
    ids_out: UnsafePointer[Int32, MutExternalOrigin],
):
    """For each query: zero scratch → CSC scatter → topk → write row.

    The scratch score buffer is `List[Float32]`-backed (Mojo-owned, one
    allocation per call), but the hot inner loops access it through the
    raw `UnsafePointer` retrieved via `unsafe_ptr()`. List indexing
    has bookkeeping (bounds checks, generic indirection) that the
    pointer path skips — the inner CSC scatter is the bench-critical
    loop, and that's where Mojo was losing ground to Numba's
    LLVM-tuned codegen on x86.

    `scores_out` and `ids_out` are caller-owned ``(batch_size, k)``
    row-major buffers.
    """
    var scratch_list = List[Float32](length=n_docs, fill=Float32(0))
    var scratch = scratch_list.unsafe_ptr()

    for q in range(batch_size):
        # Zero scratch for this query via raw pointer.
        for d in range(n_docs):
            scratch[d] = Float32(0)

        # CSC scatter: accumulate every column the query references.
        # All array accesses go through UnsafePointer indexing — no
        # List-level bookkeeping in the hot loop.
        var q_start = Int(queries_offsets[q])
        var q_end = Int(queries_offsets[q + 1])
        for qt_idx in range(q_start, q_end):
            var t = Int(queries_concat[qt_idx])
            var col_start = Int(indptr[t])
            var col_end = Int(indptr[t + 1])
            for j in range(col_start, col_end):
                var row = Int(indices[j])
                scratch[row] = scratch[row] + data[j]

        # topk reads scratch via the same UnsafePointer — its N-element
        # scan avoids List indexing overhead too.
        var pair = topk_heap_impl_ptr(scratch, n_docs, k)
        var values = pair[0].copy()
        var idxs = pair[1].copy()
        var k_actual = len(values)
        for i in range(k_actual):
            scores_out[q * k + i] = values[i]
            ids_out[q * k + i] = idxs[i]
        # If k_actual < k (corpus smaller than k), the remaining row
        # positions stay at their init (zero) values. Python facade
        # documents this contract.
