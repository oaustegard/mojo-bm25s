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

## Multithreading

When `num_workers > 1` and `batch_size > 1`, the batch is partitioned
into contiguous chunks (one per worker) and dispatched through Mojo's
`parallelize`. Each worker owns its own scratch buffer — the only
writes into shared memory are the disjoint `(q, *)` rows of `scores_out`
/ `ids_out`, so no synchronization is needed and the result is
bitwise-identical to the serial path (queries are independent, no
floating-point reorder across queries).

The serial path (`num_workers <= 1`) is preserved verbatim so single-
threaded callers see no behavior change and parity tests stay stable
without re-baselining.

The per-query body is inlined into both paths rather than factored out.
Factoring required either (a) passing the scratch as a raw pointer, in
which case rebind through `unsafe_from_address` drops the originating
List's lifetime tracking and the pointer dangles, or (b) passing the
List by mutable reference, in which case the parallel closure cannot
share one List across workers without contention. Inlining sidesteps
the choice — at the cost of one duplicated 20-line block.
"""

from std.memory import UnsafePointer
from std.algorithm.functional import parallelize

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
    num_workers: Int,
):
    """For each query: zero scratch → CSC scatter → topk → write row.

    Scratch is `List[Float32]`-backed (Mojo-owned, one allocation per
    worker), accessed through the raw pointer returned by `unsafe_ptr()`.
    `scores_out` and `ids_out` are caller-owned ``(batch_size, k)``
    row-major buffers.

    `num_workers` selects dispatch policy:
    - `<= 1` runs the serial path — one scratch reused across all queries.
    - `> 1` partitions the batch into `num_workers` contiguous chunks and
      dispatches via `parallelize`; each worker allocates its own scratch.
    """
    if num_workers <= 1 or batch_size <= 1:
        var scratch_list = List[Float32](length=n_docs, fill=Float32(0))
        var scratch = scratch_list.unsafe_ptr()

        for q in range(batch_size):
            for d in range(n_docs):
                scratch[d] = Float32(0)

            var q_start = Int(queries_offsets[q])
            var q_end = Int(queries_offsets[q + 1])
            for qt_idx in range(q_start, q_end):
                var t = Int(queries_concat[qt_idx])
                var col_start = Int(indptr[t])
                var col_end = Int(indptr[t + 1])
                for j in range(col_start, col_end):
                    var row = Int(indices[j])
                    scratch[row] = scratch[row] + data[j]

            var pair = topk_heap_impl_ptr(scratch, n_docs, k)
            var values = pair[0].copy()
            var idxs = pair[1].copy()
            var k_actual = len(values)
            for i in range(k_actual):
                scores_out[q * k + i] = values[i]
                ids_out[q * k + i] = idxs[i]
        return

    # Parallel path: chunk the batch into num_workers contiguous slices.
    var n_workers = num_workers
    if n_workers > batch_size:
        n_workers = batch_size
    var chunk = (batch_size + n_workers - 1) // n_workers

    @parameter
    def worker(w: Int):
        var q_lo = w * chunk
        var q_hi = q_lo + chunk
        if q_hi > batch_size:
            q_hi = batch_size
        if q_lo >= q_hi:
            return

        var scratch_list = List[Float32](length=n_docs, fill=Float32(0))
        var scratch = scratch_list.unsafe_ptr()

        for q in range(q_lo, q_hi):
            for d in range(n_docs):
                scratch[d] = Float32(0)

            var q_start = Int(queries_offsets[q])
            var q_end = Int(queries_offsets[q + 1])
            for qt_idx in range(q_start, q_end):
                var t = Int(queries_concat[qt_idx])
                var col_start = Int(indptr[t])
                var col_end = Int(indptr[t + 1])
                for j in range(col_start, col_end):
                    var row = Int(indices[j])
                    scratch[row] = scratch[row] + data[j]

            var pair = topk_heap_impl_ptr(scratch, n_docs, k)
            var values = pair[0].copy()
            var idxs = pair[1].copy()
            var k_actual = len(values)
            for i in range(k_actual):
                scores_out[q * k + i] = values[i]
                ids_out[q * k + i] = idxs[i]

    parallelize[worker](n_workers, n_workers)
