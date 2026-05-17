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

## Touched-rows sparse reset (issue #21)

The naive `for d in range(n_docs): scratch[d] = 0` is `O(n_docs)` per
query — for trec-covid (171K docs) with very short queries that touch
maybe 250 rows, that's ~700x more work than necessary.

The fix: track which rows we wrote into during the scatter, then reset
only those at the end of the query.

  - `was_touched: List[Bool]` of size `n_docs` — single bit (well, byte)
    per doc telling us if this query wrote into that row yet. Allocated
    once per scratch buffer; reset sparsely per query.
  - `touched: List[Int32]` — the list of rows we appended to this query.
    Allocated once per scratch buffer; cleared per query after reset.

Scatter inserts into `touched` on first-write, then accumulates as
before. After topk, we iterate `touched`, zero `scratch[row]` and
clear `was_touched[row]`, then `touched.clear()`.

For very dense queries (touching most rows) this becomes pure
bookkeeping overhead. We gate on a heuristic: when the per-query
expected scatter work (sum of csc column lengths for this query's
tokens) exceeds `n_docs / 8`, we fall back to the full-zero path —
that's the threshold the issue suggests and the one we ship. The
calculation is cheap (one pass over `q_end - q_start` int32s), and
the gate ensures the dense regression is bounded.
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
    # Heuristic threshold: if a query's expected scatter work (sum of
    # csc column lengths) exceeds this, fall back to full-zero. See
    # module docstring.
    var dense_threshold = n_docs // 8

    if num_workers <= 1 or batch_size <= 1:
        var scratch_list = List[Float32](length=n_docs, fill=Float32(0))
        var scratch = scratch_list.unsafe_ptr()
        var was_touched_list = List[Bool](length=n_docs, fill=False)
        var was_touched = was_touched_list.unsafe_ptr()
        var touched = List[Int32]()

        for q in range(batch_size):
            var q_start = Int(queries_offsets[q])
            var q_end = Int(queries_offsets[q + 1])

            # Estimate scatter work for the dense-query gate.
            var expected_touched = 0
            for qt_idx in range(q_start, q_end):
                var t = Int(queries_concat[qt_idx])
                expected_touched += Int(indptr[t + 1]) - Int(indptr[t])

            if expected_touched < dense_threshold:
                # Sparse path: track touched rows, scatter, then sparse reset.
                for qt_idx in range(q_start, q_end):
                    var t = Int(queries_concat[qt_idx])
                    var col_start = Int(indptr[t])
                    var col_end = Int(indptr[t + 1])
                    for j in range(col_start, col_end):
                        var row = Int(indices[j])
                        if not was_touched[row]:
                            was_touched[row] = True
                            touched.append(Int32(row))
                        scratch[row] = scratch[row] + data[j]

                var pair = topk_heap_impl_ptr(scratch, n_docs, k)
                var values = pair[0].copy()
                var idxs = pair[1].copy()
                var k_actual = len(values)
                for i in range(k_actual):
                    scores_out[q * k + i] = values[i]
                    ids_out[q * k + i] = idxs[i]

                # Sparse reset.
                var n_touched = len(touched)
                for i in range(n_touched):
                    var r = Int(touched[i])
                    scratch[r] = Float32(0)
                    was_touched[r] = False
                touched.clear()
            else:
                # Dense path: scatter then full-zero reset. The invariant
                # both paths maintain is "scratch is fully zero after every
                # query" — that lets the sparse path skip a pre-clean
                # check and trust was_touched.
                #
                # (touched/was_touched were already clean coming into this
                # branch because the previous query — sparse or dense —
                # left them clean.)
                #
                # SIMD-W=8 lift (issue #19): load 8 (index, data) pairs
                # per iteration via wide pointer-load, then iterate 8 lanes
                # scalar to perform the scatter. Writes stay scalar because
                # random-target scatter doesn't safely vectorize without
                # AVX-512 conflict-detect.
                comptime SCATTER_W = 8
                for qt_idx in range(q_start, q_end):
                    var t = Int(queries_concat[qt_idx])
                    var col_start = Int(indptr[t])
                    var col_end = Int(indptr[t + 1])
                    var j = col_start
                    while j + SCATTER_W <= col_end:
                        var idx_vec = (indices + j).load[width=SCATTER_W]()
                        var data_vec = (data + j).load[width=SCATTER_W]()
                        for lane in range(SCATTER_W):
                            var row = Int(idx_vec[lane])
                            scratch[row] = scratch[row] + data_vec[lane]
                        j += SCATTER_W
                    while j < col_end:
                        var row = Int(indices[j])
                        scratch[row] = scratch[row] + data[j]
                        j += 1

                var pair = topk_heap_impl_ptr(scratch, n_docs, k)
                var values = pair[0].copy()
                var idxs = pair[1].copy()
                var k_actual = len(values)
                for i in range(k_actual):
                    scores_out[q * k + i] = values[i]
                    ids_out[q * k + i] = idxs[i]

                for d in range(n_docs):
                    scratch[d] = Float32(0)
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
        var was_touched_list = List[Bool](length=n_docs, fill=False)
        var was_touched = was_touched_list.unsafe_ptr()
        var touched = List[Int32]()

        for q in range(q_lo, q_hi):
            var q_start = Int(queries_offsets[q])
            var q_end = Int(queries_offsets[q + 1])

            var expected_touched = 0
            for qt_idx in range(q_start, q_end):
                var t = Int(queries_concat[qt_idx])
                expected_touched += Int(indptr[t + 1]) - Int(indptr[t])

            if expected_touched < dense_threshold:
                for qt_idx in range(q_start, q_end):
                    var t = Int(queries_concat[qt_idx])
                    var col_start = Int(indptr[t])
                    var col_end = Int(indptr[t + 1])
                    for j in range(col_start, col_end):
                        var row = Int(indices[j])
                        if not was_touched[row]:
                            was_touched[row] = True
                            touched.append(Int32(row))
                        scratch[row] = scratch[row] + data[j]

                var pair = topk_heap_impl_ptr(scratch, n_docs, k)
                var values = pair[0].copy()
                var idxs = pair[1].copy()
                var k_actual = len(values)
                for i in range(k_actual):
                    scores_out[q * k + i] = values[i]
                    ids_out[q * k + i] = idxs[i]

                var n_touched = len(touched)
                for i in range(n_touched):
                    var r = Int(touched[i])
                    scratch[r] = Float32(0)
                    was_touched[r] = False
                touched.clear()
            else:
                # Dense path (parallel): scatter then full-zero reset to
                # maintain the "scratch is zero between queries"
                # invariant. See serial-path comment.
                #
                # SIMD-W=8 lift (issue #19) — same shape as serial dense path.
                comptime SCATTER_W = 8
                for qt_idx in range(q_start, q_end):
                    var t = Int(queries_concat[qt_idx])
                    var col_start = Int(indptr[t])
                    var col_end = Int(indptr[t + 1])
                    var j = col_start
                    while j + SCATTER_W <= col_end:
                        var idx_vec = (indices + j).load[width=SCATTER_W]()
                        var data_vec = (data + j).load[width=SCATTER_W]()
                        for lane in range(SCATTER_W):
                            var row = Int(idx_vec[lane])
                            scratch[row] = scratch[row] + data_vec[lane]
                        j += SCATTER_W
                    while j < col_end:
                        var row = Int(indices[j])
                        scratch[row] = scratch[row] + data[j]
                        j += 1

                var pair = topk_heap_impl_ptr(scratch, n_docs, k)
                var values = pair[0].copy()
                var idxs = pair[1].copy()
                var k_actual = len(values)
                for i in range(k_actual):
                    scores_out[q * k + i] = values[i]
                    ids_out[q * k + i] = idxs[i]

                for d in range(n_docs):
                    scratch[d] = Float32(0)

    parallelize[worker](n_workers, n_workers)
