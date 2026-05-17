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
the choice — at the cost of a longer per-path block.

## Three paths (post-#34)

For each query the kernel picks one of three back-ends based on
`upper_bound = Σ (indptr[t+1] - indptr[t])` over query tokens:

  1. **Hashmap path** (`upper_bound < n_docs // 32`) — open-addressed
     `(keys: Int32, vals: Float32)` table sized to `next_pow2(upper_bound * 2)`.
     Scatter writes into the table; topk walks only populated entries.
     `O(touched × log k)` instead of `O(n_docs)` for the full-scan.
  2. **Sparse-reset path** (`upper_bound < n_docs // 8`) — dense
     `scratch[n_docs]` with a `was_touched[n_docs]` companion + a
     `touched` append-list so the post-topk reset is `O(touched)`
     rather than `O(n_docs)`. (Issue #21.)
  3. **Dense path** — full `O(n_docs)` zero-fill. The fallback for
     queries that touch most of the corpus, where the bookkeeping
     overhead of the other two paths isn't worth it.

The thresholds are empirically tuned (see `benchmarks/bench_hashmap.py`).

## Open-addressed hash table (issue #34)

- `keys: List[Int32]` size `next_pow2(upper_bound * 2)`, `-1` = empty slot.
- `vals: List[Float32]` parallel array, same size.
- `touched: List[Int32]` records slot indices of populated entries
  (NOT doc-ids — direct index into keys/vals avoids re-hashing on reset
  and topk).
- Hash: `(d * 0x9E3779B9) >> shift` where `shift = 32 - log2(cap)`.
  Fibonacci multiplier; cheap, decent distribution for sequential doc-ids
  with the high-bit shift mixing.
- Probe: linear, `pos = (pos + 1) & mask`. Cache-friendly at the small
  capacities we operate at (a few hundred to a few thousand slots).
- Reset between queries: walk `touched`, restamp `keys[slot] = -1` and
  `vals[slot] = 0.0`, clear `touched`. `O(touched)`, not `O(cap)`.

Worst-case allocation per worker: `8 * next_pow2(max_upper_bound * 2)`
bytes (Int32 + Float32 per slot). For a query touching 5,000 rows in a
100K-doc corpus, that's `8 * 16384 = 128 KB` per worker — vs `4 * 100,000
= 400 KB` for the dense scratch. Still saves space, and the hot
working set (populated slots only) is far smaller than that.
"""

from std.memory import UnsafePointer
from std.algorithm.functional import parallelize

from topk import topk_heap_impl_ptr, topk_heap_pairs_ptr


def _next_pow2_capacity(min_capacity: Int) -> Int:
    """Smallest power of two >= `min_capacity`, floored at 16.

    The floor avoids degenerate capacities where the Fibonacci-hash
    shift would be > 28. Open-coded `bit_ceil` because the stdlib
    `bit` package isn't reliably exposed across Mojo distribution
    flavors.
    """
    var c = min_capacity
    if c < 16:
        return 16
    # Round up to next power of two.
    var p = 16
    while p < c:
        p = p << 1
    return p


def _shift_for_capacity(cap: Int) -> Int:
    """For a power-of-two `cap`, return the right shift such that
    `(d * 0x9E3779B9) >> shift` lands in `[0, cap)`. `shift = 32 - log2(cap)`.
    """
    var s = 32
    var c = cap
    while c > 1:
        c = c >> 1
        s -= 1
    return s


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
    path_mode: Int = 0,
):
    """For each query: scatter (into hashmap or dense scratch) → topk
    → write row to caller's output.

    Scratch is `List[Float32]`-backed (Mojo-owned, one allocation per
    worker), accessed through the raw pointer returned by `unsafe_ptr()`.
    `scores_out` and `ids_out` are caller-owned ``(batch_size, k)``
    row-major buffers.

    `num_workers` selects dispatch policy:
    - `<= 1` runs the serial path — one scratch reused across all queries.
    - `> 1` partitions the batch into `num_workers` contiguous chunks and
      dispatches via `parallelize`; each worker allocates its own scratch.

    `path_mode` is the per-query path selector:
    - 0 = auto (heuristic on `upper_bound = Σ col_len`).
    - 1 = force dense / sparse-reset (pre-#34 behavior).
    - 2 = force hashmap (issue #34 open-addressed scratch).
    """
    # Heuristic thresholds:
    # - `dense_threshold`: above this we fall back to the full-n_docs
    #   zero-fill path (#21 boundary).
    # - `hashmap_threshold`: tighter — the hashmap path is only worth
    #   it when touched rows are tiny relative to n_docs.
    var dense_threshold = n_docs // 8
    # Hashmap-eligibility gate set at `n_docs // 16` empirically. From
    # `benchmarks/microbench_hashmap.py` on n_docs=100K:
    #   ub/n_docs ratio < 0.03 : hashmap ≈ sparse-reset (both << dense)
    #   ub/n_docs ratio ≈ 0.06 : hashmap is 2.8x faster than sparse-reset
    #                            (the topk scan over n_docs starts to
    #                            dominate at this density)
    #   ub/n_docs ratio > 0.15 : hashmap loses (probe chains lengthen,
    #                            cache footprint of populated entries
    #                            approaches that of dense scratch)
    # Picking n_docs // 16 (ratio 0.0625) captures the 2-3x win at the
    # crossover band while staying safely on the winning side of the
    # high-density falloff.
    var hashmap_threshold = n_docs // 16
    var FIB: UInt32 = 0x9E3779B9

    if num_workers <= 1 or batch_size <= 1:
        # Dense scratch reused across queries (sparse-reset + dense paths).
        var scratch_list = List[Float32](length=n_docs, fill=Float32(0))
        var scratch = scratch_list.unsafe_ptr()
        var was_touched_list = List[Bool](length=n_docs, fill=False)
        var was_touched = was_touched_list.unsafe_ptr()
        var touched = List[Int32]()

        # Hashmap state — re-sized per query, but kept across queries so
        # the underlying List storage gets amortized growth.
        var hm_cap = 16
        var hm_keys = List[Int32](length=hm_cap, fill=Int32(-1))
        var hm_vals = List[Float32](length=hm_cap, fill=Float32(0))
        var hm_touched = List[Int32]()

        for q in range(batch_size):
            var q_start = Int(queries_offsets[q])
            var q_end = Int(queries_offsets[q + 1])

            var expected_touched = 0
            for qt_idx in range(q_start, q_end):
                var t = Int(queries_concat[qt_idx])
                expected_touched += Int(indptr[t + 1]) - Int(indptr[t])

            var use_hashmap: Bool
            var use_sparse_reset: Bool
            if path_mode == 1:
                use_hashmap = False
                use_sparse_reset = expected_touched < dense_threshold
            elif path_mode == 2:
                use_hashmap = True
                use_sparse_reset = False
            else:
                use_hashmap = expected_touched < hashmap_threshold
                use_sparse_reset = (
                    not use_hashmap and expected_touched < dense_threshold
                )

            if use_hashmap:
                # Size table for ≤50% load factor on the upper bound.
                # `expected_touched` is the loosest possible upper bound
                # (counts duplicates); the actual unique-touched count
                # may be smaller, but sizing for the upper bound keeps
                # the load factor safe with no resize logic needed.
                var needed_cap = _next_pow2_capacity(
                    expected_touched * 2 + 1
                )
                if needed_cap > hm_cap:
                    hm_cap = needed_cap
                    hm_keys = List[Int32](length=hm_cap, fill=Int32(-1))
                    hm_vals = List[Float32](length=hm_cap, fill=Float32(0))
                var hm_mask = hm_cap - 1
                var hm_shift = _shift_for_capacity(hm_cap)
                var keys_ptr = hm_keys.unsafe_ptr()
                var vals_ptr = hm_vals.unsafe_ptr()

                # Scatter into the hash table.
                for qt_idx in range(q_start, q_end):
                    var t = Int(queries_concat[qt_idx])
                    var col_start = Int(indptr[t])
                    var col_end = Int(indptr[t + 1])
                    for j in range(col_start, col_end):
                        var row = Int(indices[j])
                        # Fibonacci hash + linear probe.
                        var h = (UInt32(row) * FIB) >> UInt32(hm_shift)
                        var pos = Int(h) & hm_mask
                        while True:
                            var existing = Int(keys_ptr[pos])
                            if existing == -1:
                                # Empty slot: claim it.
                                keys_ptr[pos] = Int32(row)
                                vals_ptr[pos] = data[j]
                                hm_touched.append(Int32(pos))
                                break
                            if existing == row:
                                # Hit: accumulate.
                                vals_ptr[pos] = vals_ptr[pos] + data[j]
                                break
                            pos = (pos + 1) & hm_mask

                # Topk over populated slots only.
                var n_touched = len(hm_touched)
                var hm_touched_ptr = hm_touched.unsafe_ptr()
                var pair = topk_heap_pairs_ptr(
                    keys_ptr, vals_ptr, hm_touched_ptr, n_touched, k
                )
                var values = pair[0].copy()
                var idxs = pair[1].copy()
                var k_actual = len(values)
                for i in range(k_actual):
                    scores_out[q * k + i] = values[i]
                    ids_out[q * k + i] = idxs[i]

                # Reset only the populated slots.
                for i in range(n_touched):
                    var slot = Int(hm_touched[i])
                    keys_ptr[slot] = Int32(-1)
                    vals_ptr[slot] = Float32(0)
                hm_touched.clear()
                continue

            if use_sparse_reset:
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

        # Per-worker hashmap state.
        var hm_cap = 16
        var hm_keys = List[Int32](length=hm_cap, fill=Int32(-1))
        var hm_vals = List[Float32](length=hm_cap, fill=Float32(0))
        var hm_touched = List[Int32]()

        for q in range(q_lo, q_hi):
            var q_start = Int(queries_offsets[q])
            var q_end = Int(queries_offsets[q + 1])

            var expected_touched = 0
            for qt_idx in range(q_start, q_end):
                var t = Int(queries_concat[qt_idx])
                expected_touched += Int(indptr[t + 1]) - Int(indptr[t])

            var use_hashmap: Bool
            var use_sparse_reset: Bool
            if path_mode == 1:
                use_hashmap = False
                use_sparse_reset = expected_touched < dense_threshold
            elif path_mode == 2:
                use_hashmap = True
                use_sparse_reset = False
            else:
                use_hashmap = expected_touched < hashmap_threshold
                use_sparse_reset = (
                    not use_hashmap and expected_touched < dense_threshold
                )

            if use_hashmap:
                var needed_cap = _next_pow2_capacity(
                    expected_touched * 2 + 1
                )
                if needed_cap > hm_cap:
                    hm_cap = needed_cap
                    hm_keys = List[Int32](length=hm_cap, fill=Int32(-1))
                    hm_vals = List[Float32](length=hm_cap, fill=Float32(0))
                var hm_mask = hm_cap - 1
                var hm_shift = _shift_for_capacity(hm_cap)
                var keys_ptr = hm_keys.unsafe_ptr()
                var vals_ptr = hm_vals.unsafe_ptr()

                for qt_idx in range(q_start, q_end):
                    var t = Int(queries_concat[qt_idx])
                    var col_start = Int(indptr[t])
                    var col_end = Int(indptr[t + 1])
                    for j in range(col_start, col_end):
                        var row = Int(indices[j])
                        var h = (UInt32(row) * FIB) >> UInt32(hm_shift)
                        var pos = Int(h) & hm_mask
                        while True:
                            var existing = Int(keys_ptr[pos])
                            if existing == -1:
                                keys_ptr[pos] = Int32(row)
                                vals_ptr[pos] = data[j]
                                hm_touched.append(Int32(pos))
                                break
                            if existing == row:
                                vals_ptr[pos] = vals_ptr[pos] + data[j]
                                break
                            pos = (pos + 1) & hm_mask

                var n_touched = len(hm_touched)
                var hm_touched_ptr = hm_touched.unsafe_ptr()
                var pair = topk_heap_pairs_ptr(
                    keys_ptr, vals_ptr, hm_touched_ptr, n_touched, k
                )
                var values = pair[0].copy()
                var idxs = pair[1].copy()
                var k_actual = len(values)
                for i in range(k_actual):
                    scores_out[q * k + i] = values[i]
                    ids_out[q * k + i] = idxs[i]

                for i in range(n_touched):
                    var slot = Int(hm_touched[i])
                    keys_ptr[slot] = Int32(-1)
                    vals_ptr[slot] = Float32(0)
                hm_touched.clear()
                continue

            if use_sparse_reset:
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
                # maintain the "scratch is zero between queries" invariant.
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

                for d in range(n_docs):
                    scratch[d] = Float32(0)

    parallelize[worker](n_workers, n_workers)
