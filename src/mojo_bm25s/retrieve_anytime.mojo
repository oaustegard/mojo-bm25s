"""Anytime impact-ordered retrieve kernel (issue #35).

Reads an impact-ordered CSC — each column's ``data[j]`` is descending —
and walks each term's posting list with a per-doc upper-bound pruning
check. Standard "term-at-a-time anytime retrieval" (Anh & Moffat,
SIGIR 2006).

Algorithm (per query)
---------------------

Process terms one at a time. Within each term, walk entries in
impact-descending order. Maintain:

- ``scratch[d]`` — running BM25 score accumulator (dense, n_docs).
- ``touched`` / ``was_touched`` — fast reset bookkeeping for ``scratch``.
- ``sum_remaining_max`` — Σ over not-yet-processed terms of that term's
  *max* impact (== ``data[indptr[t]]``, the first / largest entry).
- ``threshold`` — running k-th best score over fully-finalized scratch
  rows. Set to 0 initially; refreshed after each term completes.

The per-entry pruning check for the current term ``t``:

    if scratch[d] + data[j] + sum_remaining_max < threshold:
        break    # remaining entries of t are all <= data[j], can't help

The doc-level "could still enter top-k" check is: any doc's final score
is at most ``scratch[d] + data[j] + sum_remaining_max`` (sums over t's
remaining contributions for this doc plus the max possible from terms
not yet processed). If that's below the current threshold, no doc
beyond position j in this column can reach top-k — terminate the
column's walk.

After each term:
- Decrement ``sum_remaining_max`` by that term's max impact.
- Refresh ``threshold`` = current k-th best score over touched scratch
  rows (a small max-of-k-heap scan).

Final step: read top-k from scratch over touched rows.

Walk order
----------

Terms are walked in **descending order of max impact** (= ``data[indptr[t]]``)
because heavier terms raise the threshold fastest, which improves the
pruning power of later terms. Computed once per query.

CHUNK note
----------

We update the threshold every ``CHUNK`` entries within a single term's
loop (not just between terms) to catch the case where one term's heavy
postings dominate. This is a micro-optimization; the asymptotic bound
is unchanged.

Layout
------

- ``scratch: List[Float32](n_docs)`` — accumulator, zeroed via touched
  list between queries.
- ``was_touched: List[Bool](n_docs)`` — per-row touch flag.
- ``touched: List[Int32]`` — append-only touched-row list.

Output: top-k over scratch's touched rows, sorted descending. Written
into caller's ``(batch, k)`` row-major buffer.
"""

from std.memory import UnsafePointer
from std.algorithm.functional import parallelize


# CHUNK: refresh `threshold` every CHUNK scatter operations within a
# single term's loop. The threshold refresh is O(n_touched log k), so
# small CHUNK degenerates to O(n_touched^2 log k) per term. We pick
# a moderate value (1024) so single-term heavy-tailed queries can
# still early-exit within the column, while keeping per-entry overhead
# low. For multi-term queries the per-term-boundary refresh below
# does the heavy lifting; the intra-CHUNK check is mainly for the
# single-heavy-tail case.
alias CHUNK = 1024


def _compute_threshold(
    mut scratch_list: List[Float32],
    mut touched: List[Int32],
    k: Int,
) -> Float32:
    """Compute the current k-th best score over touched rows.

    Returns 0 if fewer than k rows have been touched (heap not full →
    no valid pruning threshold yet).
    """
    var n_touched = len(touched)
    if n_touched < k:
        return Float32(0)
    # Min-heap of size k over the touched-row values.
    var hv = List[Float32](length=k, fill=Float32(0))
    var n_filled = 0
    for ti in range(n_touched):
        var r = Int(touched[ti])
        var v = scratch_list[r]
        if n_filled < k:
            hv[n_filled] = v
            var pos = n_filled
            while pos > 0:
                var parent = (pos - 1) >> 1
                if hv[pos] < hv[parent]:
                    var tmp = hv[pos]; hv[pos] = hv[parent]; hv[parent] = tmp
                    pos = parent
                else:
                    break
            n_filled += 1
        else:
            if v > hv[0]:
                hv[0] = v
                var pos = 0
                while True:
                    var lc = 2 * pos + 1
                    var rc = 2 * pos + 2
                    var smallest = pos
                    if lc < k and hv[lc] < hv[smallest]:
                        smallest = lc
                    if rc < k and hv[rc] < hv[smallest]:
                        smallest = rc
                    if smallest == pos:
                        break
                    var tmp = hv[pos]; hv[pos] = hv[smallest]; hv[smallest] = tmp
                    pos = smallest
    return hv[0]


def _retrieve_one_query_anytime(
    data: UnsafePointer[Float32, MutExternalOrigin],
    indptr: UnsafePointer[Int32, MutExternalOrigin],
    indices: UnsafePointer[Int32, MutExternalOrigin],
    n_docs: Int,
    queries_concat: UnsafePointer[Int32, MutExternalOrigin],
    queries_offsets: UnsafePointer[Int32, MutExternalOrigin],
    q: Int,
    k: Int,
    scores_out: UnsafePointer[Float32, MutExternalOrigin],
    ids_out: UnsafePointer[Int32, MutExternalOrigin],
    mut scratch_list: List[Float32],
    mut was_touched_list: List[Bool],
    mut touched: List[Int32],
    counter_ptr: UnsafePointer[Int64, MutExternalOrigin],
    want_counter: Int,
) -> Int:
    """Process one query. Returns number of (data, indices) entries
    visited (sum over terms of processed positions)."""
    var q_start = Int(queries_offsets[q])
    var q_end = Int(queries_offsets[q + 1])
    var n_terms = q_end - q_start

    if n_terms == 0:
        return 0

    # Per-term: cache (start, end, max_impact).
    var col_start = List[Int](length=n_terms, fill=0)
    var col_end = List[Int](length=n_terms, fill=0)
    var max_impact = List[Float32](length=n_terms, fill=Float32(0))

    var sum_remaining_max: Float32 = Float32(0)
    for tt in range(n_terms):
        var t = Int(queries_concat[q_start + tt])
        var s = Int(indptr[t])
        var e = Int(indptr[t + 1])
        col_start[tt] = s
        col_end[tt] = e
        if s < e:
            max_impact[tt] = data[s]
        else:
            max_impact[tt] = Float32(0)
        sum_remaining_max = sum_remaining_max + max_impact[tt]

    # Process terms in descending order of max_impact. We keep an
    # index-permutation `order[i] = tt`. Simple O(n_terms^2) selection
    # sort — n_terms is small (typically <= 20).
    var order = List[Int](length=n_terms, fill=0)
    for i in range(n_terms):
        order[i] = i
    for i in range(n_terms):
        var best = i
        for j in range(i + 1, n_terms):
            if max_impact[order[j]] > max_impact[order[best]]:
                best = j
        if best != i:
            var tmp = order[i]; order[i] = order[best]; order[best] = tmp

    var threshold: Float32 = Float32(0)
    var max_scratch_so_far: Float32 = Float32(0)
    var entries_visited: Int = 0

    # Snapshot of max scratch BEFORE the current term starts processing.
    # Each (doc, term) pair appears at most once in the term's column,
    # so within term t's tail, all docs have scratch == their prior-term
    # accumulation (no in-term double-counting). Hence the bound on
    # tail-doc scratch is `max_scratch_at_term_start`, NOT the live
    # `max_scratch_so_far` (which would include scratch values written
    # earlier in the SAME term's loop — those docs are no longer in
    # the tail).
    var max_scratch_at_term_start: Float32 = Float32(0)

    for oi in range(n_terms):
        var tt = order[oi]
        var s = col_start[tt]
        var e = col_end[tt]
        var max_i = max_impact[tt]
        # `sum_after_t` = max contributions of terms NOT YET STARTED.
        # We're about to start term tt; subtract its max from
        # sum_remaining_max first.
        sum_remaining_max = sum_remaining_max - max_i
        var sum_after_t = sum_remaining_max

        # Snapshot max scratch at the start of this term's loop.
        max_scratch_at_term_start = max_scratch_so_far

        var j = s
        var ops_since_refresh: Int = 0
        var early_exit_global = False
        while j < e:
            var d_j = data[j]
            # Tail-break invariant: any doc d in entries j..e-1 of this
            # term (and any LATER term) has scratch[d] bounded by
            # `max_scratch_at_term_start` (we haven't touched d *in this
            # term* yet — guaranteed by the per-(doc, term) uniqueness
            # of impact-ordered postings). Its final score is at most
            #     max_scratch_at_term_start + d_j + sum_after_t.
            # If that's < threshold, no doc reached by entries j or
            # later (within this term OR later terms) can enter top-k.
            #
            # For LATER terms: snapshot at term start was bounded by
            # max_scratch_so_far at that point, which is <= current
            # max_scratch_so_far. So the bound used here is a SAFE
            # upper bound for later terms too (we use this snapshot
            # value as the conservative tail-doc-scratch ceiling).
            if max_scratch_at_term_start + d_j + sum_after_t < threshold:
                # Skip the rest of this term AND every subsequent term.
                # Don't bump j to e — that would inflate
                # entries_visited (we want the COUNT of entries
                # actually scattered, not the column length).
                early_exit_global = True
                break

            var row = Int(indices[j])
            if not was_touched_list[row]:
                was_touched_list[row] = True
                touched.append(Int32(row))
            var new_s = scratch_list[row] + d_j
            scratch_list[row] = new_s
            if new_s > max_scratch_so_far:
                max_scratch_so_far = new_s

            j += 1
            ops_since_refresh += 1
            if ops_since_refresh >= CHUNK:
                threshold = _compute_threshold(scratch_list, touched, k)
                ops_since_refresh = 0

        entries_visited += j - s
        # Refresh threshold once at term boundary even if CHUNK wasn't hit.
        # SKIP for the LAST term — no remaining work to prune. This
        # saves an O(touched log k) pass on the final term.
        if oi < n_terms - 1:
            threshold = _compute_threshold(scratch_list, touched, k)

        if early_exit_global:
            # All remaining terms (including this one's tail) pruned.
            break

    # Final top-k over touched rows.
    var n_touched = len(touched)
    if n_touched == 0:
        if want_counter > 0:
            counter_ptr[0] = counter_ptr[0] + Int64(entries_visited)
        return entries_visited

    var k_eff = k
    if k_eff > n_touched:
        k_eff = n_touched

    var heap_v = List[Float32](length=k_eff, fill=Float32(0))
    var heap_i = List[Int32](length=k_eff, fill=Int32(0))
    var n_in_heap = 0
    for ti in range(n_touched):
        var r = Int(touched[ti])
        var v = scratch_list[r]
        if n_in_heap < k_eff:
            heap_v[n_in_heap] = v
            heap_i[n_in_heap] = Int32(r)
            var pos = n_in_heap
            while pos > 0:
                var parent = (pos - 1) >> 1
                if heap_v[pos] < heap_v[parent]:
                    var tv = heap_v[pos]; heap_v[pos] = heap_v[parent]; heap_v[parent] = tv
                    var ti2 = heap_i[pos]; heap_i[pos] = heap_i[parent]; heap_i[parent] = ti2
                    pos = parent
                else:
                    break
            n_in_heap += 1
        else:
            if v > heap_v[0]:
                heap_v[0] = v
                heap_i[0] = Int32(r)
                var pos = 0
                while True:
                    var lc = 2 * pos + 1
                    var rc = 2 * pos + 2
                    var smallest = pos
                    if lc < k_eff and heap_v[lc] < heap_v[smallest]:
                        smallest = lc
                    if rc < k_eff and heap_v[rc] < heap_v[smallest]:
                        smallest = rc
                    if smallest == pos:
                        break
                    var tv = heap_v[pos]; heap_v[pos] = heap_v[smallest]; heap_v[smallest] = tv
                    var ti2 = heap_i[pos]; heap_i[pos] = heap_i[smallest]; heap_i[smallest] = ti2
                    pos = smallest

    # Descending sort over the k_eff window (selection sort).
    for i in range(k_eff):
        var best = i
        for j in range(i + 1, k_eff):
            if heap_v[j] > heap_v[best]:
                best = j
        if best != i:
            var tv = heap_v[i]; heap_v[i] = heap_v[best]; heap_v[best] = tv
            var ti2 = heap_i[i]; heap_i[i] = heap_i[best]; heap_i[best] = ti2

    for i in range(k_eff):
        scores_out[q * k + i] = heap_v[i]
        ids_out[q * k + i] = heap_i[i]

    if want_counter > 0:
        counter_ptr[0] = counter_ptr[0] + Int64(entries_visited)

    return entries_visited


def retrieve_batch_anytime_into(
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
    counter_ptr: UnsafePointer[Int64, MutExternalOrigin],
    want_counter: Int,
):
    """Anytime retrieve over a batch. Serial path reuses one scratch
    across queries; parallel partitions the batch into contiguous
    chunks, one worker per chunk, each with its own scratch."""

    if num_workers <= 1 or batch_size <= 1:
        var scratch_list = List[Float32](length=n_docs, fill=Float32(0))
        var was_touched_list = List[Bool](length=n_docs, fill=False)
        var touched = List[Int32]()

        for q in range(batch_size):
            _ = _retrieve_one_query_anytime(
                data, indptr, indices, n_docs,
                queries_concat, queries_offsets, q, k,
                scores_out, ids_out,
                scratch_list, was_touched_list, touched,
                counter_ptr, want_counter,
            )
            var n_touched = len(touched)
            for i in range(n_touched):
                var r = Int(touched[i])
                scratch_list[r] = Float32(0)
                was_touched_list[r] = False
            touched.clear()
        return

    # Parallel path.
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
        var was_touched_list = List[Bool](length=n_docs, fill=False)
        var touched = List[Int32]()

        for q in range(q_lo, q_hi):
            _ = _retrieve_one_query_anytime(
                data, indptr, indices, n_docs,
                queries_concat, queries_offsets, q, k,
                scores_out, ids_out,
                scratch_list, was_touched_list, touched,
                counter_ptr, want_counter,
            )
            var n_touched = len(touched)
            for i in range(n_touched):
                var r = Int(touched[i])
                scratch_list[r] = Float32(0)
                was_touched_list[r] = False
            touched.clear()

    parallelize[worker](n_workers, n_workers)
