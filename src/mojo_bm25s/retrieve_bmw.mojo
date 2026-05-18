"""Block-Max WAND retrieval (issue #33).

Per-query top-k pruning via per-block max-impact metadata. Implements the
classical BMW algorithm (Ding & Suel, SIGIR 2011):

  1. Sort query terms by descending current doc-id pointer.
  2. Find pivot term — the smallest-index term in the sorted list whose
     prefix-sum of upper-bounds exceeds the current threshold θ.
  3. Block-max check: if the sum of block-maxes (for the blocks
     containing the pivot doc-id across the live terms) does not exceed
     θ, skip — advance the lagging term's block pointer.
  4. Otherwise, fully score the pivot doc by scattering its actual
     contributions from each posting list; insert into heap if score > θ.

This is the textbook BMW. No MaxScore, no variable block sizes, no
impact-ordered postings — those are out-of-scope follow-ons.

Index-time metadata (computed once by the Python builder, passed in
verbatim):
  - `block_max_impacts[block_offsets[t] : block_offsets[t+1]]`: per-block
    max of `data[indptr[t] + b*B .. min(indptr[t] + (b+1)*B, indptr[t+1])]`.
  - `block_offsets[t]`: first block-index for term t.

Term-level upper bound (used for pivot ordering): `max` over the term's
block-max array.
"""

from std.memory import UnsafePointer
from std.algorithm.functional import parallelize

from topk import topk_heap_pairs_ptr


def _bmw_one_query(
    data: UnsafePointer[Float32, MutExternalOrigin],
    indptr: UnsafePointer[Int32, MutExternalOrigin],
    indices: UnsafePointer[Int32, MutExternalOrigin],
    n_docs: Int,
    block_max_impacts: UnsafePointer[Float32, MutExternalOrigin],
    block_offsets: UnsafePointer[Int32, MutExternalOrigin],
    block_size: Int,
    q_tokens: UnsafePointer[Int32, MutExternalOrigin],
    n_q: Int,
    k: Int,
    out_scores: UnsafePointer[Float32, MutExternalOrigin],
    out_ids: UnsafePointer[Int32, MutExternalOrigin],
):
    """Run BMW for a single query, write top-k into `out_scores`/`out_ids`.

    Per-term cursors are tracked with these parallel lists:
      term_id[i]       — vocab token id of the i-th live term
      col_start[i]     — indptr[term_id[i]]
      col_end[i]       — indptr[term_id[i] + 1]
      cursor[i]        — current j into data/indices for term i
      block_idx[i]     — current block number within term i (== (cursor[i] - col_start[i]) / B)
      n_blocks[i]      — total blocks for term i
      block_max_off[i] — block_offsets[term_id[i]]
      ub[i]            — term-level upper bound (max of term's block-maxes)

    The "current doc" for a term is `indices[cursor[i]]`, sentinel-padded
    to `n_docs` (a value larger than any real doc id) when the term is
    exhausted. Sentinel doc-ids let us avoid a separate "alive" flag.
    """
    # Filter out empty-postings tokens (df=0) and tokens duplicated within
    # the query (we keep only the first occurrence — duplicates would
    # double-count if both contributed). For BM25 the canonical behavior
    # is to also let duplicates double-count (matches scan-everything),
    # so we KEEP duplicates and treat them as separate terms with separate
    # cursors. This matches the scatter-everything path bit-for-bit.
    var n_live = 0
    var term_id = List[Int32](length=n_q, fill=Int32(0))
    var col_start = List[Int32](length=n_q, fill=Int32(0))
    var col_end = List[Int32](length=n_q, fill=Int32(0))
    var cursor = List[Int32](length=n_q, fill=Int32(0))
    var block_idx = List[Int32](length=n_q, fill=Int32(0))
    var n_blocks = List[Int32](length=n_q, fill=Int32(0))
    var block_max_off = List[Int32](length=n_q, fill=Int32(0))
    var ub = List[Float32](length=n_q, fill=Float32(0))
    var cur_doc = List[Int32](length=n_q, fill=Int32(0))

    for i in range(n_q):
        var t = Int(q_tokens[i])
        var cs = Int(indptr[t])
        var ce = Int(indptr[t + 1])
        if cs == ce:
            continue  # empty postings (df=0 token)
        var bo = Int(block_offsets[t])
        var bo_end = Int(block_offsets[t + 1])
        var nb = bo_end - bo
        # Compute term UB = max of block maxes
        var max_ub = Float32(0)
        for b in range(nb):
            var v = block_max_impacts[bo + b]
            if v > max_ub:
                max_ub = v
        term_id[n_live] = Int32(t)
        col_start[n_live] = Int32(cs)
        col_end[n_live] = Int32(ce)
        cursor[n_live] = Int32(cs)
        block_idx[n_live] = Int32(0)
        n_blocks[n_live] = Int32(nb)
        block_max_off[n_live] = Int32(bo)
        ub[n_live] = max_ub
        cur_doc[n_live] = indices[cs]
        n_live += 1

    # Top-k heap as parallel List[Float32]/List[Int32] of size k_eff;
    # min-heap rooted at heap_values[0].
    var k_eff = k
    if k_eff > n_docs:
        k_eff = n_docs
    if k_eff < 0:
        k_eff = 0
    var heap_values = List[Float32](length=k_eff, fill=Float32(0))
    var heap_ids = List[Int32](length=k_eff, fill=Int32(-1))
    var heap_len = 0
    var theta = Float32(0)  # smallest score required to enter top-k

    # Negative-score corpus is impossible for BM25 in practice (we floor
    # at 0 on retrieve), but the heap's "must exceed theta" admits any
    # strictly-positive document on first insert.
    if n_live == 0 or k_eff == 0:
        # No live terms or no slots — write zeros.
        for i in range(k):
            out_scores[i] = Float32(0)
            out_ids[i] = Int32(0)
        return

    # Allocate a small scoring scratch sized to n_live for the current-doc
    # full-score computation. Not n_docs — we score one doc at a time.
    while True:
        # ----- Sort live terms by ascending cur_doc using insertion sort.
        # n_live is bounded by typical query length (<= ~30), so insertion
        # sort is fine. We sort cursor+cur_doc+ub+block_idx in lockstep.
        # The classical WAND requires sort-by-cur-doc; this is what
        # determines the pivot.
        for i in range(1, n_live):
            var key_doc = cur_doc[i]
            var key_t = term_id[i]
            var key_cs = col_start[i]
            var key_ce = col_end[i]
            var key_cur = cursor[i]
            var key_bi = block_idx[i]
            var key_nb = n_blocks[i]
            var key_bo = block_max_off[i]
            var key_ub = ub[i]
            var j = i - 1
            while j >= 0 and cur_doc[j] > key_doc:
                cur_doc[j + 1] = cur_doc[j]
                term_id[j + 1] = term_id[j]
                col_start[j + 1] = col_start[j]
                col_end[j + 1] = col_end[j]
                cursor[j + 1] = cursor[j]
                block_idx[j + 1] = block_idx[j]
                n_blocks[j + 1] = n_blocks[j]
                block_max_off[j + 1] = block_max_off[j]
                ub[j + 1] = ub[j]
                j -= 1
            cur_doc[j + 1] = key_doc
            term_id[j + 1] = key_t
            col_start[j + 1] = key_cs
            col_end[j + 1] = key_ce
            cursor[j + 1] = key_cur
            block_idx[j + 1] = key_bi
            n_blocks[j + 1] = key_nb
            block_max_off[j + 1] = key_bo
            ub[j + 1] = key_ub

        # Drop trailing exhausted terms (cur_doc == sentinel n_docs).
        # Since the array is sorted ascending, exhausted are at the tail.
        while n_live > 0 and Int(cur_doc[n_live - 1]) >= n_docs:
            n_live -= 1
        if n_live == 0:
            break

        # ----- Find pivot term: smallest i such that
        # cumulative_ub[0..=i] > theta. Use sum-then-compare.
        var pivot_idx = -1
        var prefix_sum = Float32(0)
        for i in range(n_live):
            prefix_sum = prefix_sum + ub[i]
            if prefix_sum > theta:
                pivot_idx = i
                break
        if pivot_idx == -1:
            # Even the sum of ALL term UBs <= theta: no remaining doc can
            # enter top-k. Done.
            break

        var pivot_doc = Int(cur_doc[pivot_idx])
        if pivot_doc >= n_docs:
            break

        # Extend pivot_idx to include ALL terms with cur_doc == pivot_doc.
        # This is necessary for the block-max sum (and the score) to be a
        # valid UB on score(pivot_doc): every term whose cur_doc == pivot_doc
        # CAN contribute, so all must be in the block-max sum, not just the
        # ones whose UB happened to push the prefix sum past theta.
        while (
            pivot_idx + 1 < n_live
            and Int(cur_doc[pivot_idx + 1]) == pivot_doc
        ):
            pivot_idx += 1

        # ----- Block-max refinement. For each live term (0..pivot_idx),
        # advance its block pointer until either its current block covers
        # pivot_doc, OR it has moved past pivot_doc. Then sum the block
        # maxes of the relevant blocks and compare to theta.
        # This is the "Block-Max WAND" refinement over classical WAND.

        var pivot_block_sum = Float32(0)
        for i in range(pivot_idx + 1):
            # Advance block pointer until the current block contains doc
            # >= pivot_doc OR cursor has moved into that block.
            var bi = Int(block_idx[i])
            var nb = Int(n_blocks[i])
            var cs = Int(col_start[i])
            var ce = Int(col_end[i])
            var cur = Int(cursor[i])
            # Block b spans [cs + b*B, min(cs + (b+1)*B, ce)).
            # The last doc-id in block b is indices[end-1].
            # We want the smallest b such that the block's last doc-id >= pivot_doc
            # (i.e. the block could contain pivot_doc).
            # Move bi forward as long as the block's last doc < pivot_doc.
            while bi < nb:
                var b_end = cs + (bi + 1) * block_size
                if b_end > ce:
                    b_end = ce
                # Last doc id in this block is indices[b_end - 1].
                # If that doc < pivot_doc, block cannot contain pivot_doc;
                # advance.
                if b_end <= cur:
                    # Block is entirely before the cursor; safe to skip
                    # (the cursor already moved past).
                    bi += 1
                    continue
                var last_doc = Int(indices[b_end - 1])
                if last_doc < pivot_doc:
                    bi += 1
                else:
                    break
            block_idx[i] = Int32(bi)
            if bi >= nb:
                # Term exhausted for blocks
                pivot_block_sum = pivot_block_sum + Float32(0)
            else:
                pivot_block_sum = pivot_block_sum + block_max_impacts[
                    Int(block_max_off[i]) + bi
                ]

        if pivot_block_sum <= theta:
            # Skip pivot_doc. We must advance at least one term in
            # [0..pivot_idx] strictly forward, otherwise we loop forever.
            #
            # Two cases:
            #   (a) Some term has cur_doc < pivot_doc. Advance it forward
            #       to >= pivot_doc (the standard WAND "fast forward").
            #   (b) Every term in [0..pivot_idx] has cur_doc == pivot_doc.
            #       Then pivot_block_sum is the exact UB for this doc;
            #       since it's <= theta this doc can't make the heap.
            #       Advance every such term by 1 to guarantee progress.
            var advanced_any = False
            for i in range(pivot_idx + 1):
                if Int(cur_doc[i]) < pivot_doc:
                    var ce_a = Int(col_end[i])
                    var cur_a = Int(cursor[i])
                    while cur_a < ce_a and Int(indices[cur_a]) < pivot_doc:
                        cur_a += 1
                    cursor[i] = Int32(cur_a)
                    if cur_a >= ce_a:
                        cur_doc[i] = Int32(n_docs)
                        block_idx[i] = Int32(n_blocks[i])
                    else:
                        cur_doc[i] = indices[cur_a]
                        block_idx[i] = Int32(
                            (cur_a - Int(col_start[i])) // block_size
                        )
                    advanced_any = True
                    break  # advance only one term; re-pivot will pick again

            if not advanced_any:
                # Case (b): all cur_doc == pivot_doc; advance all of them.
                for i in range(pivot_idx + 1):
                    if Int(cur_doc[i]) == pivot_doc:
                        var cur_i = Int(cursor[i]) + 1
                        var ce_i = Int(col_end[i])
                        cursor[i] = Int32(cur_i)
                        if cur_i >= ce_i:
                            cur_doc[i] = Int32(n_docs)
                            block_idx[i] = Int32(n_blocks[i])
                        else:
                            cur_doc[i] = indices[cur_i]
                            block_idx[i] = Int32(
                                (cur_i - Int(col_start[i])) // block_size
                            )
            continue

        # ----- Block-max test passed: check whether the leading term's
        # cur_doc equals pivot_doc (all terms 0..pivot_idx must have
        # cur_doc <= pivot_doc; for full score we need cur_doc == pivot_doc
        # for each contributing term, plus terms pivot_idx+1..n_live whose
        # cur_doc happens to equal pivot_doc).
        if Int(cur_doc[0]) < pivot_doc:
            # Advance term 0 toward pivot_doc, then re-pivot.
            var cs0 = Int(col_start[0])
            var ce0 = Int(col_end[0])
            var cur0 = Int(cursor[0])
            var bi0 = Int(block_idx[0])
            var nb0 = Int(n_blocks[0])
            # Skip blocks whose end-doc < pivot_doc.
            while bi0 < nb0:
                var b_end = cs0 + (bi0 + 1) * block_size
                if b_end > ce0:
                    b_end = ce0
                if b_end <= cur0:
                    bi0 += 1
                    continue
                var last_doc = Int(indices[b_end - 1])
                if last_doc < pivot_doc:
                    bi0 += 1
                else:
                    break
            # bi0 now points to the block containing pivot_doc (or term
            # exhausted). Set cursor to max(b_start, cur0), then walk.
            if bi0 >= nb0:
                cursor[0] = Int32(ce0)
                cur_doc[0] = Int32(n_docs)
                block_idx[0] = Int32(nb0)
            else:
                var b_start = cs0 + bi0 * block_size
                if b_start < cur0:
                    b_start = cur0
                cur0 = b_start
                while cur0 < ce0 and Int(indices[cur0]) < pivot_doc:
                    cur0 += 1
                cursor[0] = Int32(cur0)
                if cur0 >= ce0:
                    cur_doc[0] = Int32(n_docs)
                    block_idx[0] = Int32(nb0)
                else:
                    cur_doc[0] = indices[cur0]
                    block_idx[0] = Int32((cur0 - cs0) // block_size)
            continue

        # ----- Full score pivot_doc by walking ALL live terms whose
        # cur_doc == pivot_doc.
        var doc_score = Float32(0)
        for i in range(n_live):
            if Int(cur_doc[i]) == pivot_doc:
                doc_score = doc_score + data[Int(cursor[i])]

        # Heap insert.
        if heap_len < k_eff:
            # Push.
            heap_values[heap_len] = doc_score
            heap_ids[heap_len] = Int32(pivot_doc)
            heap_len += 1
            # Sift-down (bubble up in min-heap terms).
            var pos = heap_len - 1
            while pos > 0:
                var parent = (pos - 1) >> 1
                if heap_values[pos] < heap_values[parent]:
                    var tv = heap_values[pos]
                    heap_values[pos] = heap_values[parent]
                    heap_values[parent] = tv
                    var ti = heap_ids[pos]
                    heap_ids[pos] = heap_ids[parent]
                    heap_ids[parent] = ti
                    pos = parent
                else:
                    break
            if heap_len == k_eff:
                theta = heap_values[0]
        elif doc_score > heap_values[0]:
            heap_values[0] = doc_score
            heap_ids[0] = Int32(pivot_doc)
            # Sift-up (sink the root in the min-heap).
            var pos = 0
            while True:
                var left = 2 * pos + 1
                if left >= heap_len:
                    break
                var smallest = pos
                if heap_values[left] < heap_values[smallest]:
                    smallest = left
                var right = left + 1
                if right < heap_len and heap_values[right] < heap_values[smallest]:
                    smallest = right
                if smallest == pos:
                    break
                var tv = heap_values[pos]
                heap_values[pos] = heap_values[smallest]
                heap_values[smallest] = tv
                var ti = heap_ids[pos]
                heap_ids[pos] = heap_ids[smallest]
                heap_ids[smallest] = ti
                pos = smallest
            theta = heap_values[0]

        # Advance every term whose cur_doc == pivot_doc to its next doc.
        for i in range(n_live):
            if Int(cur_doc[i]) == pivot_doc:
                var cur_i = Int(cursor[i]) + 1
                var ce_i = Int(col_end[i])
                var cs_i = Int(col_start[i])
                cursor[i] = Int32(cur_i)
                if cur_i >= ce_i:
                    cur_doc[i] = Int32(n_docs)
                    block_idx[i] = Int32(n_blocks[i])
                else:
                    cur_doc[i] = indices[cur_i]
                    block_idx[i] = Int32((cur_i - cs_i) // block_size)

    # ----- Drain heap: sort descending into output buffers (selection sort,
    # k_eff is small).
    for i in range(k_eff):
        var best = i
        for j in range(i + 1, heap_len):
            if heap_values[j] > heap_values[best]:
                best = j
        if best != i:
            var tv = heap_values[i]
            heap_values[i] = heap_values[best]
            heap_values[best] = tv
            var ti = heap_ids[i]
            heap_ids[i] = heap_ids[best]
            heap_ids[best] = ti

    for i in range(k):
        if i < heap_len:
            out_scores[i] = heap_values[i]
            out_ids[i] = heap_ids[i]
        else:
            out_scores[i] = Float32(0)
            out_ids[i] = Int32(0)


def retrieve_batch_bmw_into(
    data: UnsafePointer[Float32, MutExternalOrigin],
    indptr: UnsafePointer[Int32, MutExternalOrigin],
    indices: UnsafePointer[Int32, MutExternalOrigin],
    n_docs: Int,
    block_max_impacts: UnsafePointer[Float32, MutExternalOrigin],
    block_offsets: UnsafePointer[Int32, MutExternalOrigin],
    block_size: Int,
    queries_concat: UnsafePointer[Int32, MutExternalOrigin],
    queries_offsets: UnsafePointer[Int32, MutExternalOrigin],
    batch_size: Int,
    k: Int,
    scores_out: UnsafePointer[Float32, MutExternalOrigin],
    ids_out: UnsafePointer[Int32, MutExternalOrigin],
    num_workers: Int,
):
    """Batched BMW retrieve. One Mojo crossing per batch.

    Serial path when `num_workers <= 1 or batch_size <= 1`; otherwise
    chunked-parallel via `parallelize`.

    Each query's `out_scores`/`out_ids` row is `(q * k + i)` in the output
    buffers.
    """
    if num_workers <= 1 or batch_size <= 1:
        for q in range(batch_size):
            var q_start = Int(queries_offsets[q])
            var q_end = Int(queries_offsets[q + 1])
            var n_q = q_end - q_start
            _bmw_one_query(
                data, indptr, indices, n_docs,
                block_max_impacts, block_offsets, block_size,
                queries_concat + q_start, n_q, k,
                scores_out + q * k, ids_out + q * k,
            )
        return

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
        for q in range(q_lo, q_hi):
            var q_start = Int(queries_offsets[q])
            var q_end = Int(queries_offsets[q + 1])
            var n_q = q_end - q_start
            _bmw_one_query(
                data, indptr, indices, n_docs,
                block_max_impacts, block_offsets, block_size,
                queries_concat + q_start, n_q, k,
                scores_out + q * k, ids_out + q * k,
            )

    parallelize[worker](n_workers, n_workers)
