"""Top-k selection kernels.

Two algorithms exposed for benchmarking:

- `topk_heap`: O(N log k) min-heap of size k. Single pass over the input.
  Cache-friendly when k is small (k ≤ 100) because the working set fits
  in L1.

- `topk_quickselect`: O(N) average via partition, then sort of the top k.
  Better when k is large relative to N or N is very large.

Both return two parallel buffers (scores, indices) sorted by descending
score. Tie-breaking is implementation-defined for equal-score boundary
elements (`bm25s.selection.topk` exhibits the same indeterminacy).

`topk_heap_impl_ptr` is the pointer-input variant used by
`retrieve.retrieve_batch_into` to avoid List indexing overhead on the
N-element scan (the small k-sized heap itself stays as List, since
the mutating sift helpers want List ownership).
"""

from std.memory import UnsafePointer


# ---------------------------------------------------------------------------
# Min-heap helpers — direct port of bm25s/numba/selection.py:32+ (sift_down
# and sift_up). Operate on parallel `values` (scores) and `indices` buffers.
# A min-heap means values[0] is the smallest, so an incoming score replaces
# the root iff it exceeds the current minimum.
# ---------------------------------------------------------------------------

def _sift_down(
    mut values: List[Float32],
    mut indices: List[Int32],
    startpos: Int,
    pos_in: Int,
):
    var pos = pos_in
    var new_value = values[pos]
    var new_index = indices[pos]
    while pos > startpos:
        var parentpos = (pos - 1) >> 1
        var parent_value = values[parentpos]
        if new_value < parent_value:
            values[pos] = parent_value
            indices[pos] = indices[parentpos]
            pos = parentpos
            continue
        break
    values[pos] = new_value
    indices[pos] = new_index


def _sift_up(
    mut values: List[Float32],
    mut indices: List[Int32],
    pos_in: Int,
    length: Int,
):
    var startpos = pos_in
    var pos = pos_in
    var new_value = values[pos]
    var new_index = indices[pos]
    var childpos = 2 * pos + 1
    while childpos < length:
        var rightpos = childpos + 1
        if rightpos < length and values[rightpos] < values[childpos]:
            childpos = rightpos
        values[pos] = values[childpos]
        indices[pos] = indices[childpos]
        pos = childpos
        childpos = 2 * pos + 1
    values[pos] = new_value
    indices[pos] = new_index
    _sift_down(values, indices, startpos, pos)


def _heap_push(
    mut values: List[Float32],
    mut indices: List[Int32],
    value: Float32,
    index: Int32,
    length: Int,
):
    values[length] = value
    indices[length] = index
    _sift_down(values, indices, 0, length)


# ---------------------------------------------------------------------------
# Algorithm 1: min-heap of size k. After one linear pass, the heap holds
# the top-k unsorted; a final descending sort puts them in rank order.
# ---------------------------------------------------------------------------

def topk_heap_impl(
    scores: List[Float32], k: Int
) -> Tuple[List[Float32], List[Int32]]:
    var n = len(scores)
    var k_eff = k
    if k_eff > n:
        k_eff = n

    var values = List[Float32](length=k_eff, fill=Float32(0))
    var indices = List[Int32](length=k_eff, fill=Int32(0))
    var length = 0

    for i in range(n):
        var v = scores[i]
        if length < k_eff:
            _heap_push(values, indices, v, Int32(i), length)
            length += 1
        else:
            if v > values[0]:
                values[0] = v
                indices[0] = Int32(i)
                _sift_up(values, indices, 0, length)

    # Sort heap contents in descending order. k_eff is small (≤ a few
    # hundred typical), so an in-place selection sort is fine — and
    # avoids allocating a permutation buffer.
    for i in range(k_eff):
        var best = i
        for j in range(i + 1, k_eff):
            if values[j] > values[best]:
                best = j
        if best != i:
            var tv = values[i]
            var ti = indices[i]
            values[i] = values[best]
            indices[i] = indices[best]
            values[best] = tv
            indices[best] = ti

    return (values^, indices^)


# ---------------------------------------------------------------------------
# Pointer-input variant. Same min-heap, but the N-element input scan
# reads via `UnsafePointer[Float32]` instead of `List[Float32]`. Used
# by `retrieve.retrieve_batch_into` where the scratch buffer is
# already exposed as a raw pointer; this avoids one source of List
# indexing overhead per scratch read on every query.
# ---------------------------------------------------------------------------

def topk_heap_impl_ptr(
    scores: UnsafePointer[Float32, _],
    n: Int,
    k: Int,
) -> Tuple[List[Float32], List[Int32]]:
    var k_eff = k
    if k_eff > n:
        k_eff = n

    var values = List[Float32](length=k_eff, fill=Float32(0))
    var indices = List[Int32](length=k_eff, fill=Int32(0))
    var length = 0

    for i in range(n):
        var v = scores[i]
        if length < k_eff:
            _heap_push(values, indices, v, Int32(i), length)
            length += 1
        else:
            if v > values[0]:
                values[0] = v
                indices[0] = Int32(i)
                _sift_up(values, indices, 0, length)

    # Sort heap contents descending.
    for i in range(k_eff):
        var best = i
        for j in range(i + 1, k_eff):
            if values[j] > values[best]:
                best = j
        if best != i:
            var tv = values[i]
            var ti = indices[i]
            values[i] = values[best]
            indices[i] = indices[best]
            values[best] = tv
            indices[best] = ti

    return (values^, indices^)


# ---------------------------------------------------------------------------
# Pairs variant for the hash-map scratch (issue #34).
#
# Walks a list of populated slot indices into parallel `keys` (doc-ids,
# Int32) and `vals` (Float32) arrays — i.e. only the populated entries
# of an open-addressed hash table, not the whole slot array. This is
# the payoff of the hash-map data structure: the topk scan is O(touched
# × log k) instead of O(n_docs × log k).
#
# The heap kernel itself is the same min-heap of size k as
# `topk_heap_impl_ptr`; only the input-side scan changes.
# ---------------------------------------------------------------------------


def topk_heap_pairs_ptr(
    keys: UnsafePointer[Int32, _],
    vals: UnsafePointer[Float32, _],
    touched: UnsafePointer[Int32, _],
    n_touched: Int,
    k: Int,
) -> Tuple[List[Float32], List[Int32]]:
    """Top-k from a hash-map's populated slots.

    ``touched[i]`` for ``i in 0..n_touched`` is the slot index of the
    i-th populated entry. ``keys[slot]`` is the doc-id stored at that
    slot; ``vals[slot]`` is its accumulated score. Returns parallel
    (scores, doc_ids) lists of length ``min(k, n_touched)``, sorted
    by descending score.

    A min-heap of size ``k`` filters the populated stream. After the
    scan the heap contents are sorted descending (selection-sort over
    the k-sized window — k is small in practice).
    """
    var k_eff = k
    if k_eff > n_touched:
        k_eff = n_touched
    if k_eff <= 0:
        return (List[Float32](), List[Int32]())

    var values = List[Float32](length=k_eff, fill=Float32(0))
    var indices = List[Int32](length=k_eff, fill=Int32(0))
    var length = 0

    for i in range(n_touched):
        var slot = Int(touched[i])
        var v = vals[slot]
        var doc_id = keys[slot]
        if length < k_eff:
            _heap_push(values, indices, v, doc_id, length)
            length += 1
        else:
            if v > values[0]:
                values[0] = v
                indices[0] = doc_id
                _sift_up(values, indices, 0, length)

    # Descending selection-sort over the heap window.
    for i in range(k_eff):
        var best = i
        for j in range(i + 1, k_eff):
            if values[j] > values[best]:
                best = j
        if best != i:
            var tv = values[i]
            var ti = indices[i]
            values[i] = values[best]
            indices[i] = indices[best]
            values[best] = tv
            indices[best] = ti

    return (values^, indices^)


# ---------------------------------------------------------------------------
# Algorithm 2: quickselect on a copy of the score buffer paired with
# original-index tracking. After partitioning so that the top-k elements
# occupy positions [0, k), sort that prefix descending. O(N) average,
# O(N²) worst-case (degenerate pivots), but median-of-three pivot
# selection keeps the worst case rare for realistic score distributions.
# ---------------------------------------------------------------------------

def _partition(
    mut values: List[Float32],
    mut indices: List[Int32],
    lo: Int,
    hi: Int,
) -> Int:
    """Lomuto partition with median-of-three pivot. Partitions so the
    pivot value's final position separates `>= pivot` (left) from
    `< pivot` (right) — we want the **largest** elements at the front.
    Returns the final pivot index."""
    # Median-of-three pivot selection.
    var mid = (lo + hi) // 2
    if values[mid] > values[lo]:
        var tv = values[lo]; values[lo] = values[mid]; values[mid] = tv
        var ti = indices[lo]; indices[lo] = indices[mid]; indices[mid] = ti
    if values[hi] > values[lo]:
        var tv = values[lo]; values[lo] = values[hi]; values[hi] = tv
        var ti = indices[lo]; indices[lo] = indices[hi]; indices[hi] = ti
    if values[mid] > values[hi]:
        var tv = values[mid]; values[mid] = values[hi]; values[hi] = tv
        var ti = indices[mid]; indices[mid] = indices[hi]; indices[hi] = ti
    # values[hi] is now the median; use it as the pivot.

    var pivot = values[hi]
    var i = lo - 1
    for j in range(lo, hi):
        if values[j] > pivot:
            i += 1
            var tv = values[i]; values[i] = values[j]; values[j] = tv
            var ti = indices[i]; indices[i] = indices[j]; indices[j] = ti
    i += 1
    var tv = values[i]; values[i] = values[hi]; values[hi] = tv
    var ti = indices[i]; indices[i] = indices[hi]; indices[hi] = ti
    return i


def _quickselect(
    mut values: List[Float32],
    mut indices: List[Int32],
    lo_in: Int,
    hi_in: Int,
    k_target: Int,
):
    """In-place partition so that values[0..k_target) are the k_target
    largest of values[lo_in..hi_in]. Iterative loop (no recursion) to
    keep stack usage bounded."""
    var lo = lo_in
    var hi = hi_in
    while lo < hi:
        var p = _partition(values, indices, lo, hi)
        if p == k_target - 1:
            return
        elif p < k_target - 1:
            lo = p + 1
        else:
            hi = p - 1


def topk_quickselect_impl(
    scores: List[Float32], k: Int
) -> Tuple[List[Float32], List[Int32]]:
    var n = len(scores)
    var k_eff = k
    if k_eff > n:
        k_eff = n

    # Working copies — quickselect is in-place destructive.
    var values = List[Float32](length=n, fill=Float32(0))
    var indices = List[Int32](length=n, fill=Int32(0))
    for i in range(n):
        values[i] = scores[i]
        indices[i] = Int32(i)

    if k_eff < n:
        _quickselect(values, indices, 0, n - 1, k_eff)

    # Truncate to k_eff, then sort descending.
    var top_v = List[Float32](length=k_eff, fill=Float32(0))
    var top_i = List[Int32](length=k_eff, fill=Int32(0))
    for i in range(k_eff):
        top_v[i] = values[i]
        top_i[i] = indices[i]

    # Descending insertion sort over the k_eff window.
    for i in range(1, k_eff):
        var v = top_v[i]
        var idx = top_i[i]
        var j = i - 1
        while j >= 0 and top_v[j] < v:
            top_v[j + 1] = top_v[j]
            top_i[j + 1] = top_i[j]
            j -= 1
        top_v[j + 1] = v
        top_i[j + 1] = idx

    return (top_v^, top_i^)
