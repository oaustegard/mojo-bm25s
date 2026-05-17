"""CSC sparse column-slice + dot accumulator kernel.

Retrieve hot path: given a CSC matrix (data, indices, indptr) and a
query as a vector of token IDs, scatter-accumulate the requested
columns into a per-doc score array.

This mirrors ``bm25s.scoring._compute_relevance_from_scores_legacy``
bit-for-bit when both run on the same float32 inputs — they accumulate
column entries in the same iteration order, so no float-reorder skew.

The Mojo-side function operates on raw float32/int32 buffers; the
Python entry point in lib.mojo unpacks numpy arrays via the
``__array_interface__`` data pointer (no Python-level per-element
iteration).
"""

from std.memory import UnsafePointer


def csc_score_into(
    data: UnsafePointer[Float32, MutExternalOrigin],
    indptr: UnsafePointer[Int32, MutExternalOrigin],
    indices: UnsafePointer[Int32, MutExternalOrigin],
    query_token_ids: UnsafePointer[Int32, MutExternalOrigin],
    n_query: Int,
    scores_out: UnsafePointer[Float32, MutExternalOrigin],
    n_docs: Int,
):
    """Scatter-accumulate CSC columns into a caller-provided score buffer.

    ``scores_out`` MUST be zero-initialized before this call; the kernel
    only adds. Boundary contract: ``indptr`` has length ``n_vocab+1``,
    ``query_token_ids`` are valid indices into ``indptr``, and every
    ``indices[j]`` in a touched column is in ``[0, n_docs)``.

    Iteration order matches ``bm25s.scoring._compute_relevance_from_scores_legacy``:
    outer loop over query tokens, inner loop over the column's entries in
    storage order. Same accumulation order on same float32 inputs gives
    bit-identical results to numpy's ``np.add.at`` on the concatenated
    column slices.
    """
    for q in range(n_query):
        var t = Int(query_token_ids[q])
        var start = Int(indptr[t])
        var end = Int(indptr[t + 1])
        for j in range(start, end):
            var row = Int(indices[j])
            scores_out[row] = scores_out[row] + data[j]
