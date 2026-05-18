"""Tests for Block-Max WAND retrieval (issue #33).

Contract (locked here):

- ``build_block_max_metadata(data, indices, indptr, block_size=128)`` partitions
  each term's posting list into blocks of size ``block_size`` (last block may
  be short) and returns ``(block_max_impacts, block_offsets)``.

  - ``block_max_impacts[block_offsets[t] : block_offsets[t+1]]`` is the per-block
    maximum impact for term ``t``.
  - For a term with ``n_t = indptr[t+1] - indptr[t]`` postings, the number of
    blocks is ``ceil(n_t / block_size)``.

- ``retrieve_batch_bmw(retriever_or_index_dict, query_tokens_batch, k,
  num_workers, block_size)`` returns ``(scores, ids)`` matching the
  scan-everything ``retrieve_batch`` within ``atol=1e-5`` (on scores) and
  with IDs in the rank-k tie class.

- BMW indexes round-trip through save_index / load_index when the
  ``block_max_impacts`` / ``block_offsets`` are passed. Indexes without
  block-max metadata still load correctly (backward compat).

- Edge cases: empty query, single-token query, all-zero overlap, k > n_docs,
  block_size not dividing nnz.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

import bm25s

import mojo_bm25s


# ----------------------------------------------------------------------
# Module surface — the simplest possible early TDD signal: the names must
# be exported. If they aren't, every subsequent assertion is ImportError
# noise.
# ----------------------------------------------------------------------

def test_module_exports_bmw_symbols():
    assert hasattr(mojo_bm25s, "build_block_max_metadata"), (
        "mojo_bm25s.build_block_max_metadata must be exported (issue #33)"
    )
    assert hasattr(mojo_bm25s, "retrieve_batch_bmw"), (
        "mojo_bm25s.retrieve_batch_bmw must be exported (issue #33)"
    )
    assert callable(mojo_bm25s.build_block_max_metadata)
    assert callable(mojo_bm25s.retrieve_batch_bmw)


# ----------------------------------------------------------------------
# build_block_max_metadata — hand-built CSC, hand-checked block maxes
# ----------------------------------------------------------------------

def test_block_max_metadata_simple_known_values():
    """Hand-built CSC with 3 terms and B=4. Term-0 has 6 postings, term-1
    has 4, term-2 has 2 — covers full block, exact-fit, partial block.
    """
    # CSC: 3 terms, n_docs = 16, B=4
    # term 0 (6 postings): data = [1.0, 3.0, 2.0, 5.0, 4.0, 0.5]
    #   block 0 (4): max = 5.0
    #   block 1 (2): max = 4.0
    # term 1 (4 postings): data = [10.0, 2.0, 3.0, 8.0]
    #   block 0 (4): max = 10.0
    # term 2 (2 postings): data = [0.1, 0.2]
    #   block 0 (2): max = 0.2
    data = np.array(
        [1.0, 3.0, 2.0, 5.0, 4.0, 0.5, 10.0, 2.0, 3.0, 8.0, 0.1, 0.2],
        dtype=np.float32,
    )
    indices = np.array(
        [0, 1, 2, 3, 4, 5,  0, 1, 2, 3,  0, 1], dtype=np.int32
    )
    indptr = np.array([0, 6, 10, 12], dtype=np.int32)

    bmax, boff = mojo_bm25s.build_block_max_metadata(
        data, indices, indptr, block_size=4
    )
    # Per-term block counts: ceil(6/4)=2, ceil(4/4)=1, ceil(2/4)=1. Total = 4.
    assert boff.tolist() == [0, 2, 3, 4]
    np.testing.assert_array_equal(bmax, np.array([5.0, 4.0, 10.0, 0.2], dtype=np.float32))


def test_block_max_metadata_default_block_size_128():
    """With default B=128 and term lengths < 128, each term has exactly 1
    block whose max equals the term's overall max.
    """
    # 5 terms, each with 3 postings
    data = np.array(
        [1.0, 2.0, 3.0,  10.0, 5.0, 2.0,  0.5, 0.1, 0.2,
         100.0, 99.0, 98.0,  0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    indices = np.zeros(15, dtype=np.int32)
    indptr = np.array([0, 3, 6, 9, 12, 15], dtype=np.int32)

    bmax, boff = mojo_bm25s.build_block_max_metadata(
        data, indices, indptr  # default block_size=128
    )
    assert boff.tolist() == [0, 1, 2, 3, 4, 5]
    np.testing.assert_array_equal(
        bmax, np.array([3.0, 10.0, 0.5, 100.0, 0.0], dtype=np.float32)
    )


def test_block_max_metadata_empty_term():
    """A term with no postings has zero blocks."""
    data = np.array([5.0, 3.0], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    indptr = np.array([0, 0, 2, 2], dtype=np.int32)  # term 0 and 2 empty

    bmax, boff = mojo_bm25s.build_block_max_metadata(
        data, indices, indptr, block_size=4
    )
    # term 0: 0 blocks, term 1: 1 block, term 2: 0 blocks.
    assert boff.tolist() == [0, 0, 1, 1]
    np.testing.assert_array_equal(bmax, np.array([5.0], dtype=np.float32))


def test_block_max_metadata_dtype_and_shape():
    """The returned arrays have well-defined dtypes."""
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.int32)
    indptr = np.array([0, 3], dtype=np.int32)

    bmax, boff = mojo_bm25s.build_block_max_metadata(data, indices, indptr)
    assert bmax.dtype == np.float32
    assert boff.dtype == np.int32
    assert boff.shape == (2,)  # n_vocab + 1
    assert bmax.shape == (1,)  # 1 block for 3 postings under B=128


def test_block_max_metadata_partial_last_block():
    """B does not divide nnz_t evenly — last block is short."""
    # B=3, 7 postings -> blocks of [3, 3, 1].
    data = np.array([1.0, 2.0, 5.0,   4.0, 0.5, 0.3,   9.0], dtype=np.float32)
    indices = np.arange(7, dtype=np.int32)
    indptr = np.array([0, 7], dtype=np.int32)

    bmax, boff = mojo_bm25s.build_block_max_metadata(
        data, indices, indptr, block_size=3
    )
    assert boff.tolist() == [0, 3]
    np.testing.assert_array_equal(bmax, np.array([5.0, 4.0, 9.0], dtype=np.float32))


# ----------------------------------------------------------------------
# Persistence round-trip
# ----------------------------------------------------------------------

def _build_small_retriever(method="lucene"):
    corpus = [
        ["the", "quick", "brown", "fox"],
        ["the", "lazy", "dog", "sleeps"],
        ["a", "quick", "fox", "jumps", "over", "the", "lazy", "dog"],
        ["brown", "fox", "fox", "fox"],
    ]
    r = bm25s.BM25(method=method)
    r.index(corpus, show_progress=False)
    return r, corpus


def test_block_max_metadata_roundtrip_persistence(tmp_path):
    """save_index → load_index preserves block_max_impacts and
    block_offsets bit-exact when passed."""
    from mojo_bm25s import build_index, save_index, load_index, Vocab

    corpus = [
        ["the", "quick", "brown", "fox"],
        ["the", "lazy", "dog", "sleeps"],
        ["a", "quick", "fox", "jumps", "over", "the", "lazy", "dog"],
        ["brown", "fox", "fox", "fox"],
    ]
    vocab = Vocab.from_corpus(corpus)
    ids = [vocab.tokens_to_ids(doc) for doc in corpus]
    data, indices, indptr, n_docs, l_avg, nonocc = build_index(
        ids, n_vocab=len(vocab), method="lucene"
    )
    bmax, boff = mojo_bm25s.build_block_max_metadata(
        data, indices, indptr, block_size=128
    )

    out_dir = tmp_path / "idx"
    save_index(
        out_dir,
        data=data, indices=indices, indptr=indptr,
        n_docs=n_docs, l_avg=l_avg, vocab=vocab,
        k1=1.5, b=0.75, delta=0.5,
        method="lucene", idf_method="lucene",
        nonoccurrence=nonocc,
        block_max_impacts=bmax,
        block_offsets=boff,
        block_size=128,
    )
    loaded = load_index(out_dir)
    assert loaded.block_max_impacts is not None
    assert loaded.block_offsets is not None
    assert loaded.block_size == 128
    np.testing.assert_array_equal(loaded.block_max_impacts, bmax)
    np.testing.assert_array_equal(loaded.block_offsets, boff)


def test_load_legacy_index_without_block_max(tmp_path):
    """An index saved without BMW metadata loads cleanly, with
    block_max_impacts == None."""
    from mojo_bm25s import build_index, save_index, load_index, Vocab

    corpus = [["a", "b"], ["b", "c"]]
    vocab = Vocab.from_corpus(corpus)
    ids = [vocab.tokens_to_ids(doc) for doc in corpus]
    data, indices, indptr, n_docs, l_avg, nonocc = build_index(
        ids, n_vocab=len(vocab), method="lucene"
    )
    out_dir = tmp_path / "legacy"
    save_index(
        out_dir,
        data=data, indices=indices, indptr=indptr,
        n_docs=n_docs, l_avg=l_avg, vocab=vocab,
        k1=1.5, b=0.75, delta=0.5,
        method="lucene", idf_method="lucene",
        nonoccurrence=nonocc,
    )
    loaded = load_index(out_dir)
    assert loaded.block_max_impacts is None
    assert loaded.block_offsets is None
    assert loaded.block_size is None


# ----------------------------------------------------------------------
# retrieve_batch_bmw — parity vs scan-everything (small corpus)
# ----------------------------------------------------------------------

VOCAB = [
    "cat", "dog", "fish", "bird", "horse",
    "fast", "slow", "loud", "quiet", "small",
    "river", "ocean", "mountain", "forest", "city",
    "the", "and", "of", "to", "with",
]


def _make_corpus(n: int = 50, max_len: int = 12, seed: int = 0) -> list[list[str]]:
    rng = np.random.default_rng(seed)
    return [
        list(rng.choice(VOCAB, size=int(rng.integers(4, max_len)), replace=True))
        for _ in range(n)
    ]


def _make_queries(n: int = 12, max_len: int = 4, seed: int = 1) -> list[list[str]]:
    rng = np.random.default_rng(seed)
    return [
        list(rng.choice(
            VOCAB, size=int(rng.integers(1, max_len)), replace=False
        ))
        for _ in range(n)
    ]


@pytest.fixture(scope="module")
def small_indexed():
    corpus = _make_corpus()
    r = bm25s.BM25()
    r.index(corpus, show_progress=False)
    queries = _make_queries()
    return r, queries


def _scores_equiv(got: np.ndarray, want: np.ndarray, atol: float = 1e-5):
    """Compare top-k score arrays — sort each row descending, then
    allclose. Tolerates id tie-swap at the rank-k boundary."""
    g = np.sort(got, axis=1)[:, ::-1]
    w = np.sort(want, axis=1)[:, ::-1]
    np.testing.assert_allclose(g, w, atol=atol)


@pytest.mark.parametrize("k", [1, 5, 10])
def test_bmw_parity_small_corpus(small_indexed, k):
    r, queries = small_indexed
    ref_scores, ref_ids = mojo_bm25s.retrieve_batch(r, queries, k=k)
    bmw_scores, bmw_ids = mojo_bm25s.retrieve_batch_bmw(r, queries, k=k)
    assert bmw_scores.shape == ref_scores.shape
    assert bmw_ids.shape == ref_ids.shape
    _scores_equiv(bmw_scores, ref_scores)


@pytest.mark.parametrize("method", ["lucene", "atire", "bm25l", "bm25+"])
def test_bmw_parity_methods(method):
    corpus = _make_corpus(n=80, seed=7)
    r = bm25s.BM25(method=method)
    r.index(corpus, show_progress=False)
    queries = _make_queries(n=10, max_len=4, seed=9)

    ref_scores, _ = mojo_bm25s.retrieve_batch(r, queries, k=10)
    bmw_scores, _ = mojo_bm25s.retrieve_batch_bmw(r, queries, k=10)
    _scores_equiv(bmw_scores, ref_scores, atol=1e-5)


def test_bmw_parity_against_bm25s_endtoend():
    """BMW path agrees with bm25s.BM25.retrieve on a tiny fixture."""
    corpus = [
        ["cat", "dog", "fish"],
        ["cat", "bird"],
        ["dog", "bird", "horse"],
        ["fish", "horse", "horse"],
        ["the", "cat", "and", "the", "dog"],
    ]
    r = bm25s.BM25()
    r.index(corpus, show_progress=False)
    queries = [["cat", "dog"], ["bird"], ["horse", "the"]]
    k = 3
    bmw_scores, bmw_ids = mojo_bm25s.retrieve_batch_bmw(r, queries, k=k)
    # bm25s native retrieve
    for i, q in enumerate(queries):
        ref_ids, ref_scores = r.retrieve([q], k=k, show_progress=False)
        np.testing.assert_allclose(
            np.sort(bmw_scores[i])[::-1],
            np.sort(ref_scores[0])[::-1],
            atol=1e-5,
            err_msg=f"query={q}",
        )


# ----------------------------------------------------------------------
# Medium-corpus parity — query lengths {1, 2, 5, 20}
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def medium_indexed():
    vocab_size = 1000
    vocab_strs = [f"tok{i}" for i in range(vocab_size)]
    rng = np.random.default_rng(0)
    corpus = []
    for _ in range(5000):
        doc_len = int(rng.integers(20, 100))
        doc = [vocab_strs[int(rng.integers(0, vocab_size))] for _ in range(doc_len)]
        corpus.append(doc)
    r = bm25s.BM25()
    r.index(corpus, show_progress=False)
    return r, vocab_strs


@pytest.mark.parametrize("query_len", [1, 2, 5, 20])
@pytest.mark.parametrize("k", [1, 10, 100])
def test_bmw_parity_medium(medium_indexed, query_len, k):
    r, vocab_strs = medium_indexed
    rng = np.random.default_rng(query_len * 100 + k)
    queries = []
    for _ in range(8):
        q = [vocab_strs[int(rng.integers(0, len(vocab_strs)))]
             for _ in range(query_len)]
        queries.append(q)

    ref_scores, _ = mojo_bm25s.retrieve_batch(r, queries, k=k)
    bmw_scores, _ = mojo_bm25s.retrieve_batch_bmw(r, queries, k=k)
    assert bmw_scores.shape == ref_scores.shape
    _scores_equiv(bmw_scores, ref_scores)


# ----------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------

def test_bmw_empty_query(small_indexed):
    r, _ = small_indexed
    scores, ids = mojo_bm25s.retrieve_batch_bmw(r, [[]], k=5)
    assert scores.shape == (1, 5)
    np.testing.assert_allclose(scores[0], 0.0)


def test_bmw_single_token_query(small_indexed):
    """Degenerate WAND — just scans the single posting list."""
    r, _ = small_indexed
    queries = [["cat"], ["dog"], ["fish"]]
    ref_scores, _ = mojo_bm25s.retrieve_batch(r, queries, k=5)
    bmw_scores, _ = mojo_bm25s.retrieve_batch_bmw(r, queries, k=5)
    _scores_equiv(bmw_scores, ref_scores)


def test_bmw_no_overlap_with_corpus(small_indexed):
    """Query terms not in vocab → all zero scores. Pre-validation should
    raise — but a query for tokens with empty postings should yield zeros
    without crashing."""
    r, _ = small_indexed
    # Use valid tokens but in a query whose terms don't all show together
    # — we just verify BMW doesn't crash and returns sensible zero pad.
    # Construct a query with a single token that's rare.
    queries = [["mountain"]]
    ref_scores, _ = mojo_bm25s.retrieve_batch(r, queries, k=5)
    bmw_scores, _ = mojo_bm25s.retrieve_batch_bmw(r, queries, k=5)
    _scores_equiv(bmw_scores, ref_scores)


def test_bmw_k_larger_than_corpus():
    corpus = [["a", "b"], ["a", "c"]]
    r = bm25s.BM25()
    r.index(corpus, show_progress=False)
    scores, ids = mojo_bm25s.retrieve_batch_bmw(r, [["a"]], k=5)
    assert scores.shape == (1, 5)
    assert np.all(scores[0, 2:] == 0.0)


def test_bmw_block_size_not_dividing_nnz():
    """A block size that doesn't divide the longest term's posting list
    exercises the partial-last-block path in retrieval."""
    corpus = _make_corpus(n=200, max_len=8, seed=42)
    r = bm25s.BM25()
    r.index(corpus, show_progress=False)
    queries = _make_queries(n=5, max_len=4, seed=43)

    # Use a tiny block size to force many partial blocks
    ref_scores, _ = mojo_bm25s.retrieve_batch(r, queries, k=10)
    bmw_scores, _ = mojo_bm25s.retrieve_batch_bmw(
        r, queries, k=10, block_size=7
    )
    _scores_equiv(bmw_scores, ref_scores)


def test_bmw_returns_well_typed_arrays(small_indexed):
    r, queries = small_indexed
    scores, ids = mojo_bm25s.retrieve_batch_bmw(r, queries, k=10)
    assert scores.dtype == np.float32
    assert ids.dtype == np.int32


def test_bmw_scores_sorted_descending_per_row(small_indexed):
    r, queries = small_indexed
    scores, _ = mojo_bm25s.retrieve_batch_bmw(r, queries, k=10)
    diffs = np.diff(scores, axis=1)
    assert (diffs <= 1e-7).all()


def test_bmw_rejects_invalid_k(small_indexed):
    r, queries = small_indexed
    with pytest.raises(ValueError):
        mojo_bm25s.retrieve_batch_bmw(r, queries, k=0)
    with pytest.raises(ValueError):
        mojo_bm25s.retrieve_batch_bmw(r, queries, k=-1)
