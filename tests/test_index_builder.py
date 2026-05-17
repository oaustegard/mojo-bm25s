"""Tests for ``mojo_bm25s.build_index`` (issue #25).

Parity oracle: ``bm25s.BM25(method=..., idf_method=..., k1=..., b=...,
delta=...).index(corpus_tokens=...)``. After indexing, the reference's
``retriever.scores`` dict has ``data, indices, indptr, num_docs`` and
``retriever.nonoccurrence_array`` is set for bm25l / bm25+.

Our builder takes pre-tokenized **integer token IDs** plus an explicit
``n_vocab`` (so it stays decoupled from vocab construction — that's
issue #24). To get parity, both sides must use the **same** id mapping,
so the test fixtures build the vocab themselves and pass it to both
oracles.

Tests progress: API surface → small synthetic shape parity → numerical
parity across the 5-method matrix → nonoccurrence parity for bm25l /
bm25+ → end-to-end scifact retrieve parity → edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest

import bm25s


# ----------------------------------------------------------------------
# Fixture builders
# ----------------------------------------------------------------------

def _build_vocab_and_ids(corpus_tokens):
    """Walk the corpus in document/left-to-right order and assign
    sequential int32 IDs to first-seen tokens. Returns ``(vocab_dict,
    corpus_token_ids)``. Deterministic (no set iteration), so we can
    feed the same id mapping to both bm25s and our builder.
    """
    vocab_dict: dict[str, int] = {}
    corpus_token_ids: list[np.ndarray] = []
    for doc in corpus_tokens:
        ids = np.empty(len(doc), dtype=np.int32)
        for i, tok in enumerate(doc):
            if tok not in vocab_dict:
                vocab_dict[tok] = len(vocab_dict)
            ids[i] = vocab_dict[tok]
        corpus_token_ids.append(ids)
    return vocab_dict, corpus_token_ids


def _ref_build(
    corpus_token_ids,
    vocab_dict,
    *,
    method="lucene",
    idf_method=None,
    k1=1.5,
    b=0.75,
    delta=0.5,
):
    """Build the bm25s reference index from the same (vocab, ids) we
    pass our builder. Returns ``(retriever, scores_dict, nonoccurrence)``
    where ``scores_dict`` is what bm25s stores on ``retriever.scores``.
    """
    retriever = bm25s.BM25(
        k1=k1, b=b, delta=delta, method=method, idf_method=idf_method,
        csc_backend="numpy", backend="numpy",
    )
    # The (token_ids, vocab_dict) tuple path in bm25s.index avoids the
    # set-iteration nondeterminism in get_unique_tokens; the vocab_dict
    # we pass in is the authoritative id-to-position map.
    # Convert ids to python lists since bm25s indexes them with python
    # iteration semantics.
    corpus_ids_lists = [ids.tolist() for ids in corpus_token_ids]
    retriever.index((corpus_ids_lists, dict(vocab_dict)),
                    create_empty_token=False, show_progress=False)
    return retriever


# ----------------------------------------------------------------------
# API surface
# ----------------------------------------------------------------------

def test_build_index_is_exported():
    import mojo_bm25s

    assert hasattr(mojo_bm25s, "build_index"), (
        "mojo_bm25s.build_index should be exported per issue #25 API"
    )
    assert callable(mojo_bm25s.build_index)


def test_build_index_return_tuple_shape():
    """Return shape: ``(data, indices, indptr, n_docs, l_avg, nonoccurrence)``.

    nonoccurrence is None for methods that don't need it.
    """
    from mojo_bm25s import build_index

    corpus_tokens = [["a", "b"], ["b", "c"]]
    vocab_dict, ids = _build_vocab_and_ids(corpus_tokens)

    result = build_index(ids, n_vocab=len(vocab_dict), method="lucene")
    assert isinstance(result, tuple)
    assert len(result) == 6
    data, indices, indptr, n_docs, l_avg, nonoccurrence = result
    assert data.dtype == np.float32
    assert indices.dtype == np.int32
    assert indptr.dtype == np.int32
    assert indptr.shape == (len(vocab_dict) + 1,)
    assert n_docs == 2
    assert isinstance(l_avg, float)
    # lucene doesn't need nonoccurrence
    assert nonoccurrence is None


# ----------------------------------------------------------------------
# Shape parity on a small synthetic corpus
# ----------------------------------------------------------------------

SMALL_CORPUS = [
    ["the", "quick", "brown", "fox"],
    ["the", "lazy", "dog", "sleeps"],
    ["a", "quick", "fox", "jumps", "over", "the", "lazy", "dog"],
    ["brown", "fox", "fox", "fox"],  # repeated token within doc (high tf)
]


def test_shape_parity_small_lucene():
    from mojo_bm25s import build_index

    vocab_dict, ids = _build_vocab_and_ids(SMALL_CORPUS)
    ref = _ref_build(ids, vocab_dict, method="lucene")

    data, indices, indptr, n_docs, l_avg, _ = build_index(
        ids, n_vocab=len(vocab_dict), method="lucene",
    )
    ref_data = np.asarray(ref.scores["data"])
    ref_indices = np.asarray(ref.scores["indices"])
    ref_indptr = np.asarray(ref.scores["indptr"])
    ref_n_docs = int(ref.scores["num_docs"])

    assert data.shape == ref_data.shape
    assert indices.shape == ref_indices.shape
    assert indptr.shape == ref_indptr.shape
    assert n_docs == ref_n_docs


# ----------------------------------------------------------------------
# Numerical parity per (method) — full matrix
# ----------------------------------------------------------------------

@pytest.mark.parametrize("method", ["lucene", "robertson", "atire", "bm25l", "bm25+"])
def test_numerical_parity_small_corpus(method):
    from mojo_bm25s import build_index

    vocab_dict, ids = _build_vocab_and_ids(SMALL_CORPUS)
    ref = _ref_build(ids, vocab_dict, method=method)

    data, indices, indptr, n_docs, l_avg, nonoccurrence = build_index(
        ids, n_vocab=len(vocab_dict), method=method,
    )

    ref_data = np.asarray(ref.scores["data"], dtype=np.float32)
    ref_indices = np.asarray(ref.scores["indices"], dtype=np.int32)
    ref_indptr = np.asarray(ref.scores["indptr"], dtype=np.int32)

    # indptr / indices: exact integer match (CSC ordering is part of the
    # contract — within each column the doc ids are monotone ascending,
    # bm25s relies on this in _np_csc_jit_ready)
    np.testing.assert_array_equal(indptr, ref_indptr,
                                  err_msg=f"{method}: indptr mismatch")
    np.testing.assert_array_equal(indices, ref_indices,
                                  err_msg=f"{method}: indices mismatch")
    # data: float32 tolerance
    np.testing.assert_allclose(
        data, ref_data, atol=1e-5,
        err_msg=f"{method}: data mismatch (max |delta|="
                f"{np.max(np.abs(data - ref_data)) if data.size else 0:.3e})",
    )


# ----------------------------------------------------------------------
# Nonoccurrence parity (bm25l, bm25+)
# ----------------------------------------------------------------------

@pytest.mark.parametrize("method", ["bm25l", "bm25+"])
def test_nonoccurrence_parity_small(method):
    from mojo_bm25s import build_index

    vocab_dict, ids = _build_vocab_and_ids(SMALL_CORPUS)
    ref = _ref_build(ids, vocab_dict, method=method)
    assert ref.nonoccurrence_array is not None, (
        f"bm25s should produce a nonoccurrence_array for {method}"
    )

    _, _, _, _, _, nonoccurrence = build_index(
        ids, n_vocab=len(vocab_dict), method=method,
    )
    assert nonoccurrence is not None
    assert nonoccurrence.dtype == np.float32
    assert nonoccurrence.shape == (len(vocab_dict),)

    ref_non = np.asarray(ref.nonoccurrence_array, dtype=np.float32)
    np.testing.assert_allclose(
        nonoccurrence, ref_non, atol=1e-5,
        err_msg=f"{method}: nonoccurrence_array mismatch",
    )


@pytest.mark.parametrize("method", ["lucene", "robertson", "atire"])
def test_nonoccurrence_is_none_for_non_required_methods(method):
    from mojo_bm25s import build_index

    vocab_dict, ids = _build_vocab_and_ids(SMALL_CORPUS)
    _, _, _, _, _, nonoccurrence = build_index(
        ids, n_vocab=len(vocab_dict), method=method,
    )
    assert nonoccurrence is None, (
        f"{method} should not have a nonoccurrence_array (matches bm25s)"
    )


# ----------------------------------------------------------------------
# idf_method separate from tfc method
# ----------------------------------------------------------------------

def test_idf_method_independent_of_tfc_method():
    """Per BM25 API, ``idf_method`` can differ from ``method``."""
    from mojo_bm25s import build_index

    vocab_dict, ids = _build_vocab_and_ids(SMALL_CORPUS)
    ref = _ref_build(ids, vocab_dict, method="lucene", idf_method="atire")

    data, indices, indptr, _, _, _ = build_index(
        ids, n_vocab=len(vocab_dict), method="lucene", idf_method="atire",
    )
    np.testing.assert_array_equal(indptr, np.asarray(ref.scores["indptr"]))
    np.testing.assert_array_equal(indices, np.asarray(ref.scores["indices"]))
    np.testing.assert_allclose(
        data, np.asarray(ref.scores["data"], dtype=np.float32), atol=1e-5,
    )


# ----------------------------------------------------------------------
# n_docs / l_avg parity
# ----------------------------------------------------------------------

def test_n_docs_and_l_avg_match():
    from mojo_bm25s import build_index

    vocab_dict, ids = _build_vocab_and_ids(SMALL_CORPUS)
    _, _, _, n_docs, l_avg, _ = build_index(
        ids, n_vocab=len(vocab_dict), method="lucene",
    )

    ref_avg = float(np.array([len(doc) for doc in ids]).mean())
    assert n_docs == len(ids)
    assert abs(l_avg - ref_avg) < 1e-6


# ----------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------

def test_empty_corpus():
    from mojo_bm25s import build_index

    # No docs, no tokens. nvocab=0 still legal.
    data, indices, indptr, n_docs, l_avg, _ = build_index(
        [], n_vocab=0, method="lucene",
    )
    assert data.shape == (0,)
    assert indices.shape == (0,)
    assert indptr.shape == (1,)
    assert int(indptr[0]) == 0
    assert n_docs == 0
    # l_avg of an empty corpus is 0.0 by convention (no docs to average over)
    assert l_avg == 0.0


def test_empty_corpus_with_predeclared_vocab():
    """Vocab exists but no docs ever observed — every df = 0."""
    from mojo_bm25s import build_index

    data, indices, indptr, n_docs, l_avg, _ = build_index(
        [], n_vocab=3, method="lucene",
    )
    assert data.shape == (0,)
    assert indices.shape == (0,)
    # indptr length is n_vocab + 1; all zeros (no entries in any column)
    np.testing.assert_array_equal(indptr, np.zeros(4, dtype=np.int32))
    assert n_docs == 0


def test_single_doc_corpus():
    from mojo_bm25s import build_index

    corpus_tokens = [["alpha", "beta", "gamma"]]
    vocab_dict, ids = _build_vocab_and_ids(corpus_tokens)
    ref = _ref_build(ids, vocab_dict, method="lucene")

    data, indices, indptr, n_docs, l_avg, _ = build_index(
        ids, n_vocab=len(vocab_dict), method="lucene",
    )
    np.testing.assert_array_equal(indptr, np.asarray(ref.scores["indptr"]))
    np.testing.assert_array_equal(indices, np.asarray(ref.scores["indices"]))
    np.testing.assert_allclose(
        data, np.asarray(ref.scores["data"], dtype=np.float32), atol=1e-5,
    )
    assert n_docs == 1


def test_high_tf_within_single_doc():
    """Single token repeated many times within a doc."""
    from mojo_bm25s import build_index

    corpus_tokens = [["x"] * 10, ["y", "x"]]
    vocab_dict, ids = _build_vocab_and_ids(corpus_tokens)
    ref = _ref_build(ids, vocab_dict, method="lucene")

    data, indices, indptr, n_docs, l_avg, _ = build_index(
        ids, n_vocab=len(vocab_dict), method="lucene",
    )
    np.testing.assert_array_equal(indices, np.asarray(ref.scores["indices"]))
    np.testing.assert_allclose(
        data, np.asarray(ref.scores["data"], dtype=np.float32), atol=1e-5,
    )


def test_df_zero_token_in_vocab():
    """Vocab includes a token never seen at index time.

    bm25s zeroes out IDF for df=0 (see ``_build_idf_array`` — only sets
    idf_array[token_id] if df != 0). The column should be empty (no
    entries in ``data`` / ``indices`` for that token), and indptr should
    show zero width for that column.
    """
    from mojo_bm25s import build_index

    corpus_tokens = [["a", "b"], ["b", "c"]]
    vocab_dict, ids = _build_vocab_and_ids(corpus_tokens)
    # Manually inject an unseen token at the next id slot — our builder
    # should tolerate n_vocab > #observed tokens.
    n_vocab = len(vocab_dict) + 1  # one extra ghost token

    data, indices, indptr, n_docs, l_avg, _ = build_index(
        ids, n_vocab=n_vocab, method="lucene",
    )
    assert indptr.shape == (n_vocab + 1,)
    # The last column (the ghost token) has zero width — no data entries.
    assert int(indptr[-1]) == int(indptr[-2])


# ----------------------------------------------------------------------
# End-to-end scifact retrieve parity
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def scifact_tokens():
    """Tokenize scifact once for the module."""
    import sys
    from pathlib import Path

    _REPO_ROOT = Path(__file__).resolve().parent.parent
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    from benchmarks.datasets import load_beir

    ds = load_beir("scifact")
    corpus_tokens = ds.corpus_tokens()
    query_tokens = ds.query_tokens()[:10]
    return corpus_tokens, query_tokens


def test_end_to_end_scifact_retrieve_parity(scifact_tokens):
    """Build with our index builder, retrieve via bm25s.BM25.retrieve.

    We swap our (data, indices, indptr, num_docs) into a bm25s retriever
    instance and run its native retrieve path — the score arrays should
    match a fully bm25s-built retriever within float32 tolerance.
    """
    from mojo_bm25s import build_index

    corpus_tokens, query_tokens = scifact_tokens

    vocab_dict, ids = _build_vocab_and_ids(corpus_tokens)

    # Reference: pure bm25s on the same (ids, vocab)
    ref = _ref_build(ids, vocab_dict, method="lucene")

    # Ours: build our CSC, then assemble a bm25s.BM25 around it.
    data, indices, indptr, n_docs, l_avg, _ = build_index(
        ids, n_vocab=len(vocab_dict), method="lucene",
    )

    ours = bm25s.BM25(method="lucene", csc_backend="numpy", backend="numpy")
    ours.scores = {
        "data": data,
        "indices": indices,
        "indptr": indptr,
        "num_docs": n_docs,
    }
    ours.vocab_dict = dict(vocab_dict)
    ours.unique_token_ids_set = set(vocab_dict.values())
    ours.nonoccurrence_array = None

    # Now run retrieve via bm25s's own path on both
    ref_results = ref.retrieve(
        query_tokens, k=10, show_progress=False, n_threads=0,
    )
    our_results = ours.retrieve(
        query_tokens, k=10, show_progress=False, n_threads=0,
    )

    # Scores match within tolerance
    np.testing.assert_allclose(
        our_results.scores, ref_results.scores, atol=1e-5,
        err_msg="scifact retrieve scores diverge between mojo-built and "
                "bm25s-built indices",
    )
    # Doc IDs at top-k must match in rank-k tie class — exact match is
    # the expected outcome when scores are bitwise-equal modulo float32
    # accumulation noise; if there's a true tie at the boundary the
    # selection backend's tiebreak is deterministic so this should still
    # be exact-equal.
    np.testing.assert_array_equal(
        our_results.documents, ref_results.documents,
        err_msg="scifact retrieve top-10 doc ids diverge",
    )


@pytest.mark.parametrize("method", ["lucene", "bm25l", "bm25+"])
def test_end_to_end_scifact_retrieve_parity_per_method(scifact_tokens, method):
    """Same as above but across a couple methods including those with
    nonoccurrence_array (bm25l, bm25+).
    """
    from mojo_bm25s import build_index

    corpus_tokens, query_tokens = scifact_tokens

    vocab_dict, ids = _build_vocab_and_ids(corpus_tokens)

    ref = _ref_build(ids, vocab_dict, method=method)

    data, indices, indptr, n_docs, l_avg, nonoccurrence = build_index(
        ids, n_vocab=len(vocab_dict), method=method,
    )

    ours = bm25s.BM25(method=method, csc_backend="numpy", backend="numpy")
    ours.scores = {
        "data": data,
        "indices": indices,
        "indptr": indptr,
        "num_docs": n_docs,
    }
    ours.vocab_dict = dict(vocab_dict)
    ours.unique_token_ids_set = set(vocab_dict.values())
    ours.nonoccurrence_array = nonoccurrence

    ref_results = ref.retrieve(
        query_tokens, k=10, show_progress=False, n_threads=0,
    )
    our_results = ours.retrieve(
        query_tokens, k=10, show_progress=False, n_threads=0,
    )

    np.testing.assert_allclose(
        our_results.scores, ref_results.scores, atol=1e-5,
        err_msg=f"{method}: scifact retrieve scores diverge",
    )
