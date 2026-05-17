"""Integration tests for the bm25s monkey-patch (issue #6).

Goal: a stock `bm25s.BM25` retriever, patched via `mojo_bm25s.patch_bm25s`,
produces the same top-k results as the unpatched one — modulo
score-tie reordering at the rank-k boundary.
"""

from __future__ import annotations

import numpy as np
import pytest

import bm25s

import mojo_bm25s


# ---------------------------------------------------------------------------
# Corpus + query fixtures. Use plain lists of strings rather than
# `bm25s.tokenize` so the test doesn't depend on NLTK/PyStemmer assets.
# ---------------------------------------------------------------------------

VOCAB = [
    "cat", "dog", "fish", "bird", "horse",
    "fast", "slow", "loud", "quiet", "small",
    "river", "ocean", "mountain", "forest", "city",
    "the", "and", "of", "to", "with",
]


def _make_corpus(n_docs: int = 100, seed: int = 0) -> list[list[str]]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_docs):
        length = int(rng.integers(4, 12))
        doc = list(rng.choice(VOCAB, size=length, replace=True))
        out.append(doc)
    return out


def _make_queries(n_queries: int = 10, seed: int = 1) -> list[list[str]]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_queries):
        length = int(rng.integers(1, 4))
        q = list(rng.choice(VOCAB, size=length, replace=False))
        out.append(q)
    return out


@pytest.fixture(scope="module")
def corpus_and_queries():
    return _make_corpus(100), _make_queries(10)


def _build(corpus: list[list[str]]) -> bm25s.BM25:
    r = bm25s.BM25()
    r.index(corpus)
    return r


# ---------------------------------------------------------------------------
# patch_bm25s wires up the kernels.
# ---------------------------------------------------------------------------

def test_patch_returns_same_retriever(corpus_and_queries):
    corpus, _ = corpus_and_queries
    r = _build(corpus)
    rv = mojo_bm25s.patch_bm25s(r)
    assert rv is r  # patch_bm25s returns the input for chaining


def test_patch_rejects_non_float32_index(corpus_and_queries):
    """A retriever built with dtype=float64 should refuse the patch
    early, not silently lose precision at retrieve time."""
    corpus, _ = corpus_and_queries
    r = bm25s.BM25(dtype="float64")
    r.index(corpus)
    with pytest.raises((ValueError, TypeError)) as exc_info:
        mojo_bm25s.patch_bm25s(r)
    # Error message should mention dtype so the user knows why.
    assert "dtype" in str(exc_info.value).lower() or "float32" in str(exc_info.value).lower()


def test_patch_replaces_hot_path_methods(corpus_and_queries):
    """After patching, the two hot-path methods on the instance must not
    be the class defaults (i.e. the patch took effect)."""
    corpus, _ = corpus_and_queries
    r = _build(corpus)
    cls_compute = bm25s.BM25._compute_relevance_from_scores
    cls_topk = bm25s.BM25._get_top_k_results

    mojo_bm25s.patch_bm25s(r)

    inst_compute = r.__dict__.get("_compute_relevance_from_scores")
    inst_topk = r.__dict__.get("_get_top_k_results")
    assert inst_compute is not None and inst_compute is not cls_compute
    assert inst_topk is not None and inst_topk is not cls_topk


# ---------------------------------------------------------------------------
# Parity: patched ↔ unpatched on the same corpus + queries.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def patched_and_unpatched(corpus_and_queries):
    corpus, _ = corpus_and_queries
    a = _build(corpus)
    b = _build(corpus)
    mojo_bm25s.patch_bm25s(b)
    return a, b


@pytest.mark.parametrize("k", [1, 5, 10])
def test_top_k_scores_match(patched_and_unpatched, corpus_and_queries, k):
    a, b = patched_and_unpatched
    _, queries = corpus_and_queries
    for q in queries:
        ids_ref, sc_ref = a.retrieve([q], k=k, show_progress=False)
        ids_got, sc_got = b.retrieve([q], k=k, show_progress=False)
        # Top-k scores must match within float32 tolerance regardless of
        # which doc IDs they were assigned to (ties are allowed).
        np.testing.assert_allclose(
            np.sort(sc_got[0])[::-1], np.sort(sc_ref[0])[::-1], atol=1e-5
        )


@pytest.mark.parametrize("k", [1, 5, 10])
def test_top_k_doc_ids_in_tie_class(patched_and_unpatched, corpus_and_queries, k):
    """Every doc ID returned by the patched backend must have a score
    at least as high as the rank-k boundary from the unpatched backend.

    This is the "modulo score-tie reordering" formulation from the
    issue acceptance: when several docs share the boundary score,
    either backend may pick any of them.
    """
    a, b = patched_and_unpatched
    _, queries = corpus_and_queries
    for q in queries:
        full_scores = a.get_scores(q)  # unpatched view, full doc-score vector
        _, sc_ref = a.retrieve([q], k=k, show_progress=False)
        ids_got, _ = b.retrieve([q], k=k, show_progress=False)
        boundary = float(sc_ref[0, -1])
        for picked_id in ids_got[0].tolist():
            picked_score = float(full_scores[picked_id])
            assert picked_score + 1e-5 >= boundary, (
                f"q={q!r} k={k}: patched picked id={picked_id} with "
                f"score {picked_score:.6f}, below rank-k boundary "
                f"{boundary:.6f}"
            )


def test_quickstart_readme_pattern():
    """The exact shape of the bm25s README quickstart, with patch_bm25s
    inserted between index() and retrieve(). Must run end-to-end and
    return top-2 results with a non-degenerate score."""
    corpus = [
        ["cat", "feline", "purr"],
        ["dog", "human", "friend", "play"],
        ["bird", "animal", "fly"],
        ["fish", "water", "swim"],
    ]
    retriever = bm25s.BM25()
    retriever.index(corpus)
    mojo_bm25s.patch_bm25s(retriever)

    query = [["fish", "purr", "cat"]]
    ids, scores = retriever.retrieve(query, k=2, show_progress=False)
    assert ids.shape == (1, 2)
    assert scores.shape == (1, 2)
    assert scores.dtype == np.float32
    # The "cat" doc (index 0) and the "fish" doc (index 3) should beat
    # the other two; their scores should be strictly positive.
    assert (scores[0] > 0).all()
    assert set(ids[0].tolist()) == {0, 3}
