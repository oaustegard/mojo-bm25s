"""Tests for the standalone ``mojo_bm25s.Retriever`` class (issue #27).

Contract (locked here):

- **Export surface**: ``from mojo_bm25s import Retriever`` works; class is
  re-exported at the package level alongside the existing free functions.
- **API**: matches the issue spec exactly — ``__init__(k1, b, delta,
  method, idf_method, stopwords, stemmer)``, ``index(corpus) -> self``,
  ``retrieve(queries, k) -> (scores, ids)``, ``save(path)``, ``load(path)``.
- **Wiring**: ``index`` composes ``tokenize → vocab → build_index``;
  ``retrieve`` mirrors the tokenize/stem pipeline on the query side and
  routes through the same Mojo kernel ``retrieve_batch`` uses.
- **Parity**: end-to-end on BEIR scifact, top-10 scores agree with the
  ``patch_bm25s`` path within ``atol=1e-5``; IDs lie in the rank-k tie
  class. This is the "Phase 2 standalone is correct" headline.
- **Save/load**: round-trip produces identical retrieve output.
- **Stemmer integration**: a user-supplied callable is applied to both
  corpus AND query tokens — otherwise queries miss everything.
- **Edge cases**: empty corpus → clear error; OOV-only query → zeros;
  retrieve-before-index → clear error; index called twice → replace not
  append.

Tests must fail against a stub that imports cleanly but returns dummies —
all assertions are on real values (scores, ids, parity).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import bm25s

import mojo_bm25s


# ----------------------------------------------------------------------
# Small in-tree corpora — for fast tests that don't need scifact.
# ----------------------------------------------------------------------

SMALL_CORPUS = [
    "a cat is small",
    "a dog is big",
    "fish swim in the sea",
    "the lazy dog sleeps",
    "the quick brown fox jumps over the lazy dog",
]


# ----------------------------------------------------------------------
# Surface
# ----------------------------------------------------------------------

def test_retriever_is_exported():
    assert hasattr(mojo_bm25s, "Retriever")
    from mojo_bm25s import Retriever  # noqa: F401
    assert callable(mojo_bm25s.Retriever)


def test_default_construction_does_not_raise():
    r = mojo_bm25s.Retriever()
    # We don't lock the default values' types (issue #27 documents them);
    # we just lock that defaults work and produce a usable instance.
    assert r is not None


# ----------------------------------------------------------------------
# Basic flow + method chaining
# ----------------------------------------------------------------------

def test_index_returns_self_for_chaining():
    r = mojo_bm25s.Retriever()
    out = r.index(SMALL_CORPUS)
    assert out is r, "Retriever.index() must return self for chaining"


def test_basic_retrieve_returns_well_typed_arrays():
    r = mojo_bm25s.Retriever().index(SMALL_CORPUS)
    scores, ids = r.retrieve(["fish swim"], k=2)
    assert scores.dtype == np.float32, (
        f"scores must be float32, got {scores.dtype}"
    )
    assert ids.dtype == np.int32, (
        f"ids must be int32, got {ids.dtype}"
    )
    assert scores.shape == (1, 2)
    assert ids.shape == (1, 2)


def test_quickstart_from_issue_body():
    """The exact snippet from issue #27's Acceptance section."""
    r = mojo_bm25s.Retriever()
    r.index(["a cat is small", "a dog is big", "fish swim"])
    scores, ids = r.retrieve(["fish"], k=2)
    assert scores.shape == (1, 2)
    assert ids.shape == (1, 2)
    # "fish" appears only in doc 2 → it must be the top result with
    # a strictly positive score. The other slot is a no-match (zero).
    assert int(ids[0, 0]) == 2, (
        f"top hit for 'fish' must be doc 2, got {int(ids[0, 0])}"
    )
    assert scores[0, 0] > 0, (
        f"top score for 'fish' must be positive, got {float(scores[0, 0])}"
    )


def test_retrieve_descending_score_per_row():
    r = mojo_bm25s.Retriever().index(SMALL_CORPUS)
    scores, _ = r.retrieve(["the quick lazy dog"], k=3)
    diffs = np.diff(scores, axis=1)
    assert (diffs <= 1e-7).all()


# ----------------------------------------------------------------------
# Method coverage — make sure the Retriever wires each build_index method.
# ----------------------------------------------------------------------

@pytest.mark.parametrize("method", ["lucene", "atire", "bm25l", "bm25+"])
def test_methods_wired_correctly(method):
    """Each method must produce non-zero scores for an obvious query
    match. This is a regression guard: a Retriever that hard-codes
    'lucene' under the hood would silently pass scores but fail
    differential checks downstream.
    """
    r = mojo_bm25s.Retriever(method=method).index(SMALL_CORPUS)
    scores, ids = r.retrieve(["fish swim"], k=1)
    # doc 2 ("fish swim in the sea") must be the top hit for any
    # reasonable BM25 method.
    assert int(ids[0, 0]) == 2, (
        f"method={method}: expected top hit doc 2, got {int(ids[0, 0])}"
    )
    assert float(scores[0, 0]) > 0, (
        f"method={method}: expected positive score, got {float(scores[0, 0])}"
    )


# ----------------------------------------------------------------------
# Stopwords customization
# ----------------------------------------------------------------------

def test_custom_stopwords_set():
    """A custom stopword set must change the vocab the Retriever builds."""
    # With 'fish' as a stopword, the query "fish" should have no in-vocab
    # tokens and return all-zero scores.
    r_filtered = mojo_bm25s.Retriever(stopwords={"fish"}).index(SMALL_CORPUS)
    scores, _ = r_filtered.retrieve(["fish"], k=2)
    assert np.allclose(scores, 0.0), (
        "with 'fish' as a stopword, the query 'fish' must return zeros"
    )


def test_no_stopwords_via_none():
    """``stopwords=None`` must mean "no filtering" — common stopwords
    like 'the' end up in the vocab and queries on them score."""
    r = mojo_bm25s.Retriever(stopwords=None).index(SMALL_CORPUS)
    scores, _ = r.retrieve(["the"], k=3)
    # 'the' appears in multiple docs, so the top score must be > 0.
    assert float(scores[0, 0]) > 0, (
        f"with stopwords=None, query 'the' must score > 0; got {scores[0]}"
    )


# ----------------------------------------------------------------------
# Stemmer integration — both corpus AND query must be stemmed.
# ----------------------------------------------------------------------

def test_stemmer_applied_to_corpus_and_queries():
    """If a stemmer is configured, the query side must stem too —
    otherwise the query token never matches the stemmed vocab entry.

    Fixture: docs say "runs" / "running" / "ran"; query says "running".
    Without stemming, "running" only matches the second doc (exact).
    With stemming, the stems collide ("run" / "run" / "ran") so "running"
    → "run" matches doc 0 AND doc 1.
    """
    corpus = [
        "the cat runs fast",
        "the dog is running quickly",
        "the bird flew yesterday",
    ]

    import Stemmer
    stemmer = Stemmer.Stemmer("english").stemWord

    r_stem = mojo_bm25s.Retriever(stemmer=stemmer).index(corpus)
    scores_stem, ids_stem = r_stem.retrieve(["running"], k=3)
    # Top two hits must be docs 0 and 1 (both contain run-stem tokens),
    # and both must have strictly positive scores.
    top_two = set(int(x) for x in ids_stem[0, :2])
    assert top_two == {0, 1}, (
        f"stemmed retriever: top-2 for 'running' must be {{0, 1}}, "
        f"got {top_two}; ids={ids_stem[0].tolist()} "
        f"scores={scores_stem[0].tolist()}"
    )
    assert float(scores_stem[0, 0]) > 0
    assert float(scores_stem[0, 1]) > 0

    # Sanity check: WITHOUT a stemmer, only doc 1 (exact "running") scores.
    r_nostem = mojo_bm25s.Retriever().index(corpus)
    scores_ns, ids_ns = r_nostem.retrieve(["running"], k=3)
    assert int(ids_ns[0, 0]) == 1, (
        f"no-stemmer retriever: top hit for 'running' must be doc 1 (exact), "
        f"got {int(ids_ns[0, 0])}"
    )
    # Doc 0 only had "runs", which doesn't match "running" without stemming.
    assert float(scores_ns[0, 1]) == 0.0, (
        f"no-stemmer: second-rank score must be 0 (no other matches); "
        f"got {float(scores_ns[0, 1])}"
    )


# ----------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------

def test_retrieve_before_index_raises():
    r = mojo_bm25s.Retriever()
    with pytest.raises((RuntimeError, ValueError)) as exc_info:
        r.retrieve(["fish"], k=2)
    # Error message must mention 'index' — give the user a hint, not a
    # raw AttributeError.
    assert "index" in str(exc_info.value).lower(), (
        f"unhelpful error message: {exc_info.value!r}"
    )


def test_empty_corpus_is_rejected_with_clear_error():
    """Empty corpus is ambiguous — there is no vocab to score against, so
    every query returns zeros. We require the Retriever to reject this
    up front rather than silently produce zeros forever."""
    r = mojo_bm25s.Retriever()
    with pytest.raises((ValueError, RuntimeError)) as exc_info:
        r.index([])
    assert "empty" in str(exc_info.value).lower() or "corpus" in str(exc_info.value).lower(), (
        f"unhelpful error message for empty corpus: {exc_info.value!r}"
    )


def test_query_with_zero_in_vocab_tokens_returns_zeros():
    """A query consisting entirely of OOV tokens (or stopwords) must
    return zero scores — not raise, not crash on the kernel side."""
    r = mojo_bm25s.Retriever().index(SMALL_CORPUS)
    # 'zebra' and 'aardvark' don't appear in SMALL_CORPUS.
    scores, ids = r.retrieve(["zebra aardvark"], k=3)
    assert scores.shape == (1, 3)
    assert ids.shape == (1, 3)
    assert np.all(scores == 0.0), (
        f"OOV-only query must return zero scores, got {scores[0]}"
    )


def test_index_called_twice_replaces_not_appends():
    """Re-indexing must replace, not extend. A retrieve after the second
    index must see only the second corpus's docs."""
    r = mojo_bm25s.Retriever()
    r.index(["alpha beta gamma", "delta epsilon zeta"])
    # Re-index with a smaller corpus.
    r.index(["just one document with text"])
    scores, ids = r.retrieve(["one"], k=1)
    # Only one doc exists post-replace; its ID must be 0.
    assert int(ids[0, 0]) == 0, (
        f"after re-index, top hit must be doc 0 in the NEW corpus, "
        f"got {int(ids[0, 0])}"
    )
    # And we must NOT be able to retrieve anything matching the OLD corpus
    # — 'alpha' is no longer in vocab → zero score.
    scores_old, _ = r.retrieve(["alpha"], k=1)
    assert float(scores_old[0, 0]) == 0.0, (
        f"after re-index, OLD-corpus token 'alpha' must score 0; "
        f"got {float(scores_old[0, 0])}"
    )


# ----------------------------------------------------------------------
# Multi-query batching produces a single (batch, k) array
# ----------------------------------------------------------------------

def test_multi_query_batch_shape():
    r = mojo_bm25s.Retriever().index(SMALL_CORPUS)
    scores, ids = r.retrieve(["fish", "dog", "the lazy fox"], k=2)
    assert scores.shape == (3, 2)
    assert ids.shape == (3, 2)


# ----------------------------------------------------------------------
# Save / load round trip
# ----------------------------------------------------------------------

def test_save_and_load_round_trip_returns_identical_results(tmp_path):
    """Build → save → load → retrieve must produce identical (scores, ids)
    to the original in-memory retriever for the same queries."""
    r = mojo_bm25s.Retriever().index(SMALL_CORPUS)
    queries = ["the lazy dog", "fish swim", "quick brown fox"]
    s_mem, i_mem = r.retrieve(queries, k=3)

    idx_dir = tmp_path / "test_idx"
    r.save(idx_dir)
    assert idx_dir.is_dir(), "save() must create the directory"

    r2 = mojo_bm25s.Retriever.load(idx_dir)
    s_load, i_load = r2.retrieve(queries, k=3)

    # Byte-identical — same arrays, same kernel, deterministic.
    np.testing.assert_array_equal(s_mem, s_load)
    np.testing.assert_array_equal(i_mem, i_load)


def test_load_classmethod_returns_retriever_instance(tmp_path):
    r = mojo_bm25s.Retriever().index(SMALL_CORPUS)
    r.save(tmp_path / "idx")
    loaded = mojo_bm25s.Retriever.load(tmp_path / "idx")
    assert isinstance(loaded, mojo_bm25s.Retriever)


def test_save_load_preserves_hyperparams(tmp_path):
    """The loaded retriever must use the saved k1/b/delta/method/idf_method —
    a no-op load would silently fall back to defaults and produce wrong
    scores for non-default configs."""
    r = mojo_bm25s.Retriever(
        method="bm25+", idf_method="lucene",
        k1=2.0, b=0.5, delta=1.0,
    ).index(SMALL_CORPUS)
    queries = ["fish", "the lazy dog"]
    s_mem, i_mem = r.retrieve(queries, k=2)

    r.save(tmp_path / "idx2")
    r2 = mojo_bm25s.Retriever.load(tmp_path / "idx2")
    s_load, i_load = r2.retrieve(queries, k=2)

    np.testing.assert_array_equal(s_mem, s_load)
    np.testing.assert_array_equal(i_mem, i_load)


def test_save_load_with_named_porter_stemmer_round_trips(tmp_path):
    """``Retriever(stemmer=mojo_bm25s.stem)`` MUST round-trip: the stemmer
    is the in-tree Porter implementation, identifiable by reference, so
    save/load reconstructs it. The loaded retriever stems queries the
    same way → retrieve output matches.

    Arbitrary user-supplied callables can't be persisted (documented
    contract); we test the named-stemmer case here.
    """
    r = mojo_bm25s.Retriever(stemmer=mojo_bm25s.stem).index(SMALL_CORPUS)
    queries = ["running fast", "the lazy dog"]
    s_mem, i_mem = r.retrieve(queries, k=2)

    r.save(tmp_path / "idx_stem")
    r2 = mojo_bm25s.Retriever.load(tmp_path / "idx_stem")
    s_load, i_load = r2.retrieve(queries, k=2)

    np.testing.assert_array_equal(s_mem, s_load)
    np.testing.assert_array_equal(i_mem, i_load)


# Headline parity tests live in tests/parity/test_retriever_standalone.py
# alongside the existing patch_bm25s parity suite — that's where the
# scifact fixture is defined.
