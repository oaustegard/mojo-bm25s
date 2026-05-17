"""Tests for ``mojo_bm25s.tokenize`` (issue #22).

Parity oracle: ``bm25s.tokenize(..., stopwords='en', stemmer=None,
return_ids=False, show_progress=False)`` — the underlying behavior
that gets exercised is the sklearn-style regex ``(?u)\\b\\w\\w+\\b``,
lowercase, and the canonical bm25s ``STOPWORDS_EN`` list.

Tests progress from contract micro-cases (so a stub is RED for the
right reason — wrong token contents, not ImportError) up to a small
hand-picked parity fixture, then a full BEIR scifact ≥99% overlap
acceptance check.
"""

from __future__ import annotations

import bm25s
import pytest


def _bm25s_tokens(texts, stopwords="en"):
    """Reference: call bm25s with stemmer disabled and return list-of-lists."""
    return bm25s.tokenize(
        texts,
        stopwords=stopwords,
        stemmer=None,
        return_ids=False,
        show_progress=False,
    )


# --- Module surface ---------------------------------------------------------

def test_tokenize_is_exported_from_package():
    import mojo_bm25s

    assert hasattr(mojo_bm25s, "tokenize"), (
        "mojo_bm25s.tokenize should be exported per issue #22 API"
    )
    assert callable(mojo_bm25s.tokenize)


def test_tokenize_returns_list_of_lists_of_str():
    from mojo_bm25s import tokenize

    out = tokenize(["hello world"])
    assert isinstance(out, list)
    assert len(out) == 1
    assert isinstance(out[0], list)
    assert all(isinstance(t, str) for t in out[0])


# --- Lowercase + basic split -----------------------------------------------

def test_simple_lowercase_and_split():
    from mojo_bm25s import tokenize

    out = tokenize(["Hello World"])
    # default stopwords = English ("hello", "world" are not stopwords)
    assert out == [["hello", "world"]]


def test_lowercase_false_preserves_case():
    from mojo_bm25s import tokenize

    out = tokenize(["Hello World"], lowercase=False, stopwords=None)
    assert out == [["Hello", "World"]]


# --- Stopword filtering -----------------------------------------------------

def test_default_stopwords_remove_english():
    from mojo_bm25s import tokenize

    # "the", "and" are in bm25s STOPWORDS_EN
    out = tokenize(["the quick brown fox and the lazy dog"])
    assert out == [["quick", "brown", "fox", "lazy", "dog"]]


def test_stopwords_none_keeps_everything():
    from mojo_bm25s import tokenize

    out = tokenize(["the quick brown fox"], stopwords=None)
    assert out == [["the", "quick", "brown", "fox"]]


def test_stopwords_custom_set():
    from mojo_bm25s import tokenize

    out = tokenize(["foo bar baz qux"], stopwords={"bar", "qux"})
    assert out == [["foo", "baz"]]


def test_stopwords_custom_list_also_works():
    """Custom stopwords can be a list, not just a set — match bm25s flexibility."""
    from mojo_bm25s import tokenize

    out = tokenize(["foo bar baz"], stopwords=["bar"])
    assert out == [["foo", "baz"]]


# --- Edge cases -------------------------------------------------------------

def test_empty_input_list():
    from mojo_bm25s import tokenize

    assert tokenize([]) == []


def test_empty_string_yields_empty_inner_list():
    from mojo_bm25s import tokenize

    # bm25s returns [[]] for [""]
    assert tokenize([""]) == _bm25s_tokens([""])


def test_punctuation_only_yields_empty_inner_list():
    from mojo_bm25s import tokenize

    assert tokenize(["!!!"]) == _bm25s_tokens(["!!!"])
    assert tokenize(["..."]) == _bm25s_tokens(["..."])
    assert tokenize(["?!.,;:"]) == _bm25s_tokens(["?!.,;:"])


def test_single_char_tokens_are_dropped():
    """Per bm25s regex ``\\b\\w\\w+\\b`` — 2+ word chars required."""
    from mojo_bm25s import tokenize

    # "a b c d" — none survive the 2+-char rule
    assert tokenize(["a b c d"], stopwords=None) == [[]]
    # Mixed: "I am here" — "I" and "am" — "am" is 2 chars, kept; "I" dropped
    out = tokenize(["I am here"], stopwords=None)
    # bm25s lowercases first then matches
    assert out == _bm25s_tokens(["I am here"], stopwords=None)


def test_numbers_kept_as_tokens():
    from mojo_bm25s import tokenize

    out = tokenize(["test 123 word42"], stopwords=None)
    assert out == _bm25s_tokens(["test 123 word42"], stopwords=None)
    assert "123" in out[0]
    assert "word42" in out[0]


def test_hyphens_split_words():
    """bm25s regex splits on non-word chars including '-'."""
    from mojo_bm25s import tokenize

    out = tokenize(["state-of-the-art well-known"])
    assert out == _bm25s_tokens(["state-of-the-art well-known"])


def test_non_ascii_passthrough():
    """Unicode word chars (`\\w` with re.UNICODE) survive."""
    from mojo_bm25s import tokenize

    out = tokenize(["café naïve résumé"], stopwords=None)
    assert out == _bm25s_tokens(["café naïve résumé"], stopwords=None)
    # And specifically, the accented chars are preserved
    assert any("café" in tok for tok in out[0])


def test_multiple_documents():
    from mojo_bm25s import tokenize

    docs = ["First document text", "Second document here", "Third one"]
    out = tokenize(docs)
    assert out == _bm25s_tokens(docs)


def test_whitespace_variants():
    """Tabs, newlines, multiple spaces — same regex semantics."""
    from mojo_bm25s import tokenize

    out = tokenize(["foo\tbar\nbaz   qux"], stopwords=None)
    assert out == _bm25s_tokens(["foo\tbar\nbaz   qux"], stopwords=None)


# --- Small hand-picked fixture parity --------------------------------------

HAND_FIXTURE = [
    "The quick brown fox jumps over the lazy dog.",
    "BM25 is a ranking function used by search engines to estimate relevance.",
    "Information retrieval (IR) is the science of searching for information.",
    "Mojo-native kernels promise C-level speed with Python ergonomics.",
    "All work and no play makes Jack a dull boy.",
    "She sells seashells by the seashore on a sunny afternoon.",
    "Tokenization splits text into atomic units called tokens.",
    "Stopwords like 'the', 'and', 'is' are often removed before indexing.",
    "Naïve tokenizers ignore café names and résumé content.",
    "Numbers 123 and 42 should survive tokenization as well.",
]


def test_hand_fixture_exact_match():
    """Every doc in the hand-picked fixture matches bm25s exactly."""
    from mojo_bm25s import tokenize

    ours = tokenize(HAND_FIXTURE)
    theirs = _bm25s_tokens(HAND_FIXTURE)
    assert ours == theirs, (
        f"hand-fixture parity broke: ours={ours!r} theirs={theirs!r}"
    )


# --- BEIR scifact ≥99% overlap acceptance ----------------------------------

@pytest.fixture(scope="module")
def scifact_corpus():
    """Raw scifact corpus texts (re-uses parity loader)."""
    import sys
    from pathlib import Path

    _REPO_ROOT = Path(__file__).resolve().parent.parent
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    from benchmarks.datasets import load_beir

    ds = load_beir("scifact")
    return ds.corpus


def test_scifact_token_overlap_at_least_99_percent(scifact_corpus):
    """Per issue acceptance: ≥99% per-document token overlap with bm25s.

    Overlap is computed per doc as
        |intersection(ours_set, theirs_set)| / |union(ours_set, theirs_set)|
    averaged over docs. (Set semantics, not multiset — matches the spirit
    of "downstream retrieval scores within ~5%".)
    """
    from mojo_bm25s import tokenize

    ours = tokenize(scifact_corpus)
    theirs = _bm25s_tokens(scifact_corpus)
    assert len(ours) == len(theirs)

    overlaps = []
    for o, t in zip(ours, theirs):
        os_, ts_ = set(o), set(t)
        if not os_ and not ts_:
            overlaps.append(1.0)
            continue
        union = os_ | ts_
        inter = os_ & ts_
        overlaps.append(len(inter) / len(union))

    mean_overlap = sum(overlaps) / len(overlaps)
    assert mean_overlap >= 0.99, (
        f"mean per-doc token overlap with bm25s = {mean_overlap:.4f}, "
        f"below 0.99 threshold (issue #22 acceptance)"
    )
