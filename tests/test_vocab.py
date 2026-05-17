"""Tests for ``mojo_bm25s.vocab.Vocab`` (issue #24).

Contract (locked here, see ``vocab.py`` docstring for the design notes):

- Ordering: tokens are assigned IDs in **first-occurrence order** across
  the corpus (deterministic; documented choice from the two options in
  the issue).
- Unknown query tokens → ``-1`` (the issue-specified sentinel; the
  retriever in #27 filters these out before they hit the kernel).
- ``tokens_to_ids`` returns a ``np.ndarray[int32]``.
- Save format: a directory containing a single ``vocab.json`` whose
  payload is a JSON object with a ``"tokens"`` key — a list of token
  strings in ID order. Stable, boring, and small.
- Composition with the tokenizer: ``Vocab.from_corpus(tokenize(...))``
  produces a vocab whose size on BEIR scifact matches
  ``bm25s.BM25().index(...).vocab_dict`` within a small delta (we don't
  carry the empty-string sentinel that bm25s appends, so we expect to
  be exactly 1 smaller per the bm25s source — assertion is "delta == 1"
  with a generous tolerance to absorb any non-determinism in the bm25s
  set-based vocab).

Tests progress from contract micro-cases (so a stub is RED for the
right reason — wrong IDs / wrong dtype / missing reverse lookup) up to
the scifact parity acceptance check.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


# --- Module surface ---------------------------------------------------------

def test_vocab_is_exported_from_package():
    import mojo_bm25s

    assert hasattr(mojo_bm25s, "Vocab"), (
        "mojo_bm25s.Vocab should be exported per issue #24 API"
    )


def test_vocab_constructable_with_no_args():
    from mojo_bm25s import Vocab

    v = Vocab()
    assert len(v) == 0


# --- First-occurrence ordering (the locked-in choice) ----------------------

def test_first_occurrence_ordering_single_doc():
    """A single doc — IDs assigned in encounter order, left-to-right."""
    from mojo_bm25s import Vocab

    v = Vocab.from_corpus([["alpha", "beta", "gamma"]])
    # IDs are 0, 1, 2 in that exact order.
    ids = v.tokens_to_ids(["alpha", "beta", "gamma"])
    assert ids.tolist() == [0, 1, 2], (
        f"first-occurrence ordering broken: got {ids.tolist()!r}"
    )


def test_first_occurrence_ordering_multi_doc_hand_assertion():
    """Hand-asserted IDs across multiple docs — first occurrence wins."""
    from mojo_bm25s import Vocab

    corpus = [
        ["the", "quick", "brown", "fox"],
        ["quick", "brown", "dog"],            # quick, brown already seen
        ["lazy", "fox", "jumps"],             # fox already seen
    ]
    v = Vocab.from_corpus(corpus)
    # Order of first encounter: the, quick, brown, fox, dog, lazy, jumps
    expected = {
        "the": 0,
        "quick": 1,
        "brown": 2,
        "fox": 3,
        "dog": 4,
        "lazy": 5,
        "jumps": 6,
    }
    for tok, eid in expected.items():
        ids = v.tokens_to_ids([tok])
        assert ids[0] == eid, (
            f"token {tok!r}: expected ID {eid}, got {int(ids[0])}; "
            f"first-occurrence ordering broken"
        )
    assert len(v) == 7


def test_duplicate_tokens_within_doc_only_register_once():
    from mojo_bm25s import Vocab

    v = Vocab.from_corpus([["a", "a", "a", "b"], ["a", "c", "c"]])
    # Distinct tokens: a, b, c
    assert len(v) == 3
    assert v.tokens_to_ids(["a", "b", "c"]).tolist() == [0, 1, 2]


# --- Round trip: corpus → vocab → ids → tokens-by-id -----------------------

def test_round_trip_token_to_id_to_token():
    """For every token in the corpus, ``vocab[vocab.tokens_to_ids([tok])[0]]``
    returns the same token. Uses the public reverse lookup (token-by-id) via
    ``v.id_to_token`` accessor — see API docstring."""
    from mojo_bm25s import Vocab

    corpus = [
        ["alpha", "beta", "gamma"],
        ["beta", "delta"],
        ["epsilon", "alpha"],
    ]
    v = Vocab.from_corpus(corpus)

    unique_tokens = []
    seen = set()
    for doc in corpus:
        for t in doc:
            if t not in seen:
                seen.add(t)
                unique_tokens.append(t)

    ids = v.tokens_to_ids(unique_tokens)
    # Every id must round-trip back via id_to_token (the reverse lookup).
    for tok, i in zip(unique_tokens, ids.tolist()):
        assert v.id_to_token(i) == tok, (
            f"round-trip failed: token={tok!r} -> id={i} -> "
            f"{v.id_to_token(i)!r}"
        )


# --- Unknown query tokens --------------------------------------------------

def test_unknown_tokens_return_minus_one_by_default():
    from mojo_bm25s import Vocab

    v = Vocab.from_corpus([["foo", "bar"]])
    ids = v.tokens_to_ids(["foo", "unseen", "bar", "alien"])
    assert ids.tolist() == [0, -1, 1, -1]


def test_unknown_sentinel_is_customizable():
    """The ``unknown`` kwarg overrides -1 — useful if a caller wants a
    different sentinel (e.g. ``n_vocab`` for sparse-row fallback)."""
    from mojo_bm25s import Vocab

    v = Vocab.from_corpus([["foo", "bar"]])
    ids = v.tokens_to_ids(["foo", "unseen"], unknown=-99)
    assert ids.tolist() == [0, -99]


# --- dtype / shape ---------------------------------------------------------

def test_tokens_to_ids_returns_int32_ndarray():
    from mojo_bm25s import Vocab

    v = Vocab.from_corpus([["foo", "bar"]])
    out = v.tokens_to_ids(["foo", "bar"])
    assert isinstance(out, np.ndarray), f"expected ndarray, got {type(out)}"
    assert out.dtype == np.int32, (
        f"tokens_to_ids must return int32 (kernel-compatible); got {out.dtype}"
    )
    assert out.shape == (2,), f"expected shape (2,), got {out.shape}"


def test_tokens_to_ids_empty_input_returns_empty_int32():
    from mojo_bm25s import Vocab

    v = Vocab.from_corpus([["foo"]])
    out = v.tokens_to_ids([])
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.int32
    assert out.shape == (0,)


# --- __len__ ---------------------------------------------------------------

def test_len_equals_n_vocab():
    from mojo_bm25s import Vocab

    v = Vocab.from_corpus([["a", "b"], ["c"], ["a"]])
    assert len(v) == 3


# --- Save / load round-trip ------------------------------------------------

def test_save_load_round_trip_preserves_ids(tmp_path):
    from mojo_bm25s import Vocab

    corpus = [
        ["alpha", "beta"],
        ["gamma", "alpha"],
        ["delta", "epsilon", "beta"],
    ]
    v = Vocab.from_corpus(corpus)

    save_dir = tmp_path / "myvocab"
    v.save(save_dir)

    v2 = Vocab.load(save_dir)
    assert len(v) == len(v2)

    # Every token must map to the same id in both, and id_to_token agrees.
    flat = sorted({t for doc in corpus for t in doc})
    ids_before = v.tokens_to_ids(flat)
    ids_after = v2.tokens_to_ids(flat)
    np.testing.assert_array_equal(ids_before, ids_after)

    for i in range(len(v)):
        assert v.id_to_token(i) == v2.id_to_token(i), (
            f"id_to_token mismatch at id={i}: "
            f"{v.id_to_token(i)!r} vs {v2.id_to_token(i)!r}"
        )


def test_save_emits_vocab_json_in_target_directory(tmp_path):
    """Lock the on-disk layout: a single ``vocab.json`` inside the
    save-dir, with a top-level ``"tokens"`` key listing tokens by ID.
    A future cleanup that breaks this layout breaks this test —
    intentional contract surface (see issue #24 + PR body)."""
    from mojo_bm25s import Vocab

    v = Vocab.from_corpus([["zero", "one", "two"]])
    save_dir = tmp_path / "ondisk"
    v.save(save_dir)

    layout_file = save_dir / "vocab.json"
    assert layout_file.exists(), (
        "expected vocab.json inside save-dir; on-disk layout is a "
        "documented contract."
    )
    payload = json.loads(layout_file.read_text(encoding="utf-8"))
    assert "tokens" in payload, "vocab.json must have a 'tokens' key"
    assert payload["tokens"] == ["zero", "one", "two"], (
        "tokens list must be in ID order (index == id)"
    )


def test_load_on_missing_dir_raises(tmp_path):
    from mojo_bm25s import Vocab

    with pytest.raises((FileNotFoundError, OSError)):
        Vocab.load(tmp_path / "does-not-exist")


# --- Edge cases ------------------------------------------------------------

def test_empty_corpus_yields_empty_vocab():
    from mojo_bm25s import Vocab

    v = Vocab.from_corpus([])
    assert len(v) == 0


def test_corpus_of_empty_docs_yields_empty_vocab():
    from mojo_bm25s import Vocab

    v = Vocab.from_corpus([[], [], []])
    assert len(v) == 0


def test_single_doc_single_repeated_token():
    from mojo_bm25s import Vocab

    v = Vocab.from_corpus([["only", "only", "only"]])
    assert len(v) == 1
    assert v.tokens_to_ids(["only"]).tolist() == [0]
    assert v.id_to_token(0) == "only"


def test_unicode_tokens_pass_through():
    """The tokenizer keeps unicode word characters; vocab must too."""
    from mojo_bm25s import Vocab

    v = Vocab.from_corpus([["café", "naïve", "résumé"], ["café"]])
    assert len(v) == 3
    ids = v.tokens_to_ids(["café", "naïve", "résumé"])
    assert ids.tolist() == [0, 1, 2]
    # And round-trip via id_to_token preserves the bytes verbatim.
    assert v.id_to_token(0) == "café"
    assert v.id_to_token(2) == "résumé"


def test_unicode_round_trip_via_save_load(tmp_path):
    from mojo_bm25s import Vocab

    v = Vocab.from_corpus([["café", "naïve"]])
    v.save(tmp_path / "u")
    v2 = Vocab.load(tmp_path / "u")
    assert v2.id_to_token(0) == "café"
    assert v2.id_to_token(1) == "naïve"


# --- Composition with tokenizer --------------------------------------------

def test_composition_with_tokenizer():
    """Build a vocab from ``tokenize`` output and check size matches the
    unique-token count after stopword filtering."""
    from mojo_bm25s import Vocab, tokenize

    texts = ["the quick brown fox", "quick brown dog"]
    tokens = tokenize(texts)  # default stopwords = English
    # Post-stopword: ["quick","brown","fox"], ["quick","brown","dog"]
    # Unique: {quick, brown, fox, dog} == 4
    v = Vocab.from_corpus(tokens)
    assert len(v) == 4
    # First-occurrence: quick=0, brown=1, fox=2, dog=3
    ids = v.tokens_to_ids(["quick", "brown", "fox", "dog"])
    assert ids.tolist() == [0, 1, 2, 3]


# --- BEIR scifact parity vs bm25s ------------------------------------------

@pytest.fixture(scope="module")
def scifact_corpus_tokens():
    """Raw scifact corpus tokenized with bm25s (matches what bm25s.BM25
    will see internally). We use bm25s.tokenize here — same input on
    both sides keeps the comparison about vocab construction, not
    tokenizer differences."""
    _REPO_ROOT = Path(__file__).resolve().parent.parent
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from benchmarks.datasets import load_beir

    import bm25s
    ds = load_beir("scifact")
    return bm25s.tokenize(
        ds.corpus, stopwords="en", stemmer=None, return_ids=False,
        show_progress=False,
    )


def test_scifact_vocab_size_matches_bm25s_within_small_delta(scifact_corpus_tokens):
    """Acceptance per issue #24: vocab size on BEIR scifact matches
    ``bm25s.BM25().index(...).vocab_dict`` within a small delta.

    bm25s appends an empty-token sentinel (`""`) as the last vocab entry
    iff the corpus is given as token lists (not pre-converted IDs), so
    we expect ``bm25s_size - our_size == 1`` exactly. Test asserts
    ``abs(delta) <= 2`` to absorb any incidental bm25s changes."""
    import bm25s
    from mojo_bm25s import Vocab

    ours = Vocab.from_corpus(scifact_corpus_tokens)

    ref = bm25s.BM25()
    ref.index(scifact_corpus_tokens, show_progress=False)
    bm25s_size = len(ref.vocab_dict)

    delta = bm25s_size - len(ours)
    assert abs(delta) <= 2, (
        f"vocab size delta vs bm25s = {delta} "
        f"(ours={len(ours)}, bm25s={bm25s_size}); "
        f"target: |delta| <= 2 (bm25s appends '' so delta=1 is expected)"
    )
