"""Tests for binary index persistence (issue #26).

Contract (locked here):

- ``save_index`` writes a directory containing ``meta.json``, ``vocab.bin``,
  ``data.bin``, ``indices.bin``, ``indptr.bin``, and ``nonoccurrence.bin``
  iff ``method in {"bm25l", "bm25+"}``.
- ``load_index`` reads that directory and returns a dataclass-like result
  whose array fields round-trip byte-for-byte (``np.array_equal``, not
  ``allclose`` — there is no math here, just bytes on disk).
- End-to-end: ``build → save → load → csc_score(loaded arrays) ==
  csc_score(in-memory arrays)`` bitwise.
- Atomic write: a crash mid-write must leave the target directory absent
  (or at least not loadable as a complete index). After a successful
  ``save_index`` call, no ``.tmp`` directory remains.
- Version field: a ``meta.json`` whose ``version`` is greater than the
  loader's known version raises a clear error (forward-compat boundary).
- Vocab: included in-the-mix as ``vocab.bin``. id↔token mapping
  preserved bit-for-bit. Unicode tokens included.
- Edge cases: empty corpus (zero docs, zero vocab); single-doc corpus.
- Hyperparams: ``k1``, ``b``, ``delta``, ``method``, ``idf_method`` all
  preserved through save/load.

These tests must FAIL against a stub that just returns dummies — they
assert on real array contents, not on import surface.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest import mock

import numpy as np
import pytest


# ----------------------------------------------------------------------
# Tiny helper: build vocab + ids from a token-list corpus.
# ----------------------------------------------------------------------

def _build_vocab_and_ids(corpus_tokens):
    """Deterministic first-occurrence vocab + per-doc int32 id arrays.

    Mirrors the helper in ``test_index_builder.py`` so persistence tests
    use the same id ordering everywhere.
    """
    from mojo_bm25s import Vocab

    v = Vocab.from_corpus(corpus_tokens)
    ids = [v.tokens_to_ids(doc) for doc in corpus_tokens]
    return v, ids


SMALL_CORPUS = [
    ["the", "quick", "brown", "fox"],
    ["the", "lazy", "dog", "sleeps"],
    ["a", "quick", "fox", "jumps", "over", "the", "lazy", "dog"],
    ["brown", "fox", "fox", "fox"],  # repeated within-doc tf
]


def _build_small_index(method="lucene", idf_method=None, k1=1.5, b=0.75, delta=0.5):
    """Build a small in-memory index. Returns (vocab, fields-dict).

    fields-dict has every kwarg ``save_index`` expects.
    """
    from mojo_bm25s import build_index

    vocab, ids = _build_vocab_and_ids(SMALL_CORPUS)
    data, indices, indptr, n_docs, l_avg, nonoccurrence = build_index(
        ids, n_vocab=len(vocab),
        method=method, idf_method=idf_method, k1=k1, b=b, delta=delta,
    )
    return vocab, dict(
        data=data, indices=indices, indptr=indptr,
        n_docs=n_docs, l_avg=l_avg, nonoccurrence=nonoccurrence,
        method=method, idf_method=idf_method or method,
        k1=k1, b=b, delta=delta,
    )


# ----------------------------------------------------------------------
# Module surface
# ----------------------------------------------------------------------

def test_save_and_load_are_exported():
    import mojo_bm25s

    assert hasattr(mojo_bm25s, "save_index"), (
        "mojo_bm25s.save_index should be exported per issue #26"
    )
    assert hasattr(mojo_bm25s, "load_index"), (
        "mojo_bm25s.load_index should be exported per issue #26"
    )
    assert callable(mojo_bm25s.save_index)
    assert callable(mojo_bm25s.load_index)


# ----------------------------------------------------------------------
# Round trip: build → save → load → all arrays byte-identical
# ----------------------------------------------------------------------

def test_round_trip_array_fields_lucene(tmp_path):
    from mojo_bm25s import save_index, load_index

    vocab, fields = _build_small_index(method="lucene")
    index_dir = tmp_path / "idx"
    save_index(index_dir, vocab=vocab, **fields)

    loaded = load_index(index_dir)

    # Byte-for-byte. No math involved.
    assert np.array_equal(loaded.data, fields["data"]), (
        f"data mismatch after round-trip; dtypes "
        f"{loaded.data.dtype} vs {fields['data'].dtype}"
    )
    assert np.array_equal(loaded.indices, fields["indices"])
    assert np.array_equal(loaded.indptr, fields["indptr"])
    # And dtypes are preserved exactly.
    assert loaded.data.dtype == np.float32
    assert loaded.indices.dtype == np.int32
    assert loaded.indptr.dtype == np.int32


def test_round_trip_scalar_fields(tmp_path):
    from mojo_bm25s import save_index, load_index

    vocab, fields = _build_small_index(method="lucene")
    save_index(tmp_path / "idx", vocab=vocab, **fields)
    loaded = load_index(tmp_path / "idx")

    assert loaded.n_docs == fields["n_docs"]
    assert loaded.l_avg == pytest.approx(fields["l_avg"])


# ----------------------------------------------------------------------
# End-to-end retrieve parity: scores from loaded arrays match in-memory.
# ----------------------------------------------------------------------

def test_end_to_end_csc_score_matches_after_round_trip(tmp_path):
    """The whole point: a retrieve against the loaded arrays must give
    bit-identical scores to a retrieve against the in-memory arrays.
    Uses ``csc_score`` directly (the Mojo kernel) so we exercise the
    exact code path #27's Retriever will."""
    from mojo_bm25s import csc_score, save_index, load_index

    vocab, fields = _build_small_index(method="lucene")
    save_index(tmp_path / "idx", vocab=vocab, **fields)
    loaded = load_index(tmp_path / "idx")

    # A handful of queries spanning the vocab. Includes unknown tokens
    # (filtered to -1) and known tokens; we filter -1 ourselves since
    # csc_score doesn't accept negative ids.
    query_strs = [
        ["the", "fox"],
        ["lazy", "dog"],
        ["quick", "brown", "fox"],
        ["jumps", "over"],
    ]
    for qstrs in query_strs:
        q_in_memory = vocab.tokens_to_ids(qstrs)
        q_loaded = loaded.vocab.tokens_to_ids(qstrs)
        # Filter out unknowns (none expected here, but be defensive).
        q_in_memory = q_in_memory[q_in_memory >= 0]
        q_loaded = q_loaded[q_loaded >= 0]

        scores_mem = csc_score(
            fields["data"], fields["indptr"], fields["indices"],
            q_in_memory, n_docs=fields["n_docs"],
        )
        scores_loaded = csc_score(
            loaded.data, loaded.indptr, loaded.indices,
            q_loaded, n_docs=loaded.n_docs,
        )
        # Bit-identical: arrays are byte-for-byte equal, kernel is
        # deterministic, so scores match exactly.
        assert np.array_equal(scores_mem, scores_loaded), (
            f"score mismatch for query {qstrs}: "
            f"max |delta|={np.max(np.abs(scores_mem - scores_loaded))}"
        )


# ----------------------------------------------------------------------
# Atomic write: the .tmp directory pattern.
# ----------------------------------------------------------------------

def test_no_tmp_dir_remains_after_successful_save(tmp_path):
    """After a clean save, the ``index_dir.tmp/`` staging directory must
    be gone (rename-over-finalize)."""
    from mojo_bm25s import save_index

    vocab, fields = _build_small_index(method="lucene")
    target = tmp_path / "idx"
    save_index(target, vocab=vocab, **fields)

    assert target.exists(), "target dir should exist after save"
    tmp_dir = tmp_path / "idx.tmp"
    assert not tmp_dir.exists(), (
        f"staging dir {tmp_dir} should be renamed away after a clean save"
    )


def test_crash_mid_write_leaves_no_loadable_index(tmp_path):
    """Simulate a crash by patching ``os.replace`` (the final atomic
    rename) to raise. The target dir must NOT exist (loader would error
    on a non-existent dir, which is the correct end state).
    """
    from mojo_bm25s import save_index, load_index

    vocab, fields = _build_small_index(method="lucene")
    target = tmp_path / "idx"

    with mock.patch("os.replace", side_effect=RuntimeError("simulated crash")):
        with pytest.raises(RuntimeError, match="simulated crash"):
            save_index(target, vocab=vocab, **fields)

    # Target dir must not exist (we never renamed into it). The staging
    # dir may or may not exist depending on cleanup — but the loadable
    # entry point (``target``) must not be loadable as success.
    assert not target.exists(), (
        f"target dir {target} should not exist after a crashed save; "
        f"the atomic-rename pattern guarantees this"
    )
    with pytest.raises((FileNotFoundError, OSError)):
        load_index(target)


def test_repeated_save_overwrites_cleanly(tmp_path):
    """Saving twice into the same target should succeed — the second
    save must atomically replace the first."""
    from mojo_bm25s import save_index, load_index

    vocab, fields = _build_small_index(method="lucene")
    target = tmp_path / "idx"
    save_index(target, vocab=vocab, **fields)

    # Build with different hyperparams; save again.
    vocab2, fields2 = _build_small_index(method="atire", k1=2.0)
    save_index(target, vocab=vocab2, **fields2)

    loaded = load_index(target)
    assert loaded.method == "atire"
    assert loaded.k1 == pytest.approx(2.0)
    # Make sure no .tmp left over.
    assert not (tmp_path / "idx.tmp").exists()


# ----------------------------------------------------------------------
# Version field
# ----------------------------------------------------------------------

def test_load_rejects_higher_version(tmp_path):
    from mojo_bm25s import save_index, load_index

    vocab, fields = _build_small_index(method="lucene")
    target = tmp_path / "idx"
    save_index(target, vocab=vocab, **fields)

    # Bump version on disk.
    meta_path = target / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["version"] = meta["version"] + 1
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises((ValueError, RuntimeError)) as exc_info:
        load_index(target)
    # Error message should mention version (clear error per spec).
    assert "version" in str(exc_info.value).lower(), (
        f"version-rejection error should mention 'version'; got: {exc_info.value}"
    )


def test_meta_records_known_version(tmp_path):
    """meta.json on disk should have a numeric ``version`` field (the
    loader keys off it)."""
    from mojo_bm25s import save_index

    vocab, fields = _build_small_index(method="lucene")
    target = tmp_path / "idx"
    save_index(target, vocab=vocab, **fields)

    meta = json.loads((target / "meta.json").read_text(encoding="utf-8"))
    assert "version" in meta
    assert isinstance(meta["version"], int)
    assert meta["version"] >= 1


# ----------------------------------------------------------------------
# nonoccurrence array — bm25l / bm25+ have one, others don't.
# ----------------------------------------------------------------------

@pytest.mark.parametrize("method", ["bm25l", "bm25+"])
def test_nonoccurrence_round_trips_for_bm25l_bm25plus(tmp_path, method):
    from mojo_bm25s import save_index, load_index

    vocab, fields = _build_small_index(method=method)
    assert fields["nonoccurrence"] is not None, (
        f"sanity: {method} should produce a nonoccurrence array"
    )

    target = tmp_path / "idx"
    save_index(target, vocab=vocab, **fields)

    # File on disk.
    assert (target / "nonoccurrence.bin").exists(), (
        f"nonoccurrence.bin should be written for {method}"
    )

    loaded = load_index(target)
    assert loaded.nonoccurrence is not None
    assert loaded.nonoccurrence.dtype == np.float32
    assert np.array_equal(loaded.nonoccurrence, fields["nonoccurrence"])


@pytest.mark.parametrize("method", ["lucene", "atire", "robertson"])
def test_no_nonoccurrence_file_for_other_methods(tmp_path, method):
    from mojo_bm25s import save_index, load_index

    vocab, fields = _build_small_index(method=method)
    assert fields["nonoccurrence"] is None, (
        f"sanity: {method} should NOT produce a nonoccurrence array"
    )

    target = tmp_path / "idx"
    save_index(target, vocab=vocab, **fields)

    # File on disk must be absent (the issue locks this down).
    assert not (target / "nonoccurrence.bin").exists(), (
        f"nonoccurrence.bin should NOT exist for {method}"
    )

    loaded = load_index(target)
    assert loaded.nonoccurrence is None, (
        f"loader should reconstruct None for {method}"
    )


# ----------------------------------------------------------------------
# Vocab in-the-mix
# ----------------------------------------------------------------------

def test_vocab_round_trips_bit_for_bit(tmp_path):
    from mojo_bm25s import save_index, load_index

    vocab, fields = _build_small_index(method="lucene")
    target = tmp_path / "idx"
    save_index(target, vocab=vocab, **fields)

    loaded = load_index(target)
    v2 = loaded.vocab

    assert len(v2) == len(vocab)
    # Every id round-trips to the same token.
    for i in range(len(vocab)):
        assert v2.id_to_token(i) == vocab.id_to_token(i), (
            f"vocab id_to_token mismatch at i={i}: "
            f"{v2.id_to_token(i)!r} vs {vocab.id_to_token(i)!r}"
        )
    # And the same forward mapping.
    flat = [vocab.id_to_token(i) for i in range(len(vocab))]
    ids_before = vocab.tokens_to_ids(flat)
    ids_after = v2.tokens_to_ids(flat)
    np.testing.assert_array_equal(ids_before, ids_after)


def test_vocab_persisted_as_vocab_bin_not_vocab_json(tmp_path):
    """The new persistence format uses ``vocab.bin`` (binary, length-
    prefixed UTF-8). The issue spec locks this in.
    """
    from mojo_bm25s import save_index

    vocab, fields = _build_small_index(method="lucene")
    target = tmp_path / "idx"
    save_index(target, vocab=vocab, **fields)

    assert (target / "vocab.bin").exists(), (
        "vocab.bin should be the on-disk format per issue #26 spec"
    )


# ----------------------------------------------------------------------
# Edge cases: empty corpus, single doc, unicode
# ----------------------------------------------------------------------

def test_empty_corpus_round_trip(tmp_path):
    """Zero docs, zero vocab: all arrays empty, indptr is [0] of length 1."""
    from mojo_bm25s import Vocab, build_index, save_index, load_index

    v = Vocab.from_corpus([])
    assert len(v) == 0
    data, indices, indptr, n_docs, l_avg, nonoccurrence = build_index(
        [], n_vocab=0, method="lucene",
    )
    assert data.shape == (0,)
    assert indices.shape == (0,)
    assert indptr.shape == (1,)  # n_vocab + 1
    assert n_docs == 0

    target = tmp_path / "empty"
    save_index(
        target, vocab=v,
        data=data, indices=indices, indptr=indptr,
        n_docs=n_docs, l_avg=l_avg, nonoccurrence=nonoccurrence,
        method="lucene", idf_method="lucene",
        k1=1.5, b=0.75, delta=0.5,
    )

    loaded = load_index(target)
    assert np.array_equal(loaded.data, data)
    assert np.array_equal(loaded.indices, indices)
    assert np.array_equal(loaded.indptr, indptr)
    assert loaded.indptr.shape == (1,)
    assert loaded.n_docs == 0
    assert len(loaded.vocab) == 0
    assert loaded.nonoccurrence is None


def test_single_doc_corpus_round_trip(tmp_path):
    from mojo_bm25s import Vocab, build_index, save_index, load_index

    corpus = [["alpha", "beta", "gamma"]]
    v = Vocab.from_corpus(corpus)
    ids = [v.tokens_to_ids(d) for d in corpus]
    data, indices, indptr, n_docs, l_avg, nonoccurrence = build_index(
        ids, n_vocab=len(v), method="lucene",
    )

    target = tmp_path / "single"
    save_index(
        target, vocab=v,
        data=data, indices=indices, indptr=indptr,
        n_docs=n_docs, l_avg=l_avg, nonoccurrence=nonoccurrence,
        method="lucene", idf_method="lucene",
        k1=1.5, b=0.75, delta=0.5,
    )

    loaded = load_index(target)
    assert np.array_equal(loaded.data, data)
    assert np.array_equal(loaded.indices, indices)
    assert np.array_equal(loaded.indptr, indptr)
    assert loaded.n_docs == 1
    assert len(loaded.vocab) == 3


def test_unicode_vocab_round_trip(tmp_path):
    """Unicode tokens must survive the binary encode/decode."""
    from mojo_bm25s import Vocab, build_index, save_index, load_index

    corpus = [["café", "naïve", "résumé"], ["café", "日本語"]]
    v = Vocab.from_corpus(corpus)
    ids = [v.tokens_to_ids(d) for d in corpus]
    data, indices, indptr, n_docs, l_avg, nonoccurrence = build_index(
        ids, n_vocab=len(v), method="lucene",
    )

    target = tmp_path / "u"
    save_index(
        target, vocab=v,
        data=data, indices=indices, indptr=indptr,
        n_docs=n_docs, l_avg=l_avg, nonoccurrence=nonoccurrence,
        method="lucene", idf_method="lucene",
        k1=1.5, b=0.75, delta=0.5,
    )

    loaded = load_index(target)
    for i in range(len(v)):
        assert loaded.vocab.id_to_token(i) == v.id_to_token(i), (
            f"unicode round-trip broke at i={i}: "
            f"{loaded.vocab.id_to_token(i)!r} != {v.id_to_token(i)!r}"
        )


# ----------------------------------------------------------------------
# Hyperparams round-trip
# ----------------------------------------------------------------------

@pytest.mark.parametrize("method,idf_method,k1,b,delta", [
    ("lucene", "lucene", 1.5, 0.75, 0.5),
    ("atire", "atire", 1.2, 0.8, 0.0),
    ("bm25l", "bm25l", 2.0, 0.6, 0.7),
    ("bm25+", "bm25+", 1.7, 0.5, 1.0),
    ("robertson", "lucene", 1.5, 0.75, 0.25),  # mixed
])
def test_hyperparams_round_trip(tmp_path, method, idf_method, k1, b, delta):
    from mojo_bm25s import save_index, load_index

    vocab, fields = _build_small_index(
        method=method, idf_method=idf_method,
        k1=k1, b=b, delta=delta,
    )
    target = tmp_path / "idx"
    save_index(target, vocab=vocab, **fields)

    loaded = load_index(target)
    assert loaded.method == method
    assert loaded.idf_method == idf_method
    assert loaded.k1 == pytest.approx(k1)
    assert loaded.b == pytest.approx(b)
    assert loaded.delta == pytest.approx(delta)


# ----------------------------------------------------------------------
# meta.json contents — locked-in surface for downstream tooling
# ----------------------------------------------------------------------

def test_meta_json_contains_documented_fields(tmp_path):
    """The issue spec locks the meta.json schema. Pin it."""
    from mojo_bm25s import save_index

    vocab, fields = _build_small_index(method="lucene")
    target = tmp_path / "idx"
    save_index(target, vocab=vocab, **fields)

    meta = json.loads((target / "meta.json").read_text(encoding="utf-8"))
    for key in ("version", "method", "idf_method", "k1", "b", "delta",
                "n_docs", "n_vocab", "l_avg", "dtype"):
        assert key in meta, f"meta.json should contain key {key!r}; got {sorted(meta)}"
    assert meta["dtype"] == "float32"
    assert meta["n_vocab"] == len(vocab)


# ----------------------------------------------------------------------
# load_index on missing dir / partial dir
# ----------------------------------------------------------------------

def test_load_on_missing_dir_raises(tmp_path):
    from mojo_bm25s import load_index

    with pytest.raises((FileNotFoundError, OSError)):
        load_index(tmp_path / "does-not-exist")


def test_load_on_partial_dir_raises(tmp_path):
    """If meta.json is present but a required binary file is missing,
    loading must fail rather than silently produce a half-index."""
    from mojo_bm25s import save_index, load_index

    vocab, fields = _build_small_index(method="lucene")
    target = tmp_path / "idx"
    save_index(target, vocab=vocab, **fields)

    # Delete one of the required arrays.
    (target / "data.bin").unlink()

    with pytest.raises((FileNotFoundError, OSError)):
        load_index(target)
