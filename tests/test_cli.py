"""Tests for the ``mojo-bm25s`` CLI (issue #28).

Contract locked here:

- ``mojo-bm25s --help`` / ``index --help`` / ``query --help`` exit 0 and
  print a usage line (argparse format).
- ``index`` reads a JSONL corpus, builds a Retriever, persists to a dir.
- ``query`` loads the persisted index, reads a JSONL query file, writes a
  results JSONL whose lines are ``{"_id", "doc_ids", "scores"}``.
- ``doc_ids`` are **the original ``_id`` strings** from the corpus, NOT
  the internal int indices. (Most likely impl bug class.)
- End-to-end CLI output matches a direct ``Retriever.retrieve()`` call on
  the same texts, byte-for-byte on floats and ID-for-ID on the top-k.
- Hyperparam flags (``--k1``, ``--b``, ``--method``) are honored — using
  non-defaults changes the scores.
- Clear errors with non-zero exit for missing required args, malformed
  JSONL (missing ``_id`` or ``text``), and ``--index`` pointing nowhere.

The CLI is exercised via ``python -m mojo_bm25s.cli`` rather than the
``mojo-bm25s`` entry point so the tests don't require ``pip install -e .``
to have been run in the test environment.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import mojo_bm25s


# ----------------------------------------------------------------------
# Fixtures: tiny synthetic corpus + queries (10 docs, 3 queries)
# ----------------------------------------------------------------------

SYNTH_CORPUS = [
    {"_id": "doc1", "text": "a cat is a small animal that purrs"},
    {"_id": "doc2", "text": "a dog is a big animal that barks"},
    {"_id": "doc3", "text": "fish swim in the deep blue sea"},
    {"_id": "doc4", "text": "the lazy dog sleeps under the tree"},
    {"_id": "doc5", "text": "the quick brown fox jumps over the lazy dog"},
    {"_id": "doc6", "text": "cats and dogs are common pets"},
    {"_id": "doc7", "text": "birds fly high above the mountains"},
    {"_id": "doc8", "text": "a small kitten plays with a ball of yarn"},
    {"_id": "doc9", "text": "the deep blue ocean hides many fish"},
    {"_id": "doc10", "text": "running fast through the forest"},
]

SYNTH_QUERIES = [
    {"_id": "q1", "text": "lazy dog"},
    {"_id": "q2", "text": "fish in the sea"},
    {"_id": "q3", "text": "small cat"},
]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def _run_cli(*args: str, check: bool = False, **kwargs) -> subprocess.CompletedProcess:
    """Run the CLI via the module form. Returns CompletedProcess."""
    cmd = [sys.executable, "-m", "mojo_bm25s.cli", *args]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=check,
        **kwargs,
    )


# ----------------------------------------------------------------------
# --help surface
# ----------------------------------------------------------------------

def test_top_level_help_exits_zero_and_mentions_subcommands():
    res = _run_cli("--help")
    assert res.returncode == 0, f"--help should exit 0, got {res.returncode}\nstderr: {res.stderr}"
    combined = res.stdout + res.stderr
    assert "usage" in combined.lower(), "argparse --help should print a usage line"
    assert "index" in combined, "top-level --help should list the 'index' subcommand"
    assert "query" in combined, "top-level --help should list the 'query' subcommand"


def test_index_subcommand_help():
    res = _run_cli("index", "--help")
    assert res.returncode == 0
    combined = res.stdout + res.stderr
    assert "usage" in combined.lower()
    assert "--corpus" in combined
    assert "--out" in combined


def test_query_subcommand_help():
    res = _run_cli("query", "--help")
    assert res.returncode == 0
    combined = res.stdout + res.stderr
    assert "usage" in combined.lower()
    assert "--index" in combined
    assert "--queries" in combined
    assert "--out" in combined
    assert "--k" in combined


def test_invoked_with_no_args_exits_nonzero():
    res = _run_cli()
    assert res.returncode != 0


# ----------------------------------------------------------------------
# End-to-end pipeline
# ----------------------------------------------------------------------

def _build_index_via_cli(tmp_path: Path, extra_flags: list[str] | None = None) -> Path:
    corpus_path = tmp_path / "corpus.jsonl"
    index_dir = tmp_path / "idx"
    _write_jsonl(corpus_path, SYNTH_CORPUS)
    args = ["index", "--corpus", str(corpus_path), "--out", str(index_dir)]
    if extra_flags:
        args.extend(extra_flags)
    res = _run_cli(*args)
    assert res.returncode == 0, (
        f"index failed: rc={res.returncode}\n"
        f"stdout: {res.stdout}\nstderr: {res.stderr}"
    )
    assert index_dir.is_dir(), "index dir must be created"
    return index_dir


def _query_via_cli(
    tmp_path: Path,
    index_dir: Path,
    k: int = 5,
    queries: list[dict] | None = None,
) -> list[dict]:
    queries_path = tmp_path / "queries.jsonl"
    out_path = tmp_path / "results.jsonl"
    _write_jsonl(queries_path, queries if queries is not None else SYNTH_QUERIES)
    res = _run_cli(
        "query",
        "--index", str(index_dir),
        "--queries", str(queries_path),
        "--out", str(out_path),
        "--k", str(k),
    )
    assert res.returncode == 0, (
        f"query failed: rc={res.returncode}\n"
        f"stdout: {res.stdout}\nstderr: {res.stderr}"
    )
    assert out_path.is_file(), "results file must be created"
    rows = []
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def test_end_to_end_pipeline_writes_well_shaped_results(tmp_path):
    index_dir = _build_index_via_cli(tmp_path)
    rows = _query_via_cli(tmp_path, index_dir, k=5)

    # One output row per query.
    assert len(rows) == len(SYNTH_QUERIES), \
        f"expected {len(SYNTH_QUERIES)} result rows, got {len(rows)}"

    valid_doc_ids = {d["_id"] for d in SYNTH_CORPUS}
    expected_q_ids = [q["_id"] for q in SYNTH_QUERIES]

    for row, expected_qid in zip(rows, expected_q_ids):
        # --- shape ---------------------------------------------------
        assert set(row.keys()) >= {"_id", "doc_ids", "scores"}, \
            f"result row missing required keys: {row!r}"
        assert row["_id"] == expected_qid, \
            f"_id mismatch: got {row['_id']!r}, expected {expected_qid!r}"
        assert isinstance(row["doc_ids"], list), "doc_ids must be a list"
        assert isinstance(row["scores"], list), "scores must be a list"
        assert len(row["doc_ids"]) == 5, \
            f"k=5 must yield 5 doc_ids per query, got {len(row['doc_ids'])}"
        assert len(row["scores"]) == 5, \
            f"k=5 must yield 5 scores per query, got {len(row['scores'])}"

        # --- doc-id round-trip (the load-bearing assertion) ----------
        # If the impl forgot to map internal int indices back to original
        # _id strings, doc_ids would be ints like [4, 3, ...] not strings.
        for did in row["doc_ids"]:
            assert isinstance(did, str), (
                f"doc_ids entries must be original _id strings, got "
                f"{type(did).__name__}: {did!r}"
            )
            assert did in valid_doc_ids, (
                f"doc_id {did!r} not in corpus; round-trip is broken"
            )

        # --- score types ---------------------------------------------
        for s in row["scores"]:
            assert isinstance(s, (int, float)), \
                f"score must be numeric, got {type(s).__name__}"


def test_relevant_doc_is_top_1_for_obvious_query(tmp_path):
    """Sanity check that scoring isn't reversed/broken."""
    index_dir = _build_index_via_cli(tmp_path)
    rows = _query_via_cli(tmp_path, index_dir, k=3)
    # q1 "lazy dog" → doc4 ("the lazy dog sleeps") or doc5 ("...lazy dog")
    # are the only ones with both "lazy" and "dog".
    q1_row = rows[0]
    assert q1_row["_id"] == "q1"
    assert q1_row["doc_ids"][0] in ("doc4", "doc5"), (
        f"expected doc4 or doc5 as top match for 'lazy dog', got "
        f"{q1_row['doc_ids'][0]!r} (full: {q1_row['doc_ids']})"
    )


# ----------------------------------------------------------------------
# Parity vs direct Retriever
# ----------------------------------------------------------------------

def test_cli_results_match_direct_retriever(tmp_path):
    """The CLI is a thin wrapper; output must match an in-process Retriever
    call on the same text content. This is the strongest parity test:
    identical top-k IDs (mapped to original _ids) and bit-identical scores.
    """
    index_dir = _build_index_via_cli(tmp_path)
    rows = _query_via_cli(tmp_path, index_dir, k=5)

    # Same Retriever, in-process, no CLI/serialization layer.
    r = mojo_bm25s.Retriever().index([d["text"] for d in SYNTH_CORPUS])
    scores, ids = r.retrieve([q["text"] for q in SYNTH_QUERIES], k=5)

    for qi, row in enumerate(rows):
        # Map internal ids back to original _id strings the same way the
        # CLI should: positional index into the corpus list.
        expected_doc_ids = [SYNTH_CORPUS[int(idx)]["_id"] for idx in ids[qi]]
        assert row["doc_ids"] == expected_doc_ids, (
            f"q{qi}: CLI doc_ids do not match direct retriever\n"
            f"CLI:    {row['doc_ids']}\n"
            f"direct: {expected_doc_ids}"
        )
        # Scores: serialized as JSON floats. JSON round-trip preserves
        # float64; the kernel returns float32. We compare via float32
        # cast of the CLI-read score back against the kernel output.
        cli_scores_f32 = np.asarray(row["scores"], dtype=np.float32)
        np.testing.assert_array_equal(
            cli_scores_f32, scores[qi].astype(np.float32),
            err_msg=(
                f"q{qi}: CLI scores do not match direct retriever\n"
                f"CLI:    {row['scores']}\n"
                f"direct: {scores[qi].tolist()}"
            ),
        )


# ----------------------------------------------------------------------
# Hyperparams
# ----------------------------------------------------------------------

def test_k1_flag_changes_scores(tmp_path):
    """Passing a non-default --k1 must change scores compared to defaults."""
    # Default-k1 index.
    default_idx = tmp_path / "default_idx"
    custom_idx = tmp_path / "custom_idx"
    corpus_path = tmp_path / "corpus.jsonl"
    _write_jsonl(corpus_path, SYNTH_CORPUS)

    for idx_dir, k1 in ((default_idx, None), (custom_idx, "3.5")):
        args = ["index", "--corpus", str(corpus_path), "--out", str(idx_dir)]
        if k1 is not None:
            args.extend(["--k1", k1])
        res = _run_cli(*args)
        assert res.returncode == 0, res.stderr

    rows_default = _query_via_cli(tmp_path, default_idx, k=3)
    rows_custom = _query_via_cli(tmp_path, custom_idx, k=3)

    differ_somewhere = False
    for rd, rc in zip(rows_default, rows_custom):
        if rd["scores"] != rc["scores"]:
            differ_somewhere = True
            break
    assert differ_somewhere, (
        "non-default --k1 should change scores at least for one query; "
        "got identical scores everywhere — flag is being ignored"
    )


def test_method_flag_is_honored(tmp_path):
    """--method bm25l uses a different TFC variant; scores differ from
    default (which is 'lucene')."""
    corpus_path = tmp_path / "corpus.jsonl"
    _write_jsonl(corpus_path, SYNTH_CORPUS)

    default_idx = tmp_path / "default_idx"
    bm25l_idx = tmp_path / "bm25l_idx"
    for idx_dir, method in ((default_idx, None), (bm25l_idx, "bm25l")):
        args = ["index", "--corpus", str(corpus_path), "--out", str(idx_dir)]
        if method is not None:
            args.extend(["--method", method])
        res = _run_cli(*args)
        assert res.returncode == 0, res.stderr

    rows_default = _query_via_cli(tmp_path, default_idx, k=3)
    rows_bm25l = _query_via_cli(tmp_path, bm25l_idx, k=3)

    any_diff = any(
        rd["scores"] != rb["scores"]
        for rd, rb in zip(rows_default, rows_bm25l)
    )
    assert any_diff, (
        "non-default --method should change scores; got identical results"
    )


def test_b_flag_is_honored(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    _write_jsonl(corpus_path, SYNTH_CORPUS)

    default_idx = tmp_path / "default_idx"
    custom_idx = tmp_path / "custom_idx"
    for idx_dir, b in ((default_idx, None), (custom_idx, "0.1")):
        args = ["index", "--corpus", str(corpus_path), "--out", str(idx_dir)]
        if b is not None:
            args.extend(["--b", b])
        res = _run_cli(*args)
        assert res.returncode == 0, res.stderr

    rows_default = _query_via_cli(tmp_path, default_idx, k=3)
    rows_custom = _query_via_cli(tmp_path, custom_idx, k=3)
    any_diff = any(
        rd["scores"] != rc["scores"]
        for rd, rc in zip(rows_default, rows_custom)
    )
    assert any_diff, "non-default --b should change scores"


# ----------------------------------------------------------------------
# Error cases
# ----------------------------------------------------------------------

def test_index_missing_corpus_flag_errors(tmp_path):
    res = _run_cli("index", "--out", str(tmp_path / "idx"))
    assert res.returncode != 0
    combined = res.stdout + res.stderr
    assert "corpus" in combined.lower(), \
        f"error should mention 'corpus'; got: {combined!r}"


def test_index_missing_out_flag_errors(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    _write_jsonl(corpus_path, SYNTH_CORPUS)
    res = _run_cli("index", "--corpus", str(corpus_path))
    assert res.returncode != 0


def test_query_missing_out_flag_errors(tmp_path):
    index_dir = _build_index_via_cli(tmp_path)
    queries_path = tmp_path / "queries.jsonl"
    _write_jsonl(queries_path, SYNTH_QUERIES)
    res = _run_cli(
        "query",
        "--index", str(index_dir),
        "--queries", str(queries_path),
    )
    assert res.returncode != 0


def test_index_malformed_jsonl_missing_text_errors(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    _write_jsonl(corpus_path, [{"_id": "doc1", "wrong_field": "..."}])
    index_dir = tmp_path / "idx"
    res = _run_cli(
        "index", "--corpus", str(corpus_path), "--out", str(index_dir),
    )
    assert res.returncode != 0
    combined = (res.stdout + res.stderr).lower()
    assert "text" in combined, f"error should mention 'text': {combined!r}"


def test_index_malformed_jsonl_missing_id_errors(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    _write_jsonl(corpus_path, [{"text": "something here"}])
    index_dir = tmp_path / "idx"
    res = _run_cli(
        "index", "--corpus", str(corpus_path), "--out", str(index_dir),
    )
    assert res.returncode != 0
    combined = (res.stdout + res.stderr).lower()
    assert "_id" in combined or "id" in combined, \
        f"error should mention _id: {combined!r}"


def test_index_malformed_jsonl_not_json_errors(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text("this is not json\n", encoding="utf-8")
    index_dir = tmp_path / "idx"
    res = _run_cli(
        "index", "--corpus", str(corpus_path), "--out", str(index_dir),
    )
    assert res.returncode != 0


def test_query_nonexistent_index_errors(tmp_path):
    queries_path = tmp_path / "queries.jsonl"
    _write_jsonl(queries_path, SYNTH_QUERIES)
    out_path = tmp_path / "results.jsonl"
    res = _run_cli(
        "query",
        "--index", str(tmp_path / "does_not_exist"),
        "--queries", str(queries_path),
        "--out", str(out_path),
    )
    assert res.returncode != 0
    combined = (res.stdout + res.stderr).lower()
    # Error should mention the missing path or "not found"-ish.
    assert (
        "does_not_exist" in combined
        or "not found" in combined
        or "no such" in combined
        or "exist" in combined
    ), f"error should hint at missing index dir: {combined!r}"


def test_query_missing_queries_file_errors(tmp_path):
    index_dir = _build_index_via_cli(tmp_path)
    out_path = tmp_path / "results.jsonl"
    res = _run_cli(
        "query",
        "--index", str(index_dir),
        "--queries", str(tmp_path / "no_such_queries.jsonl"),
        "--out", str(out_path),
    )
    assert res.returncode != 0
