"""Parity vs. ``rank_bm25`` on BEIR scifact.

rank_bm25 implements only three BM25 variants. This suite covers two
of them — Okapi and BM25+. The third (BM25L) is excluded because
plain (unpatched) ``bm25s`` 0.3.9 does not itself match
``rank_bm25`` 0.2.2 on BM25L: their formulas under the same name
diverge (mean diff ~17 on scifact; verified independently of the
Mojo patch). The Mojo patch can't fix an upstream disagreement.

The strong-form claim — Mojo backend == bm25s on all 25
(method, idf) combos — is in `test_vs_bm25s.py`. That implicitly
includes BM25L; whatever scoring bm25s does, Mojo matches it.

Variant mapping (see `bm25s/tests/__init__.py:93-98`):

    rank_bm25.BM25Okapi(epsilon=0.0)  ↔  bm25s(method="atire",  idf_method="robertson")
    rank_bm25.BM25Plus(delta=0.5)     ↔  bm25s(method="bm25+")
"""

from __future__ import annotations

import numpy as np
import pytest

import bm25s
import rank_bm25

import mojo_bm25s


@pytest.fixture(scope="module")
def rank_bm25_corpora(scifact):
    """`rank_bm25` wants the raw token-list (not the bm25s tokenized form)."""
    return scifact["corpus_tokens"], scifact["query_tokens"]


@pytest.mark.parametrize(
    "rank_cls, rank_kwargs, bm25s_kwargs",
    [
        (rank_bm25.BM25Okapi, {"k1": 1.5, "b": 0.75, "epsilon": 0.0},
         {"k1": 1.5, "b": 0.75, "method": "atire", "idf_method": "robertson"}),
        (rank_bm25.BM25Plus, {"k1": 1.5, "b": 0.75, "delta": 0.5},
         {"k1": 1.5, "b": 0.75, "delta": 0.5, "method": "bm25+"}),
    ],
    ids=["okapi", "bm25+"],
)
def test_mojo_patched_matches_rank_bm25(
    rank_bm25_corpora, rank_cls, rank_kwargs, bm25s_kwargs
):
    corpus_tokens, query_tokens = rank_bm25_corpora

    # Reference: rank_bm25.
    rank_retriever = rank_cls(corpus_tokens, **rank_kwargs)

    # Subject: bm25s with the matching config, then Mojo-patched.
    patched = bm25s.BM25(**bm25s_kwargs)
    patched.index(corpus_tokens, show_progress=False)
    mojo_bm25s.patch_bm25s(patched)

    for q in query_tokens:
        rank_scores = rank_retriever.get_scores(q).astype(np.float32)
        mojo_scores = patched.get_scores(q)
        np.testing.assert_allclose(
            mojo_scores, rank_scores, atol=1e-4,
            err_msg=f"variant={rank_cls.__name__} query_len={len(q)}",
        )
