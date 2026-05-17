"""Parity: ``mojo_bm25s.Retriever`` (standalone) vs. ``patch_bm25s`` on scifact.

The Phase 2 headline claim: the standalone path — tokenize → vocab →
build_index → retrieve, all owned by ``Retriever``, no ``bm25s`` import
required at inference — produces the same retrieve output as the existing
``patch_bm25s`` path within float32 tolerance.

If this passes, Phase 2's "no bm25s for inference" promise holds:
downstream callers can swap to ``mojo_bm25s.Retriever`` without seeing
score drift.

Coverage: parametrize over all 25 (TFC method × IDF method) combos —
same matrix as ``test_vs_bm25s.py`` so we catch a wiring regression in
any path.
"""

from __future__ import annotations

import numpy as np
import pytest

import bm25s

import mojo_bm25s


TFC_METHODS = ["robertson", "lucene", "atire", "bm25l", "bm25+"]
IDF_METHODS = ["robertson", "lucene", "atire", "bm25l", "bm25+"]


@pytest.mark.parametrize("idf_method", IDF_METHODS)
@pytest.mark.parametrize("method", TFC_METHODS)
def test_standalone_matches_patch_bm25s_on_scifact(scifact, method, idf_method):
    """For every (method, idf_method), the standalone Retriever's
    per-query top-k scores agree with the patch_bm25s path within
    ``atol=1e-5`` (float32 accumulation noise).

    Both sides use:
      - The same tokenizer + stemmer (PyStemmer / bm25s tokenize via the
        scifact fixture).
      - The same hyperparameters (defaults: k1=1.5, b=0.75, delta=0.5).
      - The same query strings (re-tokenized internally by each side).
    """
    import Stemmer

    raw_corpus = scifact["raw_corpus"]
    raw_queries = scifact["raw_queries"]
    corpus_tokens = scifact["corpus_tokens"]
    query_tokens = scifact["query_tokens"]

    # Reference: bm25s with the Mojo patch. Same pre-tokenized input
    # bm25s's own test harness uses, so the parity oracle is unchanged.
    ref = bm25s.BM25(method=method, idf_method=idf_method)
    ref.index(corpus_tokens, show_progress=False)
    mojo_bm25s.patch_bm25s(ref)

    # Standalone: own the whole pipeline. Feed the SAME stemmer + 'en'
    # stopwords that the scifact fixture's tokenizer used, so the only
    # difference is "who builds the index?", not "what tokens are in it?".
    stemmer = Stemmer.Stemmer("english").stemWord
    standalone = mojo_bm25s.Retriever(
        method=method, idf_method=idf_method,
        stopwords="en", stemmer=stemmer,
    ).index(raw_corpus)

    # Use up to 20 queries — plenty of signal across methods without
    # multiplying 25 combos × 50 queries × kernel dispatch.
    k = 10
    n_q = min(20, len(raw_queries))
    s_standalone, _ = standalone.retrieve(raw_queries[:n_q], k=k)

    max_dev = 0.0
    for i in range(n_q):
        ids_ref, scores_ref = ref.retrieve(
            [query_tokens[i]], k=k, show_progress=False,
        )
        scores_ref = scores_ref[0]

        # Sorted descending — robust to tie-class permutation between
        # the two paths' top-k picks.
        s_a = np.sort(s_standalone[i])[::-1]
        s_b = np.sort(scores_ref)[::-1]
        dev = float(np.max(np.abs(s_a - s_b)))
        max_dev = max(max_dev, dev)
        np.testing.assert_allclose(
            s_a, s_b, atol=1e-5,
            err_msg=(
                f"method={method} idf={idf_method} q[{i}]: standalone "
                f"scores diverge from patch_bm25s by max {dev:.6g}"
            ),
        )

        # Every standalone-picked score must lie in (or above) the
        # reference's rank-k tie class.
        boundary = float(scores_ref[-1])
        for picked_score in s_standalone[i].tolist():
            assert picked_score + 1e-5 >= boundary, (
                f"method={method} idf={idf_method} q[{i}]: standalone "
                f"picked score {picked_score:.6f} below ref rank-k "
                f"boundary {boundary:.6f}"
            )

    # Surface the headline number (visible with ``pytest -s``).
    print(
        f"\n  method={method:<10s} idf={idf_method:<10s} "
        f"max|delta|={max_dev:.3e}"
    )
