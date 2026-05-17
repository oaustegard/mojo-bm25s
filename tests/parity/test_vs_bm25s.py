"""Parity vs. unpatched bm25s on BEIR scifact, all 25 (method, idf) combos.

The strong-form claim: for every TFC × IDF combination bm25s supports,
the Mojo-patched retriever returns the same per-document scores as the
stock numpy backend within float32 tolerance.

Stronger than the rank_bm25 parity test (which only covers 3 variants
because rank_bm25 only implements 3); this lets us catch a regression
in any of the 25 paths.
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
def test_patched_matches_unpatched_on_scifact(scifact, method, idf_method):
    ct = scifact["corpus_tokens"]
    qt = scifact["query_tokens"]

    ref = bm25s.BM25(method=method, idf_method=idf_method)
    ref.index(ct, show_progress=False)
    patched = bm25s.BM25(method=method, idf_method=idf_method)
    patched.index(ct, show_progress=False)
    mojo_bm25s.patch_bm25s(patched)

    # Per-query score-array equality. get_scores returns the full
    # per-doc score vector — invariant to top-k reordering, so a clean
    # numerical comparison.
    for q in qt:
        scores_ref = ref.get_scores(q)
        scores_got = patched.get_scores(q)
        np.testing.assert_allclose(
            scores_got, scores_ref, atol=1e-5,
            err_msg=f"method={method} idf={idf_method} query_len={len(q)}",
        )
