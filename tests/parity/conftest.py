"""Shared fixtures for the parity test suite.

Downloads + tokenizes BEIR scifact once per pytest session; the cost is
~0.5s after first download. ``benchmarks/`` is added to sys.path so
the parity tests can import the dataset loader without an install.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make `benchmarks/` importable (peer of `src/` and `tests/`).
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture(scope="session")
def scifact():
    """Full BEIR scifact: 5,183 docs / 1,109 queries, already tokenized.

    Queries are trimmed to the first 50 — enough parity signal across
    methods without dragging out CI.
    """
    from benchmarks.datasets import load_beir

    ds = load_beir("scifact", queries_subsample=50)
    return {
        "corpus_tokens": ds.corpus_tokens(),
        "query_tokens": ds.query_tokens(),
        "raw_corpus": ds.corpus,
        "raw_queries": ds.queries[:50],
    }
