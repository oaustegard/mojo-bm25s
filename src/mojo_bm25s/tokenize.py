"""Tokenizer for the standalone mojo-bm25s library (issue #22).

Whitespace + punctuation split, lowercase, optional stopword filter.
Matches ``bm25s.tokenize(..., stopwords='en', stemmer=None)`` closely
enough for ≥99% per-document token overlap on BEIR scifact.

Implementation choice: ``.py``, not ``.mojo``.
Mojo v1.0.0b1's string ops are still rough (no native ``re``, awkward
unicode handling), and the standalone-library goal is "don't pull
``bm25s.tokenize`` and transitively NLTK/PyStemmer." A 30-line Python
function using ``re`` clears that bar; reaching for Mojo here would add
~200 lines of UTF-8 byte-walking with no measurable speed win (this is
not the hot path — retrieval is).

The parity oracle is bm25s's regex ``(?u)\\b\\w\\w+\\b`` plus its
canonical 33-word English stopwords list, both replicated here so we
have no runtime dependency on ``bm25s.tokenization`` from inside the
library.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Set, Union


# Canonical English stopwords matching ``bm25s.tokenization.STOPWORDS_EN``.
# Sourced from bm25s 0.2.x — keeping a snapshot in-tree means downstream
# users don't pull bm25s just to get this list, which is the whole point
# of the standalone-library work.
ENGLISH_STOPWORDS: frozenset = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by",
    "for", "if", "in", "into", "is", "it",
    "no", "not", "of", "on", "or", "such",
    "that", "the", "their", "then", "there", "these",
    "they", "this", "to", "was", "will", "with",
})


# Mirror sklearn / bm25s default token pattern: 2+ unicode word characters
# bounded by word boundaries. ``re.UNICODE`` is the default in Python 3.
_TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")


StopwordsArg = Union[None, str, Iterable[str]]


def _resolve_stopwords(stopwords: StopwordsArg) -> Set[str]:
    """Coerce the ``stopwords`` argument into a set for O(1) membership.

    - ``None``: no stopwords (passthrough).
    - ``"en"`` / ``"english"``: the baked-in English list.
    - any iterable of strings: used as-is.
    """
    if stopwords is None:
        return set()
    if isinstance(stopwords, str):
        key = stopwords.lower()
        if key in ("en", "english"):
            return set(ENGLISH_STOPWORDS)
        # A single-string-as-stopword is almost certainly a bug; bm25s
        # raises in ``_infer_stopwords`` on unknown language codes. We
        # follow suit to avoid silently treating "the" as an iterable of
        # characters.
        raise ValueError(
            f"unknown stopwords language string {stopwords!r}; "
            "pass None, 'en', or an iterable of stopword strings"
        )
    return set(stopwords)


def tokenize(
    texts: List[str],
    stopwords: StopwordsArg = ENGLISH_STOPWORDS,
    lowercase: bool = True,
) -> List[List[str]]:
    """Tokenize ``texts`` into a list of token lists.

    Parameters
    ----------
    texts:
        A list (or other iterable) of strings to tokenize. An empty
        iterable returns ``[]``; an empty string returns ``[[]]``,
        matching ``bm25s.tokenize``.
    stopwords:
        ``None`` for no filtering, ``"en"`` / ``"english"`` for the
        baked-in English list, or any iterable of stopword strings.
        Defaults to ``ENGLISH_STOPWORDS``.
    lowercase:
        If True (default), lowercase each text before splitting — same
        order of operations as bm25s, so stopword matching happens
        post-lowercase.

    Returns
    -------
    list[list[str]]
        One token list per input text, in input order.
    """
    stopwords_set = _resolve_stopwords(stopwords)

    out: List[List[str]] = []
    findall = _TOKEN_PATTERN.findall
    for text in texts:
        if lowercase:
            text = text.lower()
        tokens = findall(text)
        if stopwords_set:
            tokens = [t for t in tokens if t not in stopwords_set]
        out.append(tokens)
    return out
