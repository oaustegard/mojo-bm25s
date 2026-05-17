"""Vocabulary: deterministic string -> int32 ID mapping (issue #24).

Used by the index builder (#25) to turn token lists into id-list rows
and by the retriever (#27) to map query tokens to ids at query time.

Implementation choice: ``.py``, not ``.mojo``.
The issue says: "Python is probably cleaner for dict operations; Mojo's
Dict still has API rough edges in v1.0.0b1." We took that — building a
vocab is not the hot path (retrieve is), and the only thing this module
does on a per-element basis is a dict lookup, which CPython does
faster than the ergonomic cost of pushing it through Mojo's
``PythonObject`` boundary.

Contract (locked by ``tests/test_vocab.py``):

- **Ordering**: tokens are assigned IDs in **first-occurrence** order
  across the corpus, left-to-right within each document, top-to-bottom
  across the document list. Deterministic and reproducible — does not
  depend on Python's hash randomization (set iteration order in
  ``bm25s.get_unique_tokens`` is not deterministic, which is why we
  diverged from that exact approach).
- **Unknown query tokens** → ``-1`` (configurable via the ``unknown``
  kwarg on ``tokens_to_ids``). The retriever filters these out before
  passing the array to the kernel.
- **dtype**: ``tokens_to_ids`` returns ``np.ndarray[int32]`` to match
  the kernel's int32 token-id expectation.

On-disk format:

    <save_dir>/
        vocab.json     # {"version": 1, "tokens": ["tok0", "tok1", ...]}

The ``tokens`` list is in ID order — ``tokens[i]`` is the token whose
ID is ``i``. JSON keeps the format introspectable and language-portable.
``version`` is bumped if the layout ever changes (issue #26 owns the
full-index binary format separately; this file stays simple).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Union

import numpy as np


_VOCAB_FILENAME = "vocab.json"
_FORMAT_VERSION = 1


class Vocab:
    """Bidirectional token <-> int32 ID map built from a tokenized corpus.

    Use ``Vocab.from_corpus(corpus_tokens)`` to build, ``tokens_to_ids``
    to convert query / doc tokens to ids, ``id_to_token(i)`` for the
    reverse lookup. ``save`` / ``load`` persist to a directory.
    """

    __slots__ = ("_tokens", "_id_of")

    def __init__(self) -> None:
        self._tokens: List[str] = []
        self._id_of: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_corpus(
        cls, corpus_tokens: Iterable[Iterable[str]]
    ) -> "Vocab":
        """Build a Vocab from a tokenized corpus.

        Iterates docs in order, tokens in order within each doc; assigns
        a fresh sequential int32 ID the first time each token is seen.
        Repeated tokens (within or across docs) reuse the existing ID.
        """
        v = cls()
        tokens = v._tokens
        id_of = v._id_of
        for doc in corpus_tokens:
            for tok in doc:
                if tok not in id_of:
                    id_of[tok] = len(tokens)
                    tokens.append(tok)
        return v

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def tokens_to_ids(
        self,
        tokens: Sequence[str],
        unknown: int = -1,
    ) -> np.ndarray:
        """Map a sequence of token strings to an ``int32`` id array.

        Unknown tokens are mapped to ``unknown`` (default -1). The output
        is a fresh contiguous ``np.ndarray`` of dtype ``int32`` and shape
        ``(len(tokens),)``.
        """
        n = len(tokens)
        out = np.empty(n, dtype=np.int32)
        id_of = self._id_of
        unk32 = np.int32(unknown)
        for i, t in enumerate(tokens):
            out[i] = id_of.get(t, unk32)
        return out

    def id_to_token(self, i: int) -> str:
        """Reverse lookup: return the token string whose ID is ``i``.

        Raises ``IndexError`` for out-of-range ids.
        """
        return self._tokens[i]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Persist the vocab to a directory ``path`` (created if absent).

        Writes a single ``vocab.json`` file inside the directory; layout
        is part of the contract — see module docstring.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _FORMAT_VERSION,
            "tokens": self._tokens,
        }
        (path / _VOCAB_FILENAME).write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Vocab":
        """Load a vocab previously written by ``save``.

        Raises ``FileNotFoundError`` if the directory or ``vocab.json``
        is missing.
        """
        path = Path(path)
        vocab_file = path / _VOCAB_FILENAME
        if not vocab_file.exists():
            raise FileNotFoundError(
                f"vocab file not found: {vocab_file}"
            )
        payload = json.loads(vocab_file.read_text(encoding="utf-8"))
        tokens = list(payload["tokens"])
        v = cls()
        v._tokens = tokens
        v._id_of = {t: i for i, t in enumerate(tokens)}
        return v

    # ------------------------------------------------------------------
    # Dunders
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._tokens)

    def __repr__(self) -> str:
        return f"<Vocab n_vocab={len(self._tokens)}>"
