"""Binary index persistence (issue #26).

Save / load the full BM25 index — vocab + CSC matrix + hyperparams — to
a stable on-disk format. Used by deployments where indexing happens once
and ``retrieve_batch`` runs against the loaded arrays many times.

Format
------

One directory per index, simple files::

    index_dir/
        meta.json          # version, hyperparams, n_docs, n_vocab, l_avg, dtype
        vocab.bin          # length-prefixed UTF-8 strings, ID order
        data.bin           # float32 raw little-endian
        indices.bin        # int32 raw little-endian
        indptr.bin         # int32 raw little-endian
        nonoccurrence.bin  # float32 raw, present iff method in {"bm25l", "bm25+"}

The numeric files use ``np.ndarray.tofile`` / ``np.fromfile`` — raw
little-endian bytes, no header. Sizes are recovered from ``meta.json``
(``n_vocab + 1`` for indptr; data and indices share ``nnz = indptr[-1]``;
nonoccurrence has shape ``(n_vocab,)``).

``vocab.bin`` layout (in detail, so a non-Python reader can parse it):

- 4 bytes little-endian unsigned int: ``n`` = number of tokens
- For each token in ID order (i = 0 .. n-1):
    - 4 bytes little-endian unsigned int: ``L_i`` = UTF-8 byte length
    - ``L_i`` bytes: the token's UTF-8 encoded bytes

There is no terminating sentinel; the next token's length prefix follows
immediately after the previous token's bytes.

API
---

API shape: free functions ``save_index`` / ``load_index``, with
``load_index`` returning a frozen ``LoadedIndex`` dataclass for typed
field access. Class-based ``Retriever`` arrives in #27 — this module
stays orthogonal so #27 can layer over it without churn.

Atomic write
------------

``save_index`` writes everything into ``<index_dir>.tmp/``, then calls
``os.replace`` to atomically rename the staging directory into place. A
crash before the rename leaves no partial ``<index_dir>/`` — the loader
sees nothing and raises ``FileNotFoundError``. If the staging directory
exists from a previous crashed attempt, it is cleared first.

Forward-compat
--------------

``meta.json`` carries a numeric ``version`` field; ``load_index`` rejects
any ``version`` greater than the current ``_FORMAT_VERSION`` with a clear
error message. Existing-or-older versions are accepted.

Vocab and the deprecated vocab.json
-----------------------------------

``Vocab.save`` / ``Vocab.load`` (vocab.json) are kept side-by-side as
the standalone vocab persistence path — they predate this issue and
``tests/test_vocab.py`` still locks their contract. The full-index
persistence here writes ``vocab.bin`` directly (does NOT call
``Vocab.save``) per the issue #26 spec. The two paths are intentionally
parallel; #27's ``Retriever`` will use ``save_index`` / ``load_index``
and the standalone vocab path will likely fade once nothing imports it.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .vocab import Vocab


# The version of the on-disk format. Bumped if we change the layout in
# a way that older loaders can't parse. ``load_index`` rejects anything
# greater than this.
_FORMAT_VERSION = 1

# All numeric arrays are written as little-endian float32 / int32. Lock
# the dtypes so a different-endian machine catches the mismatch on load.
_DATA_DTYPE = np.dtype("<f4")     # little-endian float32
_INDEX_DTYPE = np.dtype("<i4")    # little-endian int32

# File names — locked by the issue spec.
_META_FILE = "meta.json"
_VOCAB_FILE = "vocab.bin"
_DATA_FILE = "data.bin"
_INDICES_FILE = "indices.bin"
_INDPTR_FILE = "indptr.bin"
_NONOCCURRENCE_FILE = "nonoccurrence.bin"

# Methods whose stored data was offset by a per-token nonoccurrence
# scalar at index time. Mirror of index_builder._METHODS_REQUIRING_NONOCCURRENCE.
_METHODS_WITH_NONOCCURRENCE = ("bm25l", "bm25+")


@dataclass(frozen=True)
class LoadedIndex:
    """Result of ``load_index``. All array fields are numpy arrays with
    the documented dtypes; ``nonoccurrence`` is ``None`` unless the
    saved index used a method that requires it.

    ``impact_ordered`` (issue #35) is False for the historical (default)
    doc-id-ordered layout, True for indexes produced by
    ``build_impact_ordered_index`` and saved via ``save_index(...,
    impact_ordered=True)``. Existing indexes without the flag in
    meta.json load as False — backward-compat by absence.
    """
    data: np.ndarray            # float32, shape (nnz,)
    indices: np.ndarray         # int32, shape (nnz,)
    indptr: np.ndarray          # int32, shape (n_vocab + 1,)
    n_docs: int
    l_avg: float
    vocab: Vocab
    method: str
    idf_method: str
    k1: float
    b: float
    delta: float
    nonoccurrence: Optional[np.ndarray] = None  # float32 (n_vocab,) or None
    impact_ordered: bool = False                # issue #35; default False


# ----------------------------------------------------------------------
# vocab.bin codec
# ----------------------------------------------------------------------

def _write_vocab_bin(path: Path, vocab: Vocab) -> None:
    """Serialize the vocab to ``path`` as length-prefixed UTF-8 strings.

    Layout (matches the module docstring):
        u32_le n
        for i in range(n):
            u32_le L_i
            L_i bytes (UTF-8)
    """
    # Pull tokens in ID order via id_to_token (Vocab's public surface).
    n = len(vocab)
    with open(path, "wb") as f:
        f.write(struct.pack("<I", n))
        for i in range(n):
            tok = vocab.id_to_token(i)
            tok_bytes = tok.encode("utf-8")
            f.write(struct.pack("<I", len(tok_bytes)))
            f.write(tok_bytes)


def _read_vocab_bin(path: Path) -> Vocab:
    """Inverse of ``_write_vocab_bin``. Raises if the file is malformed."""
    with open(path, "rb") as f:
        raw = f.read()
    if len(raw) < 4:
        raise ValueError(f"vocab.bin too short to contain a header: {path}")
    (n,) = struct.unpack_from("<I", raw, 0)
    offset = 4
    tokens: list[str] = []
    for _ in range(n):
        if offset + 4 > len(raw):
            raise ValueError(
                f"vocab.bin truncated: needed 4 bytes for length at "
                f"offset {offset}, file is {len(raw)} bytes"
            )
        (L,) = struct.unpack_from("<I", raw, offset)
        offset += 4
        if offset + L > len(raw):
            raise ValueError(
                f"vocab.bin truncated: needed {L} token bytes at offset "
                f"{offset}, file is {len(raw)} bytes"
            )
        tokens.append(raw[offset:offset + L].decode("utf-8"))
        offset += L
    if offset != len(raw):
        raise ValueError(
            f"vocab.bin has {len(raw) - offset} trailing bytes after the "
            f"declared {n} tokens — file is malformed"
        )
    # Reconstruct via the public-ish private slots (mirroring Vocab.load).
    v = Vocab()
    v._tokens = tokens
    v._id_of = {t: i for i, t in enumerate(tokens)}
    return v


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def save_index(
    index_dir: Union[str, Path],
    *,
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    n_docs: int,
    l_avg: float,
    vocab: Vocab,
    k1: float,
    b: float,
    delta: float,
    method: str,
    idf_method: str,
    nonoccurrence: Optional[np.ndarray] = None,
    impact_ordered: bool = False,
) -> None:
    """Persist a built index to ``index_dir`` atomically.

    Writes everything into ``<index_dir>.tmp/`` first, then ``os.replace``s
    the staging directory to ``index_dir``. If the staging directory
    already exists (from a previous crashed attempt), it is cleared
    first. After a successful call, no ``.tmp`` directory remains.

    The numeric arrays are written as raw little-endian bytes; their
    sizes are recoverable from ``meta.json``. See module docstring for
    the full on-disk layout.
    """
    index_dir = Path(index_dir)
    staging = index_dir.with_name(index_dir.name + ".tmp")

    # Clear any leftover staging from a previous crashed attempt. This
    # is the conservative choice — if a partial staging dir exists we
    # have no good way to validate its contents, so we drop and rebuild.
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    # --- meta.json -------------------------------------------------------
    # ``impact_ordered`` (issue #35) only written when True so existing
    # readers/writers that don't know about the flag continue to produce
    # byte-identical meta.json. Loaders default missing → False.
    meta = {
        "version": _FORMAT_VERSION,
        "method": str(method),
        "idf_method": str(idf_method),
        "k1": float(k1),
        "b": float(b),
        "delta": float(delta),
        "n_docs": int(n_docs),
        "n_vocab": int(len(vocab)),
        "l_avg": float(l_avg),
        "dtype": "float32",
    }
    if impact_ordered:
        meta["impact_ordered"] = True
    (staging / _META_FILE).write_text(
        json.dumps(meta, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    # --- vocab.bin -------------------------------------------------------
    _write_vocab_bin(staging / _VOCAB_FILE, vocab)

    # --- data / indices / indptr ----------------------------------------
    # Coerce to the documented little-endian dtypes before .tofile, so
    # the bytes on disk are stable across host endianness.
    np.ascontiguousarray(data, dtype=_DATA_DTYPE).tofile(staging / _DATA_FILE)
    np.ascontiguousarray(indices, dtype=_INDEX_DTYPE).tofile(staging / _INDICES_FILE)
    np.ascontiguousarray(indptr, dtype=_INDEX_DTYPE).tofile(staging / _INDPTR_FILE)

    # --- nonoccurrence (conditional) ------------------------------------
    if method in _METHODS_WITH_NONOCCURRENCE:
        if nonoccurrence is None:
            raise ValueError(
                f"method={method!r} requires a nonoccurrence array, got None"
            )
        np.ascontiguousarray(nonoccurrence, dtype=_DATA_DTYPE).tofile(
            staging / _NONOCCURRENCE_FILE,
        )
    else:
        # Defensive: if the caller passes a nonoccurrence for a method
        # that doesn't use one, drop it (and don't write the file). The
        # on-disk layout is the source of truth: file absent ↔ None.
        if nonoccurrence is not None:
            # No-op; we don't write the file. Document via doc, not error.
            pass

    # --- Atomic finalize ------------------------------------------------
    # If the target already exists, replace it. ``os.replace`` is the
    # POSIX/NT atomic rename. On most filesystems an existing target
    # directory must be empty; we handle that by rmtree'ing first.
    if index_dir.exists():
        shutil.rmtree(index_dir)
    os.replace(staging, index_dir)


def load_index(index_dir: Union[str, Path]) -> LoadedIndex:
    """Load an index previously written by ``save_index``.

    Raises ``FileNotFoundError`` if the directory or any required file
    is absent. Raises ``ValueError`` if ``meta.json``'s ``version`` is
    newer than this loader knows about.
    """
    index_dir = Path(index_dir)
    if not index_dir.is_dir():
        raise FileNotFoundError(
            f"index directory not found: {index_dir}"
        )

    meta_path = index_dir / _META_FILE
    if not meta_path.exists():
        raise FileNotFoundError(
            f"meta.json not found in {index_dir}"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    # --- Version gate --------------------------------------------------
    version = meta.get("version")
    if not isinstance(version, int):
        raise ValueError(
            f"meta.json missing or has non-integer 'version' field "
            f"(got {version!r})"
        )
    if version > _FORMAT_VERSION:
        raise ValueError(
            f"index version {version} is newer than the loader's known "
            f"version {_FORMAT_VERSION}; cannot read forward-incompatible "
            f"format. Upgrade mojo_bm25s."
        )

    # --- Required files ------------------------------------------------
    for required in (_VOCAB_FILE, _DATA_FILE, _INDICES_FILE, _INDPTR_FILE):
        if not (index_dir / required).exists():
            raise FileNotFoundError(
                f"required index file missing: {index_dir / required}"
            )

    method = str(meta["method"])
    idf_method = str(meta["idf_method"])
    n_docs = int(meta["n_docs"])
    n_vocab = int(meta["n_vocab"])
    l_avg = float(meta["l_avg"])
    k1 = float(meta["k1"])
    b = float(meta["b"])
    delta = float(meta["delta"])
    # impact_ordered (issue #35): default False if absent (backward compat).
    impact_ordered = bool(meta.get("impact_ordered", False))

    # --- vocab.bin ----------------------------------------------------
    vocab = _read_vocab_bin(index_dir / _VOCAB_FILE)
    if len(vocab) != n_vocab:
        raise ValueError(
            f"vocab size mismatch: meta.json says n_vocab={n_vocab}, "
            f"vocab.bin has {len(vocab)}"
        )

    # --- indptr -------------------------------------------------------
    indptr = np.fromfile(index_dir / _INDPTR_FILE, dtype=_INDEX_DTYPE)
    if indptr.shape != (n_vocab + 1,):
        raise ValueError(
            f"indptr shape {indptr.shape} does not match expected "
            f"(n_vocab + 1,) = ({n_vocab + 1},)"
        )

    # --- data / indices -----------------------------------------------
    # Both have length nnz = indptr[-1] (or 0 if the array is empty).
    nnz = int(indptr[-1]) if indptr.size else 0
    data = np.fromfile(index_dir / _DATA_FILE, dtype=_DATA_DTYPE)
    indices = np.fromfile(index_dir / _INDICES_FILE, dtype=_INDEX_DTYPE)
    if data.shape != (nnz,):
        raise ValueError(
            f"data shape {data.shape} does not match nnz={nnz} "
            f"derived from indptr[-1]"
        )
    if indices.shape != (nnz,):
        raise ValueError(
            f"indices shape {indices.shape} does not match nnz={nnz} "
            f"derived from indptr[-1]"
        )

    # Cast back to the in-memory native-endian dtypes for downstream
    # consumers (the Mojo kernel expects np.float32 / np.int32 native).
    # On a little-endian host this is a no-op view; on a big-endian host
    # it would byte-swap. Force a copy so the returned arrays own their
    # data and don't keep a memory-mapped reference to the file.
    data = np.ascontiguousarray(data, dtype=np.float32)
    indices = np.ascontiguousarray(indices, dtype=np.int32)
    indptr = np.ascontiguousarray(indptr, dtype=np.int32)

    # --- nonoccurrence (conditional) ----------------------------------
    nonoccurrence_path = index_dir / _NONOCCURRENCE_FILE
    if method in _METHODS_WITH_NONOCCURRENCE:
        if not nonoccurrence_path.exists():
            raise FileNotFoundError(
                f"method={method!r} requires nonoccurrence.bin but file "
                f"is missing: {nonoccurrence_path}"
            )
        nonoccurrence = np.fromfile(nonoccurrence_path, dtype=_DATA_DTYPE)
        if nonoccurrence.shape != (n_vocab,):
            raise ValueError(
                f"nonoccurrence shape {nonoccurrence.shape} does not "
                f"match expected (n_vocab,) = ({n_vocab},)"
            )
        nonoccurrence = np.ascontiguousarray(nonoccurrence, dtype=np.float32)
    else:
        if nonoccurrence_path.exists():
            # Spec: file absent ↔ None. Presence with a non-using method
            # is a strong signal of corruption.
            raise ValueError(
                f"method={method!r} does not use nonoccurrence, but "
                f"nonoccurrence.bin is present at {nonoccurrence_path}"
            )
        nonoccurrence = None

    return LoadedIndex(
        data=data,
        indices=indices,
        indptr=indptr,
        n_docs=n_docs,
        l_avg=l_avg,
        vocab=vocab,
        method=method,
        idf_method=idf_method,
        k1=k1,
        b=b,
        delta=delta,
        nonoccurrence=nonoccurrence,
        impact_ordered=impact_ordered,
    )
