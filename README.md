# mojo-bm25s

Mojo-native BM25 sparse retrieval. Inspired by [`xhluca/bm25s`](https://github.com/xhluca/bm25s) and by the PostgreSQL extension [`Intelligent-Internet/psql_bm25s`](https://github.com/Intelligent-Internet/psql_bm25s) — borrowing the eager-sparse-scoring idea, not the implementation. **No source code from `bm25s` or `psql_bm25s` is vendored or ported.**

## Status: Phase 1 (kernel-only proof)

Two-phase project:

- **Phase 1 — kernel-only (current).** Mojo implements the hot loops only — scoring, top-k, CSC column slice. Callable from Python. Drop-in backend for `bm25s.BM25.retrieve()`. Goal: a falsifiable benchmark vs `bm25s` + Numba.
- **Phase 2 — full standalone library (gated).** Tokenizer, stemmer, persistence, CLI. Filed as a second issue tranche only if Phase 1 numbers justify it. Modeled on how `psql_bm25s` separated from its Python inspiration.

Pre-registered Phase 2 trigger: Mojo backend must beat `bm25s+numba` by ≥1.3× on at least one BEIR dataset, or match `+numba` while staying SIMD-portable to ARM/Apple Silicon.

## What it will look like (Phase 1)

```python
import bm25s
import mojo_bm25s

retriever = bm25s.BM25()
retriever.index(corpus_tokens)

mojo_bm25s.patch_bm25s(retriever)  # routes hot loops through Mojo kernels

results, scores = retriever.retrieve(query_tokens, k=10)  # identical results, faster
```

## Layout

```
src/mojo_bm25s/   # Mojo kernels + Python interop shim
tests/            # parity tests vs bm25s/rank_bm25
benchmarks/       # head-to-head vs bm25s+numpy and bm25s+numba
```

## See also

- [`bm25s`](https://github.com/xhluca/bm25s) — the Python reference, source of the scoring math
- [`psql_bm25s`](https://github.com/Intelligent-Internet/psql_bm25s) — the substrate-shift precedent: Postgres-native instead of Python-native

## License

MIT.
