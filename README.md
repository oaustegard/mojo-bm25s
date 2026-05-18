# mojo-bm25s

[![CI](https://github.com/oaustegard/mojo-bm25s/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/oaustegard/mojo-bm25s/actions/workflows/ci.yml)

Mojo-native BM25 sparse retrieval. Inspired by [`xhluca/bm25s`](https://github.com/xhluca/bm25s) and by the PostgreSQL extension [`Intelligent-Internet/psql_bm25s`](https://github.com/Intelligent-Internet/psql_bm25s) — borrowing the eager-sparse-scoring idea, not the implementation. **No source code from `bm25s` or `psql_bm25s` is vendored or ported.**

## Status: Phase 1 complete — Phase 2 on hold (see [PHASE2.md](PHASE2.md))

Two-phase project:

- **Phase 1 — kernel-only (done).** Mojo implements the hot loops only — scoring, top-k, CSC column slice. Callable from Python. Drop-in backend for `bm25s.BM25.retrieve()` via `mojo_bm25s.patch_bm25s`. Parity-tested against `bm25s` and `rank_bm25`; benchmarked vs `bm25s`+numpy and `bm25s`+numba.
- **Phase 2 — full standalone library (held).** Tokenizer, stemmer, persistence, CLI. Pre-registered trigger (Mojo ≥ 1.3× numba on a BEIR dataset, OR within 10% on ARM/Apple Silicon) is not met from the x86 numbers; ARM numbers are missing. Decision and re-trigger paths live in [PHASE2.md](PHASE2.md).

## What it will look like (Phase 1)

```python
import bm25s
import mojo_bm25s

retriever = bm25s.BM25()
retriever.index(corpus_tokens)

mojo_bm25s.patch_bm25s(retriever)  # routes hot loops through Mojo kernels

results, scores = retriever.retrieve(query_tokens, k=10)  # identical results, faster
```

## Install

Linux x86-64 wheels are published to PyPI:

```bash
pip install mojo_bm25s
```

The wheel bundles the compiled Mojo kernel — no Mojo toolchain
required on the install machine. The kernel is built with
`--target-cpu=x86-64-v3` (AVX2/FMA baseline) for portability across
modern x86 hosts. macOS / Apple-Silicon wheels are a planned follow-up.

For development from source you still need [`pixi`](https://pixi.sh)
and the Modular conda channel — see *Build from source* below.

## Build from source

```bash
pixi run build            # mojo build → build/mojo_bm25s.so
pixi run test             # rebuilds, then runs pytest
python scripts/build_wheel.py --out dist   # produces dist/mojo_bm25s-*.whl
```

`scripts/build_wheel.py` orchestrates `mojo build` → copy
`build/mojo_bm25s.so` into `src/mojo_bm25s/_kernel.so` →
`python -m build`. The runtime loader (in
`src/mojo_bm25s/__init__.py`) prefers the bundled `_kernel.so` when
available and falls back to `build/mojo_bm25s.so` for in-tree dev.

Tag-driven releases publish via `.github/workflows/release.yml` to
PyPI using OIDC trusted-publisher auth.

## Layout

```
src/mojo_bm25s/   # Mojo kernels + Python interop shim
tests/            # parity tests vs bm25s/rank_bm25
benchmarks/       # head-to-head vs bm25s+numpy and bm25s+numba
scripts/          # build_wheel.py — wheel orchestrator
```

## See also

- [`bm25s`](https://github.com/xhluca/bm25s) — the Python reference, source of the scoring math
- [`psql_bm25s`](https://github.com/Intelligent-Internet/psql_bm25s) — the substrate-shift precedent: Postgres-native instead of Python-native

## License

MIT.
