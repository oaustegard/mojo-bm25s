# Phase 1 benchmark results

Falsifiable performance numbers gating the Phase 2 decision (#9). Three
backends, same `bm25s` retrieve hot path:

1. **numpy** — stock `bm25s`, default NumPy backend
2. **numba** — `bm25s` with `backend="numba"` JIT
3. **mojo** — `mojo_bm25s.retrieve_batch` (Path A — single Mojo call
   per batch)

All three produce identical top-k rankings (modulo score-tie reorder
at the rank-k boundary); see `tests/parity/` and
`tests/test_retrieve_batch.py` for the per-query parity assertions.

## Methodology change vs the pre-Path-A version

The earlier version of this file measured throughput by calling
`r.retrieve([q])` per query and dividing total queries by total wall
time. That measurement disadvantaged numba (which has a native
batched path, `_retrieve_numba_functional`) by forcing single-query
calls through batch-of-one.

The current measurement separates throughput from latency:

- **QPS** = one `retrieve_batch(all_queries)` call per repeat,
  total queries / wall time
- **Latency p50/p95/p99** = per-query `retrieve_one([q])` loop —
  captures the small-batch / interactive cost the throughput
  number hides

Apples-to-apples: numba's batched native path vs mojo's Path A
`retrieve_batch` vs numpy's still-sequential map (no native batch
on the numpy side).

## Hardware

| | |
|---|---|
| Date | 2026-05-17 |
| Machine | Claude Code on the Web container |
| CPU | Intel(R) Xeon(R) @ 2.10 GHz, 4 cores |
| Kernel | Linux 6.18.5 x86_64 |
| Python | 3.11 |
| Mojo | 1.0.0b1 |
| bm25s | 0.3.9 |
| numba | 0.65.1 |

**These numbers are from the CCotw container, not benchmarking
hardware.** Numba's relative advantage is x86 + LLVM-autovec specific;
ARM / Apple Silicon would shift the picture. The canonical Phase 2
decision should re-run on the intended deployment hardware.

## scifact — corpus=5,183 queries=1,109 k=10 repeats=3

| backend | qps | index_secs | latency_p50_ms | latency_p95_ms | latency_p99_ms | peak_rss_mb |
|---|---|---|---|---|---|---|
| numpy | 9,268 | 0.31 | 0.083 | 0.282 | 0.340 | 195 |
| numba | 66,468 | 0.31 | 0.041 | 0.070 | 0.090 | 298 |
| mojo  | 30,555 | 0.31 | 0.056 | 0.089 | 0.107 | 350 |

**mojo / numpy:** 3.30× faster.
**mojo / numba:** 0.46× (numba 2.17× faster).

## trec-covid — corpus=171,332 queries=50 k=10 repeats=2

| backend | qps | index_secs | latency_p50_ms | latency_p95_ms | latency_p99_ms | peak_rss_mb |
|---|---|---|---|---|---|---|
| numpy | 224 | 15.5 | 1.379 | 13.141 | 13.519 | 1276 |
| numba | 2,796 | 11.3 | 0.400 | 0.547 | 0.746 | 1278 |
| mojo  | 991 | 13.4 | 1.166 | 1.441 | 1.874 | 1581 |

**mojo / numpy:** 4.42× faster.
**mojo / numba:** 0.35× (numba 2.82× faster).

## What Path A delivered

Compared to the per-query monkey-patch numbers in the pre-Path-A
version of this file:

| dataset | mojo QPS before Path A | mojo QPS after Path A | speedup |
|---|---|---|---|
| scifact | 13,151 | 30,555 | **2.32×** |
| trec-covid | 766 | 991 | **1.29×** |

The diagnosis held: reducing Python ↔ Mojo crossings from O(n_queries)
to O(1) per batch gave Mojo a real speedup, especially on the smaller
corpus where boundary cost was a larger fraction of the work.

But numba also gained from the apples-to-apples measurement change
(its batched native path was being measured at batch-of-one before,
which defeated its internal batching). With both backends measured
fairly, the relative ratio barely moved.

## Natural Questions (~1M docs)

Not run in this environment — dataset is large enough that running
in the CCotw container is wasteful, and the numbers wouldn't reflect
deployment hardware anyway. To reproduce locally:

```bash
pixi run python benchmarks/run.py --dataset nq --all-backends --repeats 1
```

## Reproduction

```bash
# Install env (Mojo + Python deps via pixi).
pixi install

# All three backends, scifact:
pixi run python benchmarks/run.py --dataset scifact --all-backends --repeats 3 --k 10

# Trec-covid:
pixi run python benchmarks/run.py --dataset trec-covid --all-backends --repeats 2 --k 10

# One backend, one dataset, table output:
pixi run python benchmarks/run.py --dataset scifact --backend mojo --k 10
```

Dataset caches under `~/.cache/mojo-bm25s/datasets/` (or
`$MOJO_BM25S_CACHE`). First run downloads ~3 MB (scifact) or
~70 MB (trec-covid) from the bm25s release mirror.

## Honest read

Path A confirmed the diagnosis (boundary cost was real and addressable),
but the pre-registered Phase 2 trigger remains **not met on x86**:

- (a) Mojo beats numba by ≥1.3× on a BEIR dataset — fails. Mojo
  is 0.35–0.46× of numba.
- (b) Mojo matches numba within 10% on ARM/Apple Silicon — still
  unmeasured.

The remaining gap is structural to numba's LLVM-autovec lineage on
x86 — not something a Path A-style optimization can close. The
scatter loop in `retrieve.mojo` is the inner bottleneck (random
writes into the score scratch buffer); SIMD scatter doesn't help
much when the access pattern is irregular. Numba's LLVM
specializes this loop with x86-tuned codegen, which is exactly
where it picks up its advantage.

Path B (re-run on ARM/Apple Silicon, evaluate trigger (b)) is the
remaining unknown. If on ARM Mojo is within 10% of numba on either
dataset, Phase 2 trigger fires.

See `PHASE2.md` for the decision and the path forward.
