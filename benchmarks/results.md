# Phase 1 benchmark results

Falsifiable performance numbers gating the Phase 2 decision (#9). Three
backends, same `bm25s` retrieve hot path:

1. **numpy** — stock `bm25s`, default NumPy backend
2. **numba** — `bm25s` with `backend="numba"` JIT
3. **mojo** — `bm25s` with `mojo_bm25s.patch_bm25s` monkey-patched in

All three produce identical top-k rankings (modulo score-tie reorder
at the rank-k boundary); see `tests/parity/` for the per-query
parity assertions that back this claim.

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
| numpy | 10646.42 | 0.3024 | 0.0658 | 0.2441 | 0.2961 | 211.1 |
| numba | 26304.88 | 0.2804 | 0.0341 | 0.0611 | 0.0898 | 312.5 |
| mojo | 13150.73 | 0.3994 | 0.0666 | 0.1158 | 0.1454 | 364.6 |

**mojo / numpy:** 1.24× faster.
**mojo / numba:** 0.50× (numba is 2× faster).

## trec-covid — corpus=171,332 queries=50 k=10 repeats=2

| backend | qps | index_secs | latency_p50_ms | latency_p95_ms | latency_p99_ms | peak_rss_mb |
|---|---|---|---|---|---|---|
| numpy | 247.89 | 13.7947 | 1.0723 | 12.2457 | 12.7936 | 1293.7 |
| numba | 2259.42 | 10.6358 | 0.4266 | 0.6422 | 0.7779 | 1295.9 |
| mojo | 765.92 | 12.5753 | 1.2843 | 1.5259 | 1.7288 | 1590.1 |

**mojo / numpy:** 3.09× faster (gap widens at scale, as expected).
**mojo / numba:** 0.34× (numba pulls further ahead on the larger
corpus — LLVM-JIT'd inlining beats Python-boundary per-call cost).

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

# One backend, one dataset, table output.
pixi run python benchmarks/run.py --dataset scifact --backend mojo --k 10

# All three backends sequentially, scifact, 3 repeats:
pixi run python benchmarks/run.py --dataset scifact --all-backends --repeats 3 --k 10

# Trec-covid:
pixi run python benchmarks/run.py --dataset trec-covid --all-backends --repeats 2 --k 10
```

The dataset loader caches under `~/.cache/mojo-bm25s/datasets/` (or
`$MOJO_BM25S_CACHE`). First run downloads ~3 MB (scifact) or
~70 MB (trec-covid) from the bm25s release mirror.

## Honest read

The pre-registered Phase 2 trigger (`mojo >= 1.3× numba` on at least
one BEIR dataset, **or** match within 10% while staying SIMD-portable
to ARM) **is not met by these CCotw-container numbers**. Mojo's
inner loops are competitive — the gap to numba is dominated by
Python ↔ Mojo boundary crossings (one per `retrieve` call, currently),
not by the SIMD math.

The path to a meaningful Phase 2 case would be batching multiple
queries per Mojo call so the per-call wrapper cost amortizes; or
push the entire `retrieve()` loop (scatter + topk + masking) into
one Mojo entry point. Either is a Phase 1 follow-up issue, not a
Phase 2 commitment.

The Phase 2 decision (#9) should re-run this harness on hardware
where deployment will live, and weigh the trade-off between Mojo's
SIMD portability story and Numba's mature x86 advantage on the
metrics that actually matter for the use case.
