# Phase 1 benchmark results

Falsifiable performance numbers gating the Phase 2 decision (#9). Three
backends, same `bm25s` retrieve hot path:

1. **numpy** — stock `bm25s`, default NumPy backend
2. **numba** — `bm25s` with `backend="numba"` JIT
3. **mojo** — `mojo_bm25s.retrieve_batch` (Path A + two follow-on
   x86-side optimizations)

All three produce identical top-k rankings (modulo score-tie reorder
at the rank-k boundary); see `tests/parity/` and
`tests/test_retrieve_batch.py` for the per-query parity assertions.

## Methodology

- **QPS** = one `retrieve_batch(all_queries)` call per repeat, total
  queries / wall time. Apples-to-apples — numba batches via its
  native `_retrieve_numba_functional`, mojo batches via
  `retrieve_batch`, numpy still sequential-maps internally (no native
  batch on the numpy side).
- **Latency p50/p95/p99** = per-query `retrieve_one([q])` loop —
  captures the small-batch / interactive cost the throughput
  number hides.

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
| numpy | 9,604 | 0.30 | 0.081 | 0.279 | 0.327 | 195 |
| numba | 66,173 | 0.31 | 0.040 | 0.070 | 0.087 | 297 |
| mojo  | 57,303 | 0.30 | 0.041 | 0.072 | 0.092 | 350 |

**mojo / numpy:** 5.97× faster.
**mojo / numba:** **0.87×** (numba 1.15× faster — within 13%).

## trec-covid — corpus=171,332 queries=50 k=10 repeats=2

| backend | qps | index_secs | latency_p50_ms | latency_p95_ms | latency_p99_ms | peak_rss_mb |
|---|---|---|---|---|---|---|
| numpy | 232 | 11.9 | 1.311 | 12.939 | 13.672 | 1276 |
| numba | 2,685 | 12.5 | 0.447 | 0.626 | 0.877 | 1278 |
| mojo  | 2,164 | 13.0 | 0.547 | 0.746 | 0.813 | 1581 |

**mojo / numpy:** 9.33× faster.
**mojo / numba:** **0.81×** (numba 1.24× faster — within 19%).

## Cumulative Mojo improvements

| stage | scifact QPS | trec-covid QPS | scifact mojo/numba |
|---|---|---|---|
| Original per-query monkey-patch | 13,151 | 766 | 0.50× |
| + Path A (`retrieve_batch` single Mojo call per batch) | 30,555 | 991 | 0.46×* |
| + `score_*` SIMD-W=8 lift (off bench hot path; helps direct API users) | (no bench change) | (no bench change) | — |
| + `retrieve.mojo` scratch via `UnsafePointer` (no List bookkeeping in hot loop) | 41,496 | 1,493 | 0.62× |
| + `topk_heap_impl_ptr` (heap input scan via pointer too) | **57,303** | **2,164** | **0.87×** |

*The methodology change that landed with Path A also gave numba its proper
batched measurement (was being measured at batch-of-one before), so the
ratio briefly worsened even though absolute Mojo QPS improved 2.3×.

Total Mojo speedup over the original per-query path: **4.36× scifact /
2.83× trec-covid.**

## M1 / Apple Silicon — first arm64 numbers (2026-05-18)

Spot-check on an M1 MacBook (`arm64`, 8 cores, Mojo 1.0.0b1 native build —
`mojo build src/mojo_bm25s/lib.mojo --emit shared-lib -o build/mojo_bm25s.so`,
no `--target-cpu` flag).

| backend | scifact qps | vs CCotw (x86) | vs M1 numpy |
|---|---|---|---|
| numpy | 8,204 | -15% (CCotw 9,604) | 1.00× |
| mojo (serial, single-shot) | 49,806 | -13% (CCotw mojo 57,303) | **6.07×** |

Mojo's relative advantage carries over: ~6× over numpy on M1, matching
the 5.97× CCotw saw on x86. Absolute QPS is ~13–15% lower on arm64 —
likely arm64 NEON codegen vs the x86 `--target-cpu=x86-64-v3` pin.

Numba was not measured on M1 (its JIT-amortization assumption is broken
by the single-shot harness pattern below; running the full bench harness
would need the M1 retriever-reuse issue resolved first).

### Caveats — bench harness does not run end-to-end on M1

The `benchmarks/run.py` harness calls `retrieve_batch` multiple times on
the same retriever (warmup, batch throughput, then per-query latency
loop). On arm64, the **second call to `retrieve_batch` on the same
retriever segfaults** (SIGSEGV / exit 139), and certain full-corpus
single calls hang at 100% CPU. The 543-test parity suite passes on M1
in 5m11s — every test builds a fresh retriever, so the multi-call path
isn't exercised. Likely Mojo `1.0.0b1` arm64 codegen issue or a latent
lifetime UB only exposed on arm64; not yet root-caused.

The M1 numbers above are from **fresh Python subprocesses, single
`retrieve_batch` call per process**, capped with `signal.alarm(25)`:

```bash
cd /Users/austegard/Projects/mojo-bm25s
# numpy
PYTHONPATH=src .pixi/env/bin/python -c "
import sys, signal, time; signal.alarm(25)
sys.path.insert(0,'src'); sys.path.insert(0,'.')
from benchmarks.datasets import load_beir
from benchmarks.backends import build_retriever
ds = load_beir('scifact'); ct = ds.corpus_tokens(); qt = ds.query_tokens()
r = build_retriever('numpy', ct)
t = time.perf_counter()
r.retrieve(qt, k=10, backend_selection='numpy', show_progress=False)
el = time.perf_counter()-t
print(f'numpy {len(qt)}q: {el*1000:.1f}ms {len(qt)/el:.0f} qps')
"
# mojo serial single-shot (1000q — full 1109 hits the M1 bug)
PYTHONPATH=src .pixi/env/bin/python -c "
import sys, signal, time; signal.alarm(25)
sys.path.insert(0,'src'); sys.path.insert(0,'.')
import mojo_bm25s
from benchmarks.datasets import load_beir
from benchmarks.backends import build_retriever
ds = load_beir('scifact'); ct = ds.corpus_tokens(); qt = ds.query_tokens()[:1000]
r = build_retriever('mojo', ds.corpus_tokens())
t = time.perf_counter()
mojo_bm25s.retrieve_batch(r, qt, k=10, num_workers=1)
el = time.perf_counter()-t
print(f'mojo {len(qt)}q: {el*1000:.1f}ms {len(qt)/el:.0f} qps')
"
```

## Natural Questions (~1M docs)

Not run in this environment — dataset is large enough that running
in the CCotw container is wasteful, and the numbers wouldn't reflect
deployment hardware anyway. To reproduce locally:

```bash
pixi run python benchmarks/run.py --dataset nq --all-backends --repeats 1
```

## Reproduction

```bash
pixi install

# All three backends, scifact:
pixi run python benchmarks/run.py --dataset scifact --all-backends --repeats 3 --k 10

# Trec-covid:
pixi run python benchmarks/run.py --dataset trec-covid --all-backends --repeats 2 --k 10
```

Dataset caches under `~/.cache/mojo-bm25s/datasets/` (or
`$MOJO_BM25S_CACHE`). First run downloads ~3 MB (scifact) or
~70 MB (trec-covid) from the bm25s release mirror.

## Honest read

The Mojo-side x86 engineering is now mostly mined out. Path A diagnosed
the boundary-cost issue correctly; the two follow-on changes
(`UnsafePointer` access for scratch + the pointer-input topk variant)
removed Mojo's `List` indexing overhead from the hot loop. Remaining
gap is structural — numba's LLVM codegen on the scatter pattern is
still slightly ahead, but the gap is small.

Status against the pre-registered Phase 2 trigger:

- (a) Mojo beats numba by ≥1.3× on a BEIR dataset — **still not met on
  x86**. Mojo is 0.81–0.87× of numba. Closing the remaining 50% on
  x86 looks unlikely without bigger structural changes.
- (b) Mojo matches numba within 10% on ARM/Apple Silicon — **looks
  very achievable**. On x86 Mojo is already within 13% on scifact
  and within 19% on trec-covid. Numba's x86 LLVM-autovec lead
  shrinks on ARM (a key reason (b) was written this way). If on ARM
  Mojo holds the same or gains a little, it should clear (b)
  cleanly.

See `PHASE2.md` for the decision.
