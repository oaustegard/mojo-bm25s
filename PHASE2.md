# Phase 2 decision

## TL;DR — **Hold** (Path A tried; trigger still not met; Path B is the
remaining unknown)

Path A (batch the entire retrieve loop into one Mojo entry point)
shipped in PR #18 and delivered a real 2.3× Mojo speedup on scifact.
But the apples-to-apples re-measurement (numba's batched native path
was being measured at batch-of-one previously, disadvantaging it)
showed numba is even further ahead than the original PHASE2.md
suggested. **Trigger (a) remains decisively not met.** Trigger (b)
remains unmeasured.

## 1. Numbers

From `benchmarks/results.md` (full table + methodology + caveats).
Three backends, same `bm25s` retrieve hot path, parity-verified
against bm25s and `rank_bm25` (see `tests/parity/` and
`tests/test_retrieve_batch.py`).

| dataset | corpus | numpy QPS | numba QPS | mojo QPS | **mojo / numba** |
|---|---|---|---|---|---|
| BEIR scifact | 5,183 | 9,268 | 66,468 | 30,555 | **0.46×** |
| BEIR trec-covid | 171,332 | 224 | 2,796 | 991 | **0.35×** |

Mojo is 3.3–4.4× faster than numpy. Mojo is **2.2–2.8× slower than
numba** on x86.

The pre-Path-A version of this file reported mojo/numba ratios of 0.50
(scifact) and 0.34 (trec-covid); the ratios barely moved because Path A
fixed Mojo's batching while the methodology change also gave numba its
proper batched measurement. Both backends are now measured fairly.

Hardware: CCotw container, Intel Xeon @ 2.10 GHz, 4 cores, Linux
x86_64. **This is not benchmarking hardware** — see Caveats below.

## 2. Pre-registered trigger (verbatim, from #9)

> Ship Phase 2 (the full standalone Mojo library) only if **either**:
> - (a) Mojo backend beats `bm25s+numba` by **≥1.3×** on at least one BEIR dataset, **OR**
> - (b) Mojo backend matches `+numba` within 10% while staying SIMD-portable to ARM/Apple Silicon (where Numba is comparatively weaker)

## 3. Decision: **Hold**

**Trigger (a)** is decisively not met on x86. We tried Path A — the
boundary-cost engineering that the pre-Path-A version of this file
named as the most likely lever — and confirmed the diagnosis (2.3×
Mojo speedup on scifact) without closing the ratio. The remaining
gap is structural to numba's LLVM-autovec lineage on x86, in the
scatter loop specifically. SIMD-vectorizing the scatter doesn't
help much (random writes; cache misses dominate); numba's edge
comes from LLVM's x86-tuned codegen on exactly this loop pattern.

**Trigger (b)** still can't be evaluated — we have no ARM / Apple
Silicon numbers. (b) was written precisely because numba's
LLVM-autovec advantage is x86-specific; on ARM the gap should
narrow, and Mojo's SIMD-portability story becomes the differentiator.
Without the ARM number, we'd be guessing.

A **Go** call right now would still move the goalposts: trigger (a)
plainly fails, trigger (b) is unmeasured. A **No-go** call right now
would still archive on incomplete evidence (the missing ARM number
that (b) was built to capture).

**Hold** preserves both options pending Path B evidence.

## 4. Status of the two paths

### Path A — close trigger (a) on x86

**Tried in PR #18.** Single Mojo kernel `retrieve_batch_into` that
does scatter + topk for an entire query batch in one Python ↔ Mojo
crossing. New Python facade `mojo_bm25s.retrieve_batch(retriever,
queries, k)`. Parity vs the per-query patch verified on a 12-query
synthetic corpus across k ∈ {1, 3, 10}; the existing 25-combo
scifact parity suite continues to pass.

Result: 2.3× Mojo speedup on scifact, 1.3× on trec-covid. **Trigger
(a) not met** — mojo is still 0.35–0.46× of numba on x86. No
further x86-side engineering looks likely to close the remaining gap;
the scatter loop is the bottleneck, and it's where numba's LLVM
codegen specifically excels.

### Path B — measure trigger (b) on ARM

**Not yet tried.** Zero code change required: re-run
`benchmarks/run.py --all-backends` on Apple Silicon or Linux ARM.
If Mojo is within 10% of numba on either BEIR dataset, trigger (b)
is met → Go.

This is now **the only remaining evidence source** that could flip
the decision out of Hold without goalpost movement.

### If Path B also fails

No-go: archive the project per the issue's No-go branch
(`RETROSPECTIVE.md` + README update). Phase 1 was still useful —
the kernels work, the parity tests are solid, the bench harness is
reusable, the boundary-cost diagnosis is documented. Just doesn't
justify the standalone-library scope of Phase 2.

## 5. Caveats on the current numbers

- **CCotw container is not benchmarking hardware.** Shared Xeon,
  variable frequency, no thermal isolation. Absolute QPS numbers
  are noisy by maybe 10–20%; the relative ordering between backends
  is stable across repeated runs but worth re-verifying on real
  hardware.
- **Numba's x86 lead is part of what we're measuring.** Numba uses
  LLVM with full x86 autovectorization on the scatter loop in
  particular. On ARM that lead shrinks meaningfully — exactly why
  trigger (b) exists.
- **The Mojo CSC scatter loop is scalar.** Loop body is one indexed
  read + one indexed write per data element. SIMD scatter (`vpscatterdps`)
  doesn't help much when the write pattern is irregular — cache
  misses dominate. This is a Mojo-vs-LLVM-codegen-on-x86 problem,
  not something a Mojo-side optimization is likely to close.

## 6. Recommended next step

Run Path B. It's free — no new code, just a host with a different
ISA. If it clears (b), Go. If it doesn't, No-go and archive.

The Path A engineering investment has already been made and is
reusable for Phase 2 if it gets built; it isn't wasted either way.
