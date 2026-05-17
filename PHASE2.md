# Phase 2 decision

## TL;DR — **Hold**

Pre-registered trigger is not met from current evidence, but the
current evidence is not yet sufficient to make either a clean Go or
a clean No-go call. Two well-scoped follow-ups would each give
decisive evidence.

## 1. Numbers

From `benchmarks/results.md` (full table + hardware caveats there).
Three backends, same `bm25s` retrieve hot path, parity-verified
against bm25s and `rank_bm25` (see `tests/parity/`).

| dataset | corpus | numpy QPS | numba QPS | mojo QPS | **mojo / numba** |
|---|---|---|---|---|---|
| BEIR scifact | 5,183 | 10,646 | 26,305 | 13,151 | **0.50×** |
| BEIR trec-covid | 171,332 | 248 | 2,259 | 766 | **0.34×** |

Mojo is 1.2–3.1× faster than numpy (gap widens at scale, as
expected). Mojo is 2–3× slower than numba on both datasets.

Hardware: CCotw container, Intel Xeon @ 2.10 GHz, 4 cores, Linux
x86_64. **This is not benchmarking hardware** — see Caveats below.

## 2. Pre-registered trigger (verbatim, from #9)

> Ship Phase 2 (the full standalone Mojo library) only if **either**:
> - (a) Mojo backend beats `bm25s+numba` by **≥1.3×** on at least one BEIR dataset, **OR**
> - (b) Mojo backend matches `+numba` within 10% while staying SIMD-portable to ARM/Apple Silicon (where Numba is comparatively weaker)

## 3. Decision: **Hold**

**Trigger (a)** is definitively not met on x86. The numbers aren't
close — Mojo would need ~3× improvement to reach the 1.3× threshold
even on the dataset where it does best. Better hardware will not
close that gap by itself; the bottleneck (see §5) is the Python ↔
Mojo per-call boundary cost, not the SIMD math.

**Trigger (b)** can't be evaluated yet — we have no ARM / Apple
Silicon numbers. (b) was written precisely because Numba's
LLVM-autovec advantage is x86-specific; on ARM the gap should
narrow, and Mojo's SIMD-portability story becomes the differentiator.
Without the ARM number, we'd be guessing.

A **Go** call right now would be moving the goalposts: trigger (a)
plainly fails, trigger (b) is unmeasured, and the issue explicitly
pre-registered the threshold "so we don't move the goalposts after
seeing numbers." A **No-go** call right now would be premature: it
would commit to closing the project on incomplete evidence
(specifically the missing ARM number that trigger (b) was built to
capture).

**Hold** = neither Go nor No-go yet. Project stays alive; one of
the two paths in §4 produces decisive evidence; we re-decide.

## 4. Paths back to a decision

Either of these would supply the missing evidence and flip the call
to Go or No-go without goalpost movement.

### Path A — close trigger (a) on x86

The current bottleneck isn't the SIMD math, it's that every
`retrieve()` call crosses Python ↔ Mojo at least three times
(`csc_score` → produce score vector, `topk` → select rank-k,
back through bm25s framing for masking/nonoccurrence). Numba's
JIT inlines all of that into one function. The Mojo equivalent
would be batching the entire retrieve loop into one Mojo entry
point: take a list of query token-id arrays, return a `(scores,
ids)` matrix, no Python crossings in between.

Estimate: one focused PR. If the resulting Mojo QPS clears 1.3× of
the numbers in §1 on either dataset, trigger (a) is met → Go.

### Path B — measure trigger (b) on ARM

Re-run `benchmarks/run.py --all-backends` on Apple Silicon or
Linux ARM. If Mojo is within 10% of Numba on either BEIR dataset,
trigger (b) is met → Go.

No additional code needed; just a host with a different ISA.

### If both paths fail

No-go: archive the project per the issue's No-go branch
(`RETROSPECTIVE.md` + README update). Phase 1 was still useful —
the kernels work, the parity tests are solid, the bench harness
is reusable. Just doesn't justify the standalone-library scope of
Phase 2.

## 5. Caveats on the current numbers

- **CCotw container is not benchmarking hardware.** Shared Xeon,
  variable frequency, no thermal isolation. Absolute QPS numbers
  are noisy by maybe 10–20%; the relative ordering between
  backends is stable across repeated runs but worth re-verifying
  on real hardware.
- **Numba's x86 lead is part of what we're measuring.** Numba uses
  LLVM with full x86 autovectorization. On ARM that lead shrinks
  meaningfully — exactly why trigger (b) exists.
- **The Mojo wrappers are scalar at the buffer level.** The
  per-element loop in `score_tfc` and `score_idf_array` runs at
  `simd_width=1`. Lifting to native SIMD width (4 or 8 for
  float32) would give a constant-factor speedup on the *inner*
  math but wouldn't close the boundary-cost gap. Mentioned for
  completeness; it's not the right next lever.

## 6. Recommended next step

Run Path B first — zero code change, immediate evidence on the
trigger that was specifically written to test Mojo's
differentiation. If Path B clears (b), Go. If Path B doesn't,
decide whether the Path-A engineering investment is worth it
based on what the ARM numbers actually look like.
