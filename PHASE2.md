# Phase 2 decision

## TL;DR — **Hold** (but trigger (b) just got much more plausible)

Three rounds of x86-side optimization (Path A + two follow-on
List-overhead removals) brought Mojo from 0.50× numba to **0.87×
numba** on scifact and **0.81× numba** on trec-covid. **Trigger (a)
remains not met on x86** (would need another ~50% boost, no clear
lever left). **Trigger (b) is now well within reach**: at 0.81–0.87×
of numba on x86, ARM should put us at parity or ahead, since
numba's LLVM-autovec lead is x86-specific.

## 1. Numbers

From `benchmarks/results.md`. Three backends, parity-verified
against bm25s and `rank_bm25`.

| dataset | corpus | numpy QPS | numba QPS | mojo QPS | **mojo / numba** |
|---|---|---|---|---|---|
| BEIR scifact | 5,183 | 9,604 | 66,173 | 57,303 | **0.87×** |
| BEIR trec-covid | 171,332 | 232 | 2,685 | 2,164 | **0.81×** |

Mojo is 6–9× faster than numpy. Mojo is **1.15–1.24× slower than
numba** on x86 — within 13–19% of trigger (b)'s 10% threshold,
already on hardware that favors numba.

Cumulative Mojo improvement from the original per-query monkey-patch:
**4.36× on scifact / 2.83× on trec-covid.** All four optimization
stages and their incremental impact are documented in
`benchmarks/results.md`.

Hardware: CCotw container, Intel Xeon @ 2.10 GHz, 4 cores, Linux
x86_64. **This is not benchmarking hardware** — see Caveats below.

## 2. Pre-registered trigger (verbatim, from #9)

> Ship Phase 2 (the full standalone Mojo library) only if **either**:
> - (a) Mojo backend beats `bm25s+numba` by **≥1.3×** on at least one BEIR dataset, **OR**
> - (b) Mojo backend matches `+numba` within 10% while staying SIMD-portable to ARM/Apple Silicon (where Numba is comparatively weaker)

## 3. Decision: **Hold** (pending Path B)

**Trigger (a)** is not met on x86 — 0.81×, 0.87× of numba. Path A
plus two follow-on x86 optimizations closed about half the original
gap. The remaining gap is structural to numba's LLVM-autovec
codegen on the scatter pattern; closing the next 50% on x86 looks
unlikely without much bigger structural changes.

**Trigger (b)** is now well-positioned to clear, but still
unmeasured. The new numbers (0.81–0.87× on x86) are within striking
distance of (b)'s 10% threshold, and the x86 → ARM transition
typically narrows numba's lead because LLVM's ARM vectorizer is
less mature than its x86 vectorizer. Concrete prediction: Mojo
should land at parity-to-slightly-ahead of numba on Apple Silicon.

A **Go** call right now still requires actual ARM evidence.
A **No-go** call still archives on incomplete evidence,
particularly given how much the x86 picture improved this round.

**Hold** = wait for the ARM bench. The expected value of that
single measurement is now much higher than when this file was first
written.

## 4. Status of the two paths

### Path A — close trigger (a) on x86 ✓ done + extended

**PR #18 merged + this round.** Three Mojo-side optimizations
shipped:

1. **`retrieve_batch`** — single Mojo crossing per batch (the
   original Path A). +130% scifact, +29% trec-covid.
2. **`score_*` SIMD-W=8 lift** — score_tfc and score_idf_array
   process 8 lanes per iteration via `UnsafePointer.load[width=8]`.
   ~2.5× on those functions in micro-bench; doesn't affect the
   retrieve bench (off the hot path) but mining out a deferred
   improvement we'd promised in PR #11.
3. **`UnsafePointer` scratch in `retrieve.mojo`** — scratch backed
   by `List[Float32]` but accessed via `unsafe_ptr()` in the hot
   loop, plus a new `topk_heap_impl_ptr` variant so the topk
   N-element input scan also reads via pointer. +36% / +51% scifact
   / trec-covid on top of #1.

Net result: trigger (a) still not met, but Mojo is within 13–19%
of numba on x86. The remaining gap is numba's LLVM x86 codegen
advantage on the scatter pattern — not something a Mojo-side
optimization is likely to close.

### Path B — measure trigger (b) on ARM

**Still not tried, and now the highest-value remaining experiment.**
Zero code change: re-run `pixi run python benchmarks/run.py
--all-backends --dataset scifact` (and `trec-covid`) on Apple
Silicon or Linux ARM.

**Concrete prediction:** Mojo lands at 0.95×–1.20× of numba on
ARM (vs 0.81–0.87× on x86), clearing trigger (b).

### If Path B fails

No-go: archive per the issue's No-go branch (`RETROSPECTIVE.md` +
README update). Phase 1's deliverables stay useful — the kernels
work, the parity tests are solid, the bench harness is reusable,
the boundary-cost diagnosis is documented. Just doesn't justify
the standalone-library scope of Phase 2.

## 5. Caveats on the current numbers

- **CCotw container is not benchmarking hardware.** Shared Xeon,
  variable frequency, no thermal isolation. Absolute QPS numbers
  are noisy by maybe 10–20%; the relative ordering between
  backends is stable across repeated runs but worth re-verifying
  on real hardware.
- **Numba's x86 lead is part of what we're measuring.** Numba uses
  LLVM with full x86 autovectorization on the scatter loop in
  particular. On ARM that lead shrinks meaningfully — exactly why
  trigger (b) exists.

## 6. Recommended next step

Run Path B (`benchmarks/run.py --all-backends` on Apple Silicon).
At 0.87× of numba on x86 already, this is now the single
measurement that flips the decision out of Hold.
