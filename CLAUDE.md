# mojo-bm25s — development instructions

Mojo-native BM25 scoring kernels, callable from Python. Parity oracle is
[`xhluca/bm25s`](https://github.com/xhluca/bm25s); numerical results
must match it within `atol=1e-6` on float32.

## TDD is mandatory for this project

**Every code change lands as a failing test first, then the
implementation that turns it green.** No exceptions for "small" kernel
tweaks — the whole point of this project is that the numbers match a
reference, and the only way to know that is a parity assertion that
existed before the code did.

Loop (do not skip steps):

1. Read the issue. Scope ruthlessly — smallest coherent slice.
2. Write `tests/test_*.py` encoding the contract — including the edge
   cases the spec mentions (`df=0`, single-doc corpus, lengths not
   divisible by `simd_width`, etc). Derive the assertions from the
   reference (`bm25s`), not from what the implementation is going to do.
3. Stub `src/mojo_bm25s/<thing>.mojo` so imports succeed but the
   assertions fail. Run `pixi run test` and **confirm RED for the right
   reason** — contract violation, not an ImportError. If the test
   passes against the stub, the test is too weak.
4. Implement.
5. Run `pixi run test` again. Iterate to GREEN. If a test fails after
   the implementation looks right, the bug could be in the test —
   inspect both sides.
6. Commit test and implementation together. PR body tells the
   red→green story.

Skipping the failing-test step bolts tests onto the implementation —
they pass, but they pass because they were written to match what the
code does, not what it should do. Bolt-on tests miss exactly the bug
class that TDD catches: wrong expectations.

## Mojo version

This project targets the Mojo distributed via the Modular conda channel.
The container currently has **v1.0.0b1** (Modular reset version numbering
from the v26.x series). Notable corrections vs. the `coding-mojo` skill:

- Imports: `from std.python import ...`, `from std.os import abort`,
  `from std.python.bindings import PythonModuleBuilder`. Bare
  `from python import ...` is deprecated.
- `def`s that may raise must be marked `raises` or wrap calls in `try`.
- `PyInit_*` is single-phase init; CPython looks up `PyInit_<last
  segment of spec.name>`, not the .so filename basename.

## Build artifact loading

`pixi run build` produces `build/mojo_bm25s.so`. The Python shim in
`src/mojo_bm25s/__init__.py` importlib-loads it as the sub-module
`mojo_bm25s.kernel`, so the Mojo `@export def PyInit_kernel()` resolves
correctly without colliding with the `mojo_bm25s` package name. Add new
Mojo-exported functions to the same `PythonModuleBuilder` registration
and re-export from `__init__.py`.

## Layout

```
src/mojo_bm25s/   # Mojo kernels + Python shim
tests/            # parity tests vs bm25s/rank_bm25 (run via pixi run test)
benchmarks/       # head-to-head, added in later issues
```

## Commands

```bash
pixi run build    # mojo build → build/mojo_bm25s.so
pixi run test     # rebuilds first (declared dependency), then pytest
pixi run bench    # rebuilds first, then python benchmarks/run.py
```

`pixi run test` always rebuilds because the test depends on the build
task. Don't bypass that — running `pytest` directly will use a stale
`.so` and produce confusing greens.
