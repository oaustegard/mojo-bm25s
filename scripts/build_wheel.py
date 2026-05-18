#!/usr/bin/env python3
"""Wheel build helper (issue #29).

Orchestrates the wheel build:

    1. `mojo build` (skipped if up-to-date or --skip-mojo passed)
    2. Copy `build/mojo_bm25s.so` → `src/mojo_bm25s/_kernel.so` so
       setuptools' `package-data` picks it up
    3. `python -m build --wheel`

The .so bundling has to happen BEFORE `python -m build`, not as a
build_py hook, because the runtime kernel is built by a separate
toolchain (Mojo) that setuptools knows nothing about. Treating it as
plain package data (file-already-in-tree-when-build-starts) is the
cleanest separation.

Usage::

    python scripts/build_wheel.py              # builds into ./dist/
    python scripts/build_wheel.py --out PATH   # builds into PATH
    python scripts/build_wheel.py --skip-mojo  # reuse existing build/mojo_bm25s.so

After build, the staged ``src/mojo_bm25s/_kernel.so`` is left in place
so re-running ``python -m build`` directly still produces a wheel with
the bundled kernel. To clean up, ``git clean -fx src/mojo_bm25s/``.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
MOJO_SRC = REPO_ROOT / "src" / "mojo_bm25s" / "lib.mojo"
BUILD_SO = REPO_ROOT / "build" / "mojo_bm25s.so"
BUNDLED_SO = REPO_ROOT / "src" / "mojo_bm25s" / "_kernel.so"


def _run_mojo_build() -> None:
    BUILD_SO.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "mojo", "build", str(MOJO_SRC),
        "--emit", "shared-lib",
        "--target-cpu=x86-64-v3",
        "-o", str(BUILD_SO),
    ]
    print(f"[build_wheel] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def _stage_bundled_so() -> None:
    if not BUILD_SO.exists():
        raise SystemExit(
            f"missing {BUILD_SO}; run without --skip-mojo or `pixi run build` first"
        )
    BUNDLED_SO.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BUILD_SO, BUNDLED_SO)
    print(f"[build_wheel] staged {BUILD_SO} -> {BUNDLED_SO}", flush=True)


def _run_python_build(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "build",
        "--wheel",
        "--outdir", str(out_dir),
        str(REPO_ROOT),
    ]
    print(f"[build_wheel] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out", default=str(REPO_ROOT / "dist"),
        help="Wheel output directory (default: ./dist)",
    )
    parser.add_argument(
        "--skip-mojo", action="store_true",
        help="Skip `mojo build`; reuse existing build/mojo_bm25s.so.",
    )
    args = parser.parse_args(argv)

    if not args.skip_mojo:
        _run_mojo_build()
    _stage_bundled_so()
    _run_python_build(Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
