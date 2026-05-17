"""Python shim that loads the Mojo-built kernel from ``build/mojo_bm25s.so``.

Phase 1: only re-exports ``hello()`` so a smoke test can verify the
toolchain is wired up. Real kernel functions land in later issues.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_DIR.parent.parent
_KERNEL_PATH = _REPO_ROOT / "build" / "mojo_bm25s.so"


def _load_kernel():
    if not _KERNEL_PATH.exists():
        raise ImportError(
            f"mojo_bm25s kernel not found at {_KERNEL_PATH}. "
            "Run `pixi run build` first."
        )
    spec = importlib.util.spec_from_file_location(
        "mojo_bm25s.kernel", str(_KERNEL_PATH)
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not create import spec for {_KERNEL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["mojo_bm25s.kernel"] = module
    spec.loader.exec_module(module)
    return module


_kernel = _load_kernel()
hello = _kernel.hello

__all__ = ["hello"]
