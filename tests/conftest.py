"""Make the in-tree ``src/`` package importable without a pip install."""

import os
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Propagate to subprocesses (the CLI tests launch python -m mojo_bm25s.cli).
_existing = os.environ.get("PYTHONPATH", "")
if str(_SRC) not in _existing.split(os.pathsep):
    os.environ["PYTHONPATH"] = (
        f"{_SRC}{os.pathsep}{_existing}" if _existing else str(_SRC)
    )
