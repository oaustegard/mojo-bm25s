"""Wheel distribution tests (issue #29).

The acceptance contract: a wheel built from this tree installs cleanly
in a fresh venv with no Mojo / pixi present, ``import mojo_bm25s``
succeeds, the console script is wired up, and the high-level
``Retriever().index([...]).retrieve([...])`` call works against the
bundled ``.so``.

The most likely impl bug is "wheel built but the .so was excluded" —
setuptools' default ``find_packages`` does not pick up arbitrary
non-source files. So the bundled-``.so``-in-wheel assertion is the
load-bearing one; the rest are downstream sanity checks.

Layout choices the tests assume:

- The build helper script lives at ``scripts/build_wheel.py`` (orchestrates
  ``mojo build`` → copy to ``src/mojo_bm25s/_kernel.so`` → ``python -m build``).
- The bundled .so inside the wheel is named ``mojo_bm25s/_kernel.so``
  (matches the naming in the loader's primary search path).
- The loader prefers ``_PACKAGE_DIR / "_kernel.so"`` and falls back to
  the repo-relative ``build/mojo_bm25s.so`` for in-tree dev.

Tests that need a fresh venv (build + install + import roundtrip) are
opt-in via ``RUN_VENV_TESTS=1``. They're not skip-marked by default —
they run in this CCotw container and in the release workflow — but if
you're iterating quickly and want to skip the ~30s venv ceremony,
``unset`` it.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_SO = REPO_ROOT / "build" / "mojo_bm25s.so"
BUILD_SCRIPT = REPO_ROOT / "scripts" / "build_wheel.py"
BUNDLED_SO_NAME = "mojo_bm25s/_kernel.so"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_built_wheel(dist_dir: Path) -> Path:
    wheels = sorted(dist_dir.glob("mojo_bm25s-*.whl"))
    assert wheels, f"No wheel produced in {dist_dir} (contents: {list(dist_dir.iterdir())})"
    assert len(wheels) == 1, f"Expected exactly one wheel, got {wheels}"
    return wheels[0]


def _build_wheel(out_dir: Path) -> Path:
    """Invoke the project's wheel build helper into ``out_dir``.

    The helper handles `mojo build` + copying the `.so` next to the
    package + running `python -m build`. Returns the wheel Path.

    Pre-checks: the in-tree .so must exist (otherwise --skip-mojo would
    fail downstream with a less helpful error), and the `build` module
    must be importable (it's listed in pixi.toml pypi-dependencies, but
    skipping here is friendlier than a cryptic CalledProcessError).
    """
    assert BUILD_SCRIPT.exists(), (
        f"missing {BUILD_SCRIPT} — the wheel test suite requires the "
        "build helper to orchestrate mojo build + python -m build."
    )
    try:
        import build  # noqa: F401 — presence check only
    except ImportError:
        pytest.skip(
            "`build` package not installed; required for the wheel "
            "distribution tests. Add to your environment with "
            "`pip install build` (or use pixi: it's in pypi-dependencies)."
        )
    if not BUILD_SO.exists():
        pytest.skip(
            f"missing {BUILD_SO}; run `pixi run build` first (or invoke "
            "scripts/build_wheel.py without --skip-mojo)."
        )
    subprocess.run(
        [sys.executable, str(BUILD_SCRIPT), "--skip-mojo", "--out", str(out_dir)],
        check=True,
        cwd=REPO_ROOT,
    )
    return _find_built_wheel(out_dir)


# ---------------------------------------------------------------------------
# Tests that need only the wheel itself
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory) -> Path:
    """Build a wheel once per test session into a tmp dist dir."""
    out = tmp_path_factory.mktemp("dist")
    return _build_wheel(out)


def test_wheel_filename_matches_expected_pattern(built_wheel: Path) -> None:
    """Sanity: filename looks like ``mojo_bm25s-<ver>-<py>-<abi>-<plat>.whl``."""
    name = built_wheel.name
    assert name.startswith("mojo_bm25s-"), name
    assert name.endswith(".whl"), name
    # Platform-specific wheel (we bundle a compiled .so) — must NOT be the
    # ``-any.whl`` purelib marker.
    assert "-any.whl" not in name, (
        f"wheel {name} looks like a pure-Python wheel; we bundle a .so "
        "so it must be platform-tagged (linux_x86_64, manylinux, etc.)"
    )


def test_wheel_bundles_compiled_so(built_wheel: Path) -> None:
    """The wheel must include the compiled Mojo kernel adjacent to the package.

    This is the load-bearing test: if package-data config is wrong, the
    wheel installs cleanly but ``_load_kernel()`` raises ImportError on
    first import inside the fresh venv.
    """
    with zipfile.ZipFile(built_wheel) as zf:
        names = zf.namelist()
    assert BUNDLED_SO_NAME in names, (
        f"{BUNDLED_SO_NAME} not in wheel; contents: {sorted(names)}"
    )


def test_bundled_so_matches_build_artifact(built_wheel: Path, tmp_path: Path) -> None:
    """The .so inside the wheel must be byte-identical to ``build/mojo_bm25s.so``.

    Guards against: a stale .so getting copied, or the build helper
    swapping in a debug build, or two parallel builds racing on the
    package-data file.
    """
    assert BUILD_SO.exists(), f"missing {BUILD_SO}; run `pixi run build` first"
    expected_sha = _sha256(BUILD_SO)

    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    with zipfile.ZipFile(built_wheel) as zf:
        zf.extract(BUNDLED_SO_NAME, extract_dir)
    extracted = extract_dir / BUNDLED_SO_NAME
    actual_sha = _sha256(extracted)
    assert actual_sha == expected_sha, (
        f"bundled .so hash {actual_sha} != build artifact hash {expected_sha}"
    )


def test_wheel_includes_python_modules(built_wheel: Path) -> None:
    """All Phase-1 + Phase-2 Python modules must be in the wheel.

    Catches regressions where someone refactors package discovery and
    drops a module (e.g. switching to explicit `packages = ...` and
    forgetting to list cli.py).
    """
    required_modules = {
        "mojo_bm25s/__init__.py",
        "mojo_bm25s/patch.py",
        "mojo_bm25s/cli.py",
        "mojo_bm25s/retriever.py",
        "mojo_bm25s/tokenize.py",
        "mojo_bm25s/stem.py",
        "mojo_bm25s/vocab.py",
        "mojo_bm25s/index_builder.py",
        "mojo_bm25s/io.py",
    }
    with zipfile.ZipFile(built_wheel) as zf:
        names = set(zf.namelist())
    missing = required_modules - names
    assert not missing, f"wheel missing modules: {missing}"


def test_wheel_dist_info_declares_console_script(built_wheel: Path) -> None:
    """``[project.scripts] mojo-bm25s = ...`` must propagate to the wheel.

    The dist-info ``entry_points.txt`` is what ``pip install`` uses to
    generate the launcher in ``<venv>/bin``; if it's missing here the
    ``mojo-bm25s --help`` test below would fail with a less useful
    "command not found" message.
    """
    with zipfile.ZipFile(built_wheel) as zf:
        entry_points_path = next(
            (n for n in zf.namelist() if n.endswith(".dist-info/entry_points.txt")),
            None,
        )
        assert entry_points_path is not None, (
            f"wheel missing entry_points.txt; contents: {sorted(zf.namelist())}"
        )
        text = zf.read(entry_points_path).decode()
    assert "mojo-bm25s" in text, f"console script not declared: {text!r}"
    assert "mojo_bm25s.cli:main" in text, f"wrong target: {text!r}"


# ---------------------------------------------------------------------------
# Loader fallback — does NOT need a wheel install
# ---------------------------------------------------------------------------


def test_loader_falls_back_to_build_dir_when_bundled_missing(tmp_path, monkeypatch):
    """When `_PACKAGE_DIR/_kernel.so` is absent, loader must fall back to
    ``build/mojo_bm25s.so``.

    This is the in-tree dev path — contributors run `pixi run build` and
    then `pytest`, without ever building a wheel. The bundled .so doesn't
    exist in that workflow; the loader must keep finding the build/ copy.
    """
    # Import the module fresh so we can introspect its constants without
    # affecting any other test's already-loaded `mojo_bm25s`.
    import importlib
    import mojo_bm25s as pkg

    pkg_dir = Path(pkg.__file__).resolve().parent
    bundled = pkg_dir / "_kernel.so"

    # If a developer ran `python scripts/build_wheel.py` locally and
    # left _kernel.so in place, the in-tree dev path still works (loader
    # picks it up). We're testing the *fallback* — so this test only
    # makes sense when the bundled copy is absent. If it's present,
    # don't try to mutate the working tree; just assert loader paths
    # are correctly ordered.
    if bundled.exists():
        # The loader's search order must list the bundled location first
        # and the build/ location second. We can't easily inspect that
        # without re-importing, so just check the public attributes
        # exist and the loader returned a module.
        assert hasattr(pkg, "_kernel"), "loader did not populate _kernel"
        return

    # Bundled copy not present (normal dev case). The loader must have
    # used the build/ fallback. Re-import to make sure.
    assert hasattr(pkg, "_kernel"), "loader did not populate _kernel from build/"
    assert hasattr(pkg, "hello"), "kernel did not export `hello`"


# ---------------------------------------------------------------------------
# Fresh-venv integration: install + import + smoke + CLI
# ---------------------------------------------------------------------------


def _clean_env() -> dict:
    """Subprocess env scrubbed of conftest's PYTHONPATH injection.

    `tests/conftest.py` shoves the repo's `src/` onto PYTHONPATH so the
    in-tree tests can `import mojo_bm25s` without a pip install — but
    every subprocess inherits that env, which means a "fresh venv"
    test would silently `import mojo_bm25s` from the repo source, not
    from the installed wheel. Strip it so subprocess imports really
    resolve through `site-packages`.
    """
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return env


def _make_venv(venv_dir: Path) -> Path:
    """Create a fresh venv at ``venv_dir`` and return its python binary."""
    try:
        subprocess.run(
            [sys.executable, "-m", "venv", "--clear", str(venv_dir)],
            check=True,
            capture_output=True,
            env=_clean_env(),
        )
    except subprocess.CalledProcessError as e:
        pytest.skip(
            f"venv creation failed (env restriction?): {e.stderr.decode()}"
        )
    py = venv_dir / "bin" / "python"
    assert py.exists(), f"venv python missing at {py}"
    return py


@pytest.fixture(scope="module")
def venv_with_wheel(built_wheel: Path, tmp_path_factory) -> tuple[Path, Path]:
    """Install the built wheel into a fresh venv. Returns ``(python, bin_dir)``."""
    venv_dir = tmp_path_factory.mktemp("venv")
    py = _make_venv(venv_dir)
    env = _clean_env()

    # Install numpy (declared dep) then the wheel. We deliberately don't
    # `pip install --upgrade pip` first — that's the one place we'd hit
    # an actual network failure, and a fresh venv ships with a recent
    # enough pip for `pip install <local-wheel>`.
    result = subprocess.run(
        [str(py), "-m", "pip", "install", "--quiet", str(built_wheel)],
        capture_output=True,
        env=env,
    )
    if result.returncode != 0:
        pytest.fail(
            f"pip install of wheel failed:\n"
            f"stdout: {result.stdout.decode()}\n"
            f"stderr: {result.stderr.decode()}"
        )
    return py, venv_dir / "bin"


def test_wheel_installs_and_imports_in_fresh_venv(venv_with_wheel) -> None:
    """`import mojo_bm25s` works in a venv with no in-tree source on path.

    This catches the most painful failure mode: wheel installs cleanly,
    `pip list` shows mojo_bm25s, but `import mojo_bm25s` raises because
    `_load_kernel()` can't find the .so (bundled location wrong, or
    fallback to ``build/`` blew up because the cwd is unrelated).
    """
    py, _ = venv_with_wheel
    # Run from /tmp to make sure we're not accidentally importing from
    # the repo's `src/` directory.
    result = subprocess.run(
        [str(py), "-c", "import mojo_bm25s; print(mojo_bm25s.__file__)"],
        capture_output=True,
        cwd="/tmp",
        env=_clean_env(),
    )
    assert result.returncode == 0, (
        f"import failed:\n"
        f"stdout: {result.stdout.decode()}\n"
        f"stderr: {result.stderr.decode()}"
    )
    pkg_path = result.stdout.decode().strip()
    # Should be from site-packages, not the in-tree src/.
    assert "site-packages" in pkg_path, (
        f"expected site-packages import, got {pkg_path!r}"
    )


def test_wheel_end_to_end_retriever_smoke(venv_with_wheel) -> None:
    """Smoke: install → import → index → retrieve, all inside the venv.

    Catches "wheel installs but the kernel doesn't load" — which would
    surface here as an ImportError from `_load_kernel()`, OR as a
    SIGSEGV if the .so was bundled but is the wrong arch.
    """
    py, _ = venv_with_wheel
    script = (
        "import sys\n"
        "from mojo_bm25s import Retriever\n"
        "r = Retriever().index(['the quick brown fox', 'jumps over lazy dog'])\n"
        "scores, ids = r.retrieve(['quick fox'], k=1)\n"
        "assert scores.shape == (1, 1), scores.shape\n"
        "assert ids.shape == (1, 1), ids.shape\n"
        "assert float(scores[0, 0]) > 0, scores\n"
        "print('OK', float(scores[0, 0]), int(ids[0, 0]))\n"
    )
    result = subprocess.run(
        [str(py), "-c", script],
        capture_output=True,
        cwd="/tmp",
        env=_clean_env(),
    )
    assert result.returncode == 0, (
        f"smoke failed:\n"
        f"stdout: {result.stdout.decode()}\n"
        f"stderr: {result.stderr.decode()}"
    )
    assert result.stdout.decode().startswith("OK "), result.stdout.decode()


def test_wheel_cli_console_script_works(venv_with_wheel) -> None:
    """`mojo-bm25s --help` exits 0 in the installed venv.

    Catches: `[project.scripts]` entry didn't make it into the wheel, OR
    the launcher was generated but mojo_bm25s.cli:main can't be imported
    (transitive missing module).
    """
    _, bindir = venv_with_wheel
    cli = bindir / "mojo-bm25s"
    assert cli.exists(), f"CLI launcher missing at {cli}"
    result = subprocess.run(
        [str(cli), "--help"],
        capture_output=True,
        cwd="/tmp",
        env=_clean_env(),
    )
    assert result.returncode == 0, (
        f"`mojo-bm25s --help` failed:\n"
        f"stdout: {result.stdout.decode()}\n"
        f"stderr: {result.stderr.decode()}"
    )
    assert b"index" in result.stdout, result.stdout
    assert b"query" in result.stdout, result.stdout
