"""Minimal setup.py to mark the wheel as platform-specific.

We bundle a compiled Mojo shared library (`_kernel.so`) into the
wheel as package data. setuptools doesn't see any C/C++ Extension
objects (Mojo is built by its own toolchain outside setuptools), so
without this shim the wheel would be tagged `py3-none-any` and
installable on any platform — including ones the .so won't run on.

Overriding `has_ext_modules` to True forces bdist_wheel to emit a
platform-specific tag (e.g. `cp311-cp311-linux_x86_64`), which makes
PyPI reject install attempts from incompatible platforms instead of
letting the user pip-install a wheel that segfaults on first import.

All other build config lives in pyproject.toml.
"""

from setuptools import Distribution, setup


class BinaryDistribution(Distribution):
    """Tell setuptools this distribution contains compiled artifacts.

    Forces bdist_wheel to drop the `-py3-none-any` purelib tag in favor
    of `-<pyver>-<abi>-<plat>`.
    """

    def has_ext_modules(self) -> bool:  # type: ignore[override]
        return True


setup(distclass=BinaryDistribution)
