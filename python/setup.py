"""Wheel tagging.

Two things setuptools gets wrong by default for a package like this one, and both would ship a
wheel that lies about where it runs.

1. It is pure Python -- ctypes, no C extension -- so it works on ANY Python 3. Declaring
   `has_ext_modules` to make the wheel platform-specific also makes setuptools tag it for the exact
   CPython that built it (`cp39-cp39`), which would need one wheel per Python version for no reason.
   `get_tag` puts it back to `py3-none`.

2. The platform tag must describe the LIBRARY, not the interpreter. A CPython built as universal2
   makes setuptools tag the wheel `universal2` even when the dylib inside is arm64 only -- and that
   wheel installs happily on an Intel Mac and fails at import. The build script computes the tag
   from the actual binary and passes `--plat-name`; this only stops setuptools from overriding it.
"""

from setuptools import setup
from setuptools.dist import Distribution

try:
    from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:  # setuptools < 70.1
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


class BinaryDistribution(Distribution):
    """Carries a compiled library, so the wheel is platform-specific."""

    def has_ext_modules(self) -> bool:
        return True

    def is_pure(self) -> bool:
        return False


class bdist_wheel(_bdist_wheel):
    def finalize_options(self) -> None:
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        _, _, plat = super().get_tag()
        return "py3", "none", plat


setup(distclass=BinaryDistribution, cmdclass={"bdist_wheel": bdist_wheel})
