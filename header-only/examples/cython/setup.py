# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from skbuild import setup  # This line replaces 'from setuptools import setup'

setup(
    name="demo",
    version="0.0.1",
    license="MIT",
    packages=["demo"],
    python_requires=">=3.7",
    install_requires=["awkward>=2.0.0", "numpy"],
)
