# This is not the actual setup.py for this project, see `python/featomic/setup.py` for
# it. Instead, this file is here to enable `pip install .` from a git checkout or `pip
# install git+https://...` without having to specify a subdirectory

import os

from setuptools import setup


ROOT = os.path.realpath(os.path.dirname(__file__))

setup(
    name="featomic-git",
    version="0.0.0",
    install_requires=[
        f"featomic @ file://{ROOT}/python/featomic",
    ],
    extras_require={
        "torch": [
            f"featomic-torch @ file://{ROOT}/python/featomic-torch",
        ]
    },
    packages=[],
)
