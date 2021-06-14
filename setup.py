# -*- coding=utf-8 -*-
import os
import sys

from skbuild import setup
from wheel.bdist_wheel import bdist_wheel

ROOT = os.path.realpath(os.path.dirname(__file__))

if sys.version_info < (3, 6):
    sys.exit("Sorry, Python < 3.6 is not supported")


RASCALINE_BUILD_TYPE = os.environ.get("RASCALINE_BUILD_TYPE", "release")
if RASCALINE_BUILD_TYPE not in ["debug", "release"]:
    raise Exception(
        f"invalid build type passed: '{RASCALINE_BUILD_TYPE}',"
        "expected 'debug' or 'release'"
    )


class universal_wheel(bdist_wheel):
    # Workaround until https://github.com/pypa/wheel/issues/185 is resolved
    def get_tag(self):
        tag = bdist_wheel.get_tag(self)
        return ("py3", "none") + tag[2:]


setup(
    ext_modules=[],
    cmdclass={
        "bdist_wheel": universal_wheel,
    },
    cmake_source_dir="rascaline-c-api",
)
