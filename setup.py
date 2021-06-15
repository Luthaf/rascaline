# -*- coding=utf-8 -*-
import os
import sys

from skbuild import setup
from wheel.bdist_wheel import bdist_wheel

ROOT = os.path.realpath(os.path.dirname(__file__))

if sys.version_info < (3, 6):
    sys.exit("Sorry, Python < 3.6 is not supported")

# do not include chemfiles inside rascaline, instead users should use
# chemfiles python bindings directly
cmake_args = ["-DRASCAL_DISABLE_CHEMFILES=ON"]

RASCALINE_BUILD_TYPE = os.environ.get("RASCALINE_BUILD_TYPE", "release")
if RASCALINE_BUILD_TYPE not in ["debug", "release"]:
    raise Exception(
        f"invalid build type passed: '{RASCALINE_BUILD_TYPE}',"
        "expected 'debug' or 'release'"
    )

cmake_args.append(f"-DCMAKE_BUILD_TYPE={RASCALINE_BUILD_TYPE}")


class universal_wheel(bdist_wheel):
    # When building the wheel, the `wheel` package assumes that if we have a
    # binary extension then we are linking to `libpython.so`; and thus the wheel
    # is only usable with a single python version. This is not the case for
    # here, and the wheel will be compatible with any Python >=3.6. This is
    # tracked in https://github.com/pypa/wheel/issues/185, but until then we
    # manually override the wheel tag.
    def get_tag(self):
        tag = bdist_wheel.get_tag(self)
        # tag[2:] contains the os/arch tags, we want to keep them
        return ("py3", "none") + tag[2:]


setup(
    ext_modules=[],
    cmdclass={
        "bdist_wheel": universal_wheel,
    },
    cmake_source_dir="rascaline-c-api",
    cmake_args=cmake_args,
)
