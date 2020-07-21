# -*- coding=utf-8 -*-
import os
import sys
import shutil
import subprocess
from distutils.command.build_ext import build_ext
from wheel.bdist_wheel import bdist_wheel
from setuptools import setup
from setuptools.dist import Distribution

import rascaline

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if sys.platform.startswith("darwin"):
    DYLIB = "librascaline.dylib"
elif sys.platform.startswith("linux"):
    DYLIB = "librascaline.so"
elif sys.platform.startswith("win"):
    DYLIB = "librascaline.dll"
else:
    raise ImportError("Unknown platform. Please edit this file")


with open("requirements.txt", "r") as fd:
    REQUIREMENTS = list(filter(bool, (line.strip() for line in fd)))


class BinaryDistribution(Distribution):
    """
    This is necessary because otherwise the wheel does not know that
    we have non pure information.
    """
    def has_ext_modules(foo):
        return True


class universal_wheel(bdist_wheel):
    # Workaround until https://github.com/pypa/wheel/issues/185 is resolved
    def get_tag(self):
        tag = bdist_wheel.get_tag(self)
        return ("py2.py3", "none") + tag[2:]


class cargo_ext(build_ext):
    '''
    Build rust code using cargo
    '''
    def run(self):
        process = subprocess.Popen(
            ['cargo', 'build', '--release'],
            cwd=ROOT
        )

        status = process.wait()
        if status != 0:
            sys.exit(status)

        dist = os.path.join(ROOT, 'python', self.build_lib, 'rascaline', DYLIB)
        src = os.path.join(ROOT, 'target', 'release', DYLIB)
        if os.path.isfile(src):
            self.copy_file(src, dist)
        else:
            raise Exception('Failed to build rust code')


setup(
    name="rascaline",
    # long_description="",
    long_description_content_type="text/markdown",
    version=rascaline.__version__,
    # author="",
    # author_email="",
    # description="",
    # keywords="",
    # url="",
    packages=["rascaline"],
    zip_safe=False,
    install_requires=REQUIREMENTS,
    classifiers=[
        # TODO
    ],
    distclass=BinaryDistribution,
    ext_modules=[],
    cmdclass={
        "build_ext": cargo_ext,
        "bdist_wheel": universal_wheel,
    },
)
