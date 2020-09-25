# -*- coding=utf-8 -*-
import os
import sys
import shutil
import subprocess

from setuptools import setup
from setuptools.dist import Distribution
from wheel.bdist_wheel import bdist_wheel
from distutils.command.build_ext import build_ext

ROOT = os.path.realpath(os.path.dirname(__file__))

if sys.version_info < (3, 6):
    sys.exit('Sorry, Python < 3.6 is not supported')


class BinaryDistribution(Distribution):
    """
    This is necessary because otherwise the wheel does not know that
    it contains a compiled module
    """
    def has_ext_modules(self):
        return True


class universal_wheel(bdist_wheel):
    # Workaround until https://github.com/pypa/wheel/issues/185 is resolved
    def get_tag(self):
        tag = bdist_wheel.get_tag(self)
        return ("py3", "none") + tag[2:]


class cargo_ext(build_ext):
    '''
    Build rust code using cargo
    '''
    def run(self):
        if sys.platform.startswith("darwin"):
            dylib = "librascaline.dylib"
        elif sys.platform.startswith("linux"):
            dylib = "librascaline.so"
        elif sys.platform.startswith("win"):
            dylib = "librascaline.dll"
        else:
            raise ImportError("Unknown platform. Please edit this file")

        process = subprocess.Popen(
            ['cargo', 'build', '--release'],
            cwd=ROOT
        )

        status = process.wait()
        if status != 0:
            sys.exit(status)

        dst = os.path.join(self.build_lib, 'rascaline', dylib)
        try:
            os.makedirs(os.path.dirname(dst))
        except OSError:
            pass

        src = os.path.join(ROOT, 'target', 'release', dylib)
        if os.path.isfile(src):
            self.copy_file(src, dst)
        else:
            raise Exception('Failed to build rust code')

setup(
    distclass=BinaryDistribution,
    ext_modules=[],
    cmdclass={
        "build_ext": cargo_ext,
        "bdist_wheel": universal_wheel,
    },
)
