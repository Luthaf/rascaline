# -*- coding: utf-8 -*-
import os
import unittest

import rascaline


class TestCMakePrefixPath(unittest.TestCase):
    def test_cmake_prefix_path_exists(self):
        self.assertTrue(hasattr(rascaline.clib, "cmake_prefix_path"))
        self.assertTrue(isinstance(rascaline.clib.cmake_prefix_path, str))

    def test_cmake_files_exists(self):
        cmake = os.path.join(rascaline.clib.cmake_prefix_path, "rascaline")
        self.assertTrue(os.path.isfile(os.path.join(cmake, "rascaline-config.cmake")))
        self.assertTrue(
            os.path.isfile(os.path.join(cmake, "rascaline-config-version.cmake"))
        )
