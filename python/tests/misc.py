import os
import unittest

import rascaline


class TestCMakePrefixPath(unittest.TestCase):
    def test_cmake_prefix_path_exists(self):
        self.assertTrue(hasattr(rascaline._c_lib, "cmake_prefix_path"))
        self.assertTrue(isinstance(rascaline._c_lib.cmake_prefix_path, str))

    def test_cmake_files_exists(self):
        cmake = os.path.join(rascaline._c_lib.cmake_prefix_path, "rascaline")
        self.assertTrue(os.path.isfile(os.path.join(cmake, "rascaline-config.cmake")))
        self.assertTrue(
            os.path.isfile(os.path.join(cmake, "rascaline-config-version.cmake"))
        )


if __name__ == "__main__":
    unittest.main()
