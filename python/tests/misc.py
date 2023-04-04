import os
import unittest

import rascaline


class TestCMakePrefixPath(unittest.TestCase):
    def test_cmake_prefix_path_exists(self):
        self.assertTrue(hasattr(rascaline.utils, "cmake_prefix_path"))
        self.assertTrue(isinstance(rascaline.utils.cmake_prefix_path, str))

        # there is both the path to equistore and rascaline cmake prefix in here
        self.assertEqual(len(rascaline.utils.cmake_prefix_path.split(";")), 2)

    def test_cmake_files_exists(self):
        rascaline_cmake_path = rascaline.utils.cmake_prefix_path.split(";")[0]
        cmake = os.path.join(rascaline_cmake_path, "rascaline")

        self.assertTrue(os.path.isfile(os.path.join(cmake, "rascaline-config.cmake")))
        self.assertTrue(
            os.path.isfile(os.path.join(cmake, "rascaline-config-version.cmake"))
        )


if __name__ == "__main__":
    unittest.main()
