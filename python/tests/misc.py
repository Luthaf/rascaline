import os

import rascaline


def test_cmake_prefix_path_exists():
    assert hasattr(rascaline.utils, "cmake_prefix_path")
    assert isinstance(rascaline.utils.cmake_prefix_path, str)

    # there is both the path to equistore and rascaline cmake prefix in here
    assert len(rascaline.utils.cmake_prefix_path.split(";")), 2


def test_cmake_files_exists():
    rascaline_cmake_path = rascaline.utils.cmake_prefix_path.split(";")[0]
    cmake = os.path.join(rascaline_cmake_path, "rascaline")

    assert os.path.isfile(os.path.join(cmake, "rascaline-config.cmake"))
    assert os.path.isfile(os.path.join(cmake, "rascaline-config-version.cmake"))
