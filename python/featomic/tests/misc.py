import os

import featomic


def test_cmake_prefix_path_exists():
    assert hasattr(featomic.utils, "cmake_prefix_path")
    assert isinstance(featomic.utils.cmake_prefix_path, str)

    # there is both the path to metatensor and featomic cmake prefix in here
    assert len(featomic.utils.cmake_prefix_path.split(";")), 2


def test_cmake_files_exists():
    featomic_cmake_path = featomic.utils.cmake_prefix_path.split(";")[0]
    cmake = os.path.join(featomic_cmake_path, "featomic")

    assert os.path.isfile(os.path.join(cmake, "featomic-config.cmake"))
    assert os.path.isfile(os.path.join(cmake, "featomic-config-version.cmake"))
