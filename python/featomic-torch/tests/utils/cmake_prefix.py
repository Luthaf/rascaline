import os

import featomic.torch


def test_cmake_prefix():
    assert os.path.exists(featomic.torch.utils.cmake_prefix_path)
