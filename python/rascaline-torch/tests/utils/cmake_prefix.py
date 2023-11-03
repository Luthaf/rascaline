import os

import rascaline.torch


def test_cmake_prefix():
    assert os.path.exists(rascaline.torch.utils.cmake_prefix_path)
