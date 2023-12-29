from typing import Any, Union

import numpy as np
from metatensor import Labels, LabelsEntry, TensorBlock, TensorMap


def torch_jit_is_scripting():
    return False


def torch_jit_annotate(annotation, obj):
    return obj


def is_labels(obj: Any):
    return isinstance(obj, Labels)


check_isinstance = isinstance

try:
    from torch import Tensor as TorchTensor
except ImportError:

    class TorchTensor:
        pass


Array = Union[np.ndarray, TorchTensor]

__all__ = [
    "Labels",
    "TensorBlock",
    "TensorMap",
    "LabelsEntry",
    "torch_jit_is_scripting",
    "torch_jit_annotate",
    "check_isinstance",
    "is_labels",
]
