from typing import Any, Union

import numpy as np
from metatensor import Labels, LabelsEntry, TensorBlock, TensorMap

from ..calculator_base import CalculatorBase
from ..systems import IntoSystem


def torch_jit_is_scripting():
    return False


def torch_jit_annotate(annotation, obj):
    return obj


def torch_jit_export(func):
    return func


def is_labels(obj: Any):
    return isinstance(obj, Labels)


try:
    from torch import ScriptClass as TorchScriptClass
    from torch import Tensor as TorchTensor
    from torch.nn import Module as TorchModule
except ImportError:

    class TorchTensor:
        pass

    class TorchModule:

        def __call__(self, *arg, **kwargs):
            return self.forward(*arg, **kwargs)

    class TorchScriptClass:
        pass


Array = Union[np.ndarray, TorchTensor]

__all__ = [
    "Array",
    "CalculatorBase",
    "IntoSystem",
    "Labels",
    "TensorBlock",
    "TensorMap",
    "TorchTensor",
    "TorchModule",
    "TorchScriptClass",
    "LabelsEntry",
    "torch_jit_is_scripting",
    "torch_jit_annotate",
    "torch_jit_export",
    "is_labels",
]
