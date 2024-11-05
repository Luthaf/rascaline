from typing import Any, Union

import numpy as np
from metatensor import (  # noqa F401
    Labels,
    LabelsEntry,
    TensorBlock,
    TensorMap,
    operations,
)

from ..calculator_base import CalculatorBase  # noqa F401
from ..systems import IntoSystem  # noqa F401


def torch_jit_is_scripting():
    return False


def torch_jit_export(func):
    return func


def is_labels(obj: Any):
    return isinstance(obj, Labels)


try:
    from torch import ScriptClass as TorchScriptClass
    from torch import Tensor as TorchTensor
    from torch import device as TorchDevice
    from torch import dtype as TorchDType
    from torch.nn import Module as TorchModule
except ImportError:

    class TorchTensor:
        pass

    class TorchModule:
        def __call__(self, *arg, **kwargs):
            return self.forward(*arg, **kwargs)

    class TorchScriptClass:
        pass

    class TorchDevice:
        pass

    class TorchDType:
        pass


Array = Union[np.ndarray, TorchTensor]
DType = Union[np.dtype, TorchDType]
Device = Union[str, TorchDevice]

BACKEND_IS_METATENSOR_TORCH = False
