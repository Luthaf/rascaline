import importlib
import os
import sys
from typing import Any

import torch
from metatensor.torch import Labels, LabelsEntry, TensorBlock, TensorMap

import rascaline.utils

from .calculator_base import CalculatorModule
from .system import System


_HERE = os.path.dirname(__file__)


# For details what is happening here take a look an `rascaline.torch.calculators`.

# create the `_backend` module as an empty module
spec = importlib.util.spec_from_loader(
    "rascaline.torch.utils._backend",
    loader=None,
)
module = importlib.util.module_from_spec(spec)
# This module only exposes a handful of things, defined here. Any changes here MUST also
# be made to the `metatensor/operations/_classes.py` file, which is used in non
# TorchScript mode.
module.__dict__["Labels"] = Labels
module.__dict__["TensorBlock"] = TensorBlock
module.__dict__["TensorMap"] = TensorMap
module.__dict__["LabelsEntry"] = LabelsEntry
module.__dict__["torch_jit_is_scripting"] = torch.jit.is_scripting
module.__dict__["torch_jit_export"] = torch.jit.export
module.__dict__["TorchTensor"] = torch.Tensor
module.__dict__["TorchModule"] = torch.nn.Module
module.__dict__["TorchScriptClass"] = torch.ScriptClass
module.__dict__["Array"] = torch.Tensor
module.__dict__["CalculatorBase"] = CalculatorModule
module.__dict__["IntoSystem"] = System
module.__dict__["BACKEND_IS_METATENSOR_TORCH"] = True


def is_labels(obj: Any):
    return isinstance(obj, Labels)


if os.environ.get("RASCALINE_IMPORT_FOR_SPHINX") is None:
    is_labels = torch.jit.script(is_labels)

module.__dict__["is_labels"] = is_labels


def check_isinstance(obj, ty):
    if isinstance(ty, torch.ScriptClass):
        # This branch is taken when `ty` is a custom class (TensorMap, …). since `ty` is
        # an instance of `torch.ScriptClass` and not a class itself, there is no way to
        # check if obj is an "instance" of this class, so we always return True and hope
        # for the best. Most errors should be caught by the TorchScript compiler anyway.
        return True
    else:
        assert isinstance(ty, type)
        return isinstance(obj, ty)


# register the module in sys.modules, so future import find it directly
sys.modules[spec.name] = module

# create a module named `rascaline.torch.utils` using code from
# `rascaline.utils`
spec = importlib.util.spec_from_file_location(
    "rascaline.torch.utils", rascaline.utils.__file__
)

module = importlib.util.module_from_spec(spec)


cmake_prefix_path = os.path.realpath(os.path.join(_HERE, "..", "lib", "cmake"))
"""
Path containing the CMake configuration files for the underlying C library
"""

module.__dict__["cmake_prefix_path"] = cmake_prefix_path

# override `rascaline.torch.utils` (the module associated with the current file)
# with the newly created module
sys.modules[spec.name] = module
spec.loader.exec_module(module)
