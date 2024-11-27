import importlib
import os
import sys
from typing import Any

import metatensor.torch
import torch

import featomic.utils

from .calculator_base import CalculatorModule
from .system import System


# For details what is happening here take a look an `featomic.torch.calculators`.

# create the `_backend` module as an empty module
spec = importlib.util.spec_from_loader(
    "featomic.torch.clebsch_gordan._backend",
    loader=None,
)
module = importlib.util.module_from_spec(spec)
# This module only exposes a handful of things, defined here. Any changes here MUST also
# be made to the `featomic/clebsch_gordan/_backend.py` file, which is used in
# non-TorchScript mode.
module.__dict__["BACKEND_IS_METATENSOR_TORCH"] = True

module.__dict__["Labels"] = metatensor.torch.Labels
module.__dict__["TensorBlock"] = metatensor.torch.TensorBlock
module.__dict__["TensorMap"] = metatensor.torch.TensorMap
module.__dict__["LabelsEntry"] = metatensor.torch.LabelsEntry

module.__dict__["CalculatorBase"] = CalculatorModule
module.__dict__["IntoSystem"] = System

module.__dict__["TorchTensor"] = torch.Tensor
module.__dict__["TorchModule"] = torch.nn.Module
module.__dict__["TorchScriptClass"] = torch.ScriptClass
module.__dict__["Array"] = torch.Tensor
module.__dict__["DType"] = torch.dtype
module.__dict__["Device"] = torch.device

module.__dict__["torch_jit_is_scripting"] = torch.jit.is_scripting
module.__dict__["torch_jit_export"] = torch.jit.export

if os.environ.get("METATENSOR_IMPORT_FOR_SPHINX", "0") == "0":
    module.__dict__["operations"] = metatensor.torch.operations
else:
    # FIXME: we can remove this hack once metatensor-operations v0.2.4 is released
    module.__dict__["operations"] = None


def is_labels(obj: Any):
    return isinstance(obj, metatensor.torch.Labels)


if os.environ.get("FEATOMIC_IMPORT_FOR_SPHINX") is None:
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

# create a module named `featomic.torch.clebsch_gordan` using code from
# `featomic.clebsch_gordan`
spec = importlib.util.spec_from_file_location(
    "featomic.torch.clebsch_gordan", featomic.clebsch_gordan.__file__
)

module = importlib.util.module_from_spec(spec)

# override `featomic.torch.clebsch_gordan` (the module associated with the current file)
# with the newly created module
sys.modules[spec.name] = module
spec.loader.exec_module(module)
