import importlib
import os
import sys
from typing import Any

import torch
from metatensor.torch import Labels, LabelsEntry, TensorBlock, TensorMap

import rascaline.utils.clebsch_gordan


# For details what is happening here take a look an `rascaline.torch.calculators`.

# Step 1: create the `_classes` module as an empty module
spec = importlib.util.spec_from_loader(
    "rascaline.torch.utils.clebsch_gordan._classes",
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
module.__dict__["torch_jit_annotate"] = torch.jit.annotate
module.__dict__["torch_jit_export"] = torch.jit.export
module.__dict__["TorchTensor"] = torch.Tensor
module.__dict__["TorchModule"] = torch.nn.Module
module.__dict__["TorchScriptClass"] = torch.ScriptClass
module.__dict__["Array"] = torch.Tensor


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


module.__dict__["check_isinstance"] = check_isinstance

# register the module in sys.modules, so future import find it directly
sys.modules[spec.name] = module


# Step 2: create a module named `rascaline.torch.utils.clebsch_gordan` using code from
# `rascaline.utils.clebsch_gordan`
spec = importlib.util.spec_from_file_location(
    "rascaline.torch.utils.clebsch_gordan",
    rascaline.utils.clebsch_gordan.__file__,
)

module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
