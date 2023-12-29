import importlib
import sys
from typing import List, Optional, Union

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

import rascaline.utils.clebsch_gordan


# For details what is happening here take a look an `rascaline.torch.calculators`.

# Step 1: create te `_classes` module as an empty module
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
