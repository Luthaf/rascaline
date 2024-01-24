import importlib
import sys
from typing import List, Optional, Union

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

import rascaline.utils.power_spectrum

from ..calculator_base import CalculatorModule as CalculatorBase
from ..system import System as IntoSystem


# For details what is happening here take a look an `rascaline.torch.calculators`.

# Step 1: create te `_classes` module as an empty module
spec = importlib.util.spec_from_loader(
    "rascaline.torch.utils.power_spectrum._classes",
    loader=None,
)
module = importlib.util.module_from_spec(spec)
# This module only exposes a handful of things, defined here. Any changes here MUST also
# be made to the `metatensor/operations/_classes.py` file, which is used in non
# TorchScript mode.
module.__dict__["Labels"] = Labels
module.__dict__["TensorBlock"] = TensorBlock
module.__dict__["TensorMap"] = TensorMap
module.__dict__["CalculatorBase"] = CalculatorBase
module.__dict__["IntoSystem"] = IntoSystem

# register the module in sys.modules, so future import find it directly
sys.modules[spec.name] = module


# Step 2: create a module named `rascaline.torch.utils.power_spectrum` using code from
# `rascaline.utils.power_spectrum`
spec = importlib.util.spec_from_file_location(
    "rascaline.torch.utils.power_spectrum",
    rascaline.utils.power_spectrum.__file__,
)

module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


# Store the original class to avoid recursion problems
PowerSpectrumBase = module.PowerSpectrum


class PowerSpectrum(torch.nn.Module, PowerSpectrumBase):
    """
    Torch version of the general power spectrum of one or of two calculators.

    The class provides :py:meth:`PowerSpectrum.forward` and the integration with
    :py:class:`torch.nn.Module`. For more details see
    :py:class:`rascaline.utils.PowerSpectrum`.

    :param calculator_1: first calculator
    :param calculator_1: second calculator
    :param species: List of `species_neighbor` to fill all blocks with. This option
        might be useful when joining along the ``sample`` direction after computation.
        If :py:obj:`None` blocks are filled with `species_neighbor` from all blocks.
    :raises ValueError: If other calculators than
        :py:class:`rascaline.SphericalExpansion` or
        :py:class:`rascaline.LodeSphericalExpansion` are used.
    :raises ValueError: If ``'max_angular'`` of both calculators is different
    """

    def __init__(
        self,
        calculator_1: CalculatorBase,
        calculator_2: Optional[CalculatorBase] = None,
        species: Optional[List[int]] = None,
    ):
        torch.nn.Module.__init__(self)
        PowerSpectrumBase.__init__(
            self,
            calculator_1=calculator_1,
            calculator_2=calculator_2,
            species=species,
        )

    def forward(
        self,
        systems: Union[IntoSystem, List[IntoSystem]],
        gradients: Optional[List[str]] = None,
        use_native_system: bool = True,
    ) -> TensorMap:
        """forward just calls :py:meth:`PowerSpectrum.compute`"""

        return self.compute(
            systems=systems,
            gradients=gradients,
            use_native_system=use_native_system,
        )


module.__dict__["PowerSpectrum"] = PowerSpectrum
