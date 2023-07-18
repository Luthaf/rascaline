from typing import List, Optional, Union

import equistore.torch
import torch

from .system import System


CalculatorHolder = torch.classes.rascaline.CalculatorHolder


class CalculatorModule(torch.nn.Module):
    """
    This is the base class for calculators in rascaline-torch, providing the
    :py:meth:`CalculatorBase.compute` function and integration with
    :py:class:`torch.nn.Module`.

    One can initialize a ``Calculator`` in two ways: either directly with the registered
    name and JSON parameter string (which are documented in the
    :ref:`userdoc-calculators`); or through one of the child class documented below.

    :param name: name used to register this calculator
    :param parameters: JSON parameter string for the calculator
    """

    def __init__(self, name: str, parameters: str):
        """"""
        super().__init__()
        self._c = CalculatorHolder(name=name, parameters=parameters)

    @property
    def name(self) -> str:
        """name of this calculator"""
        return self._c.name

    @property
    def parameters(self) -> str:
        """parameters (formatted as JSON) used to create this calculator"""
        return self._c.parameters

    def compute(
        self,
        systems: Union[System, List[System]],
        gradients: Optional[List[str]] = None,
    ) -> equistore.torch.TensorMap:
        """Runs a calculation with this calculator on the given ``systems``.

        .. seealso::

            :py:func:`rascaline.CalculatorBase.compute` for more information on the
            different parameters of this function.

        :param systems: single system or list of systems on which to run the
            calculation. If any of the systems' ``positions`` or ``cell`` has
            ``requires_grad`` set to ``True``, then the corresponding gradients are
            computed and registered as a custom node in the computational graph, to
            allow backward propagation of the gradients later.

        :param gradients: List of forward gradients to keep in the output. If this is
            ``None`` or an empty list ``[]``, no gradients are kept in the output. Some
            gradients might still be computed at runtime to allow for backward
            propagation.
        """
        if gradients is None:
            gradients = []

        if not isinstance(systems, list):
            systems = [systems]

        return self._c.compute(systems=systems, gradients=gradients)

    def forward(
        self,
        systems: List[System],
        gradients: Optional[List[str]] = None,
    ) -> equistore.torch.TensorMap:
        """forward just calls :py:meth:`CalculatorModule.compute`"""

        return self.compute(systems=systems, gradients=gradients)
