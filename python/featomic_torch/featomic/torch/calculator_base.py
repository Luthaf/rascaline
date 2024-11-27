from typing import List, Optional, Union

import torch
from metatensor.torch import Labels, TensorMap
from metatensor.torch.atomistic import NeighborListOptions

from .system import System


CalculatorHolder = torch.classes.featomic.CalculatorHolder


def register_autograd(
    systems: Union[List[System], System],
    precomputed: TensorMap,
    forward_gradients: Optional[List[str]] = None,
) -> TensorMap:
    """
    Register autograd nodes between ``system.positions`` and ``system.cell`` for each of
    the systems and the values in the ``precomputed``
    :py:class:`metatensor.torch.TensorMap`.

    This is an advanced function must users should not need to use.

    The autograd nodes ``backward`` function will use the gradients already stored in
    ``precomputed``, meaning that if any of the system's positions ``requires_grad``,
    ``precomputed`` must contain ``"positions"`` gradients. Similarly, if any of the
    system's cell ``requires_grad``, ``precomputed`` must contain ``"cell"`` gradients.

    :param systems: list of system used to compute ``precomputed``
    :param precomputed: precomputed :py:class:`metatensor.torch.TensorMap`
    :param forward_gradients: which gradients to keep in the output, defaults to None
    """
    if forward_gradients is None:
        forward_gradients = []

    if not isinstance(systems, list):
        systems = [systems]

    return torch.ops.featomic.register_autograd(systems, precomputed, forward_gradients)


class CalculatorModule(torch.nn.Module):
    """
    This is the base class for calculators in featomic-torch, providing the
    :py:meth:`CalculatorModule.compute` function and integration with
    :py:class:`torch.nn.Module`.

    One can initialize a py:class:`CalculatorModule` in two ways: either directly with
    the registered name and JSON parameter string (which are documented in the
    :ref:`userdoc-calculators`); or through one of the child class documented below.

    :param name: name used to register this calculator
    :param parameters: JSON parameter string for the calculator
    """

    def __init__(self, name: str, parameters: str):
        """"""
        # empty docstring here for the docs to render corectly
        super().__init__()
        self._c_name = name
        self._c = CalculatorHolder(name=name, parameters=parameters)

    @property
    def name(self) -> str:
        """name of this calculator"""
        return self._c.name

    @property
    def c_name(self) -> str:
        """name used to register & create this calculator"""
        return self._c_name

    @property
    def parameters(self) -> str:
        """parameters (formatted as JSON) used to create this calculator"""
        return self._c.parameters

    @property
    def cutoffs(self) -> List[float]:
        """all the radial cutoffs used by this calculator's neighbors lists"""
        return self._c.cutoffs

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        options = []
        for cutoff in self.cutoffs:
            options.append(
                NeighborListOptions(
                    cutoff=cutoff,
                    full_list=False,
                    # we will re-filter the NL when converting to featomic internal
                    # type, so we don't need the engine to pre-filter it for us
                    strict=False,
                    requestor="featomic",
                )
            )
        return options

    def compute(
        self,
        systems: Union[System, List[System]],
        gradients: Optional[List[str]] = None,
        use_native_system: bool = True,
        selected_samples: Optional[Union[Labels, TensorMap]] = None,
        selected_properties: Optional[Union[Labels, TensorMap]] = None,
        selected_keys: Optional[Labels] = None,
    ) -> TensorMap:
        """Runs a calculation with this calculator on the given ``systems``.

        .. seealso::

            :py:func:`featomic.calculators.CalculatorBase.compute` for more information
            on the different parameters of this function.

        :param systems: single system or list of systems on which to run the
            calculation. If any of the systems' ``positions`` or ``cell`` has
            ``requires_grad`` set to ``True``, then the corresponding gradients are
            computed and registered as a custom node in the computational graph, to
            allow backward propagation of the gradients later.

        :param gradients: List of forward gradients to keep in the output. If this is
            ``None`` or an empty list ``[]``, no gradients are kept in the output. Some
            gradients might still be computed at runtime to allow for backward
            propagation.

        :param use_native_system: This can only be ``True``, and is here for
            compatibility with the same parameter on
            :py:meth:`featomic.calculators.CalculatorBase.compute`.

        :param selected_samples: Set of samples on which to run the calculation, with
            the same meaning as in
            :py:func:`featomic.calculators.CalculatorBase.compute`.

        :param selected_properties: Set of properties to compute, with the same meaning
            as in :py:func:`featomic.calculators.CalculatorBase.compute`.

        :param selected_keys: Selection for the keys to include in the output, with the
            same meaning as in :py:func:`featomic.calculators.CalculatorBase.compute`.
        """
        if gradients is None:
            gradients = []

        if not isinstance(systems, list):
            systems = [systems]

        # We have this parameter to have the same API as featomic.
        if not use_native_system:
            raise ValueError("only `use_native_system=True` is supported")

        options = torch.classes.featomic.CalculatorOptions()
        options.gradients = gradients
        options.selected_samples = selected_samples
        options.selected_properties = selected_properties
        options.selected_keys = selected_keys

        return self._c.compute(systems=systems, options=options)

    def forward(
        self,
        systems: List[System],
        gradients: Optional[List[str]] = None,
        use_native_system: bool = True,
        selected_samples: Optional[Union[Labels, TensorMap]] = None,
        selected_properties: Optional[Union[Labels, TensorMap]] = None,
        selected_keys: Optional[Labels] = None,
    ) -> TensorMap:
        """forward just calls :py:meth:`CalculatorModule.compute`"""

        return self.compute(
            systems=systems,
            gradients=gradients,
            use_native_system=use_native_system,
            selected_samples=selected_samples,
            selected_properties=selected_properties,
            selected_keys=selected_keys,
        )
