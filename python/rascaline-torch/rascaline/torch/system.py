import os
from typing import List, Optional, Sequence, overload

import torch

import rascaline


if os.environ.get("RASCALINE_IMPORT_FOR_SPHINX") is None:
    System = torch.classes.rascaline.System
else:
    # Documentation for the `System` class, only used when loading the code with sphinx
    class System:
        """
        A single system for which we want to run a calculation.

        Contrary to the Python API of rascaline, only this class is supported as input
        to a :py:class:`CalculatorModule`, but any of the supported system class can be
        transformed to this one with :py:func:`systems_to_torch`.
        """

        def __init__(
            self,
            species: torch.Tensor,
            positions: torch.Tensor,
            cell: torch.Tensor,
        ):
            """
            :param species: species of the atoms/particles in this system. This should
                be a 1D array of integer containing different values for different
                system. The species will typically match the atomic element, but does
                not have to.
            :param positions: positions of the atoms/particles in this system. This
                should be a ``len(species) x 3`` 2D array containing the positions of
                each atom.
            :param cell: 3x3 cell matrix for periodic boundary conditions, where each
                row is one of the cell vector. Use a matrix filled with ``0`` for
                non-periodic systems.
            """

        @property
        def species(self) -> torch.Tensor:
            """the species of the atoms/particles in this system"""

        @property
        def positions(self) -> torch.Tensor:
            """the positions of the atoms/particles in this system"""

        @property
        def cell(self) -> torch.Tensor:
            """
            the bounding box for the atoms/particles in this system under periodic
            boundary conditions, or a matrix filled with ``0`` for non-periodic systems
            """


@overload
def systems_to_torch(
    systems: rascaline.systems.IntoSystem,
    positions_requires_grad: Optional[bool] = None,
    cell_requires_grad: Optional[bool] = None,
) -> System:
    pass


@overload
def systems_to_torch(
    systems: Sequence[rascaline.systems.IntoSystem],
    positions_requires_grad: Optional[bool] = None,
    cell_requires_grad: Optional[bool] = None,
) -> List[System]:
    pass


def systems_to_torch(
    systems,
    positions_requires_grad=None,
    cell_requires_grad=None,
) -> List[System]:
    """
    Convert a arbitrary system to rascaline-torch :py:class:`System`, putting all the
    data in :py:class:`torch.Tensor` and making the overall object compatible with
    TorchScript.

    :param system: any system supported by rascaline. If this is an iterable of system,
        this function converts them all and returns a list of converted systems.

    :param positions_requires_grad: The value of ``requires_grad`` on the output
        ``positions``. If ``None`` and the positions of the input is already a
        :py:class:`torch.Tensor`, ``requires_grad`` is kept the same. Otherwise it is
        initialized to ``False``.

    :param cell_requires_grad: The value of ``requires_grad`` on the output ``cell``. If
        ``None`` and the positions of the input is already a :py:class:`torch.Tensor`,
        ``requires_grad`` is kept the same. Otherwise it is initialized to ``False``.
    """

    try:
        return _system_to_torch(systems, positions_requires_grad, cell_requires_grad)
    except TypeError:
        # try iterating over the systems
        return [
            _system_to_torch(systems, positions_requires_grad, cell_requires_grad)
            for system in systems
        ]


def _system_to_torch(system, positions_requires_grad, cell_requires_grad):
    if not _is_torch_system(system):
        system = rascaline.systems.wrap_system(system)
        system = System(
            species=torch.tensor(system.species()),
            positions=torch.tensor(system.positions()),
            cell=torch.tensor(system.cell()),
        )

    if positions_requires_grad is not None:
        system.positions.requires_grad_(positions_requires_grad)

    if cell_requires_grad is not None:
        system.cell.requires_grad_(cell_requires_grad)

    return system


def _is_torch_system(system):
    if not isinstance(system, torch.ScriptObject):
        return False

    # we would like to use system._type() here, but it is broken in torch <2.1
    properties = system._properties()
    if len(properties) != 3:
        return False

    if properties[0].name != "species":
        return False

    if properties[1].name != "positions":
        return False

    if properties[2].name != "cell":
        return False

    return True
