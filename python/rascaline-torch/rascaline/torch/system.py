from typing import List, Optional, Sequence, overload

import torch
from metatensor.torch.atomistic import System
from packaging import version

import rascaline


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
    Convert a arbitrary system to metatensor's atomistic
    :py:class:`metatensor.torch.atomistic.System`, putting all the data in
    :py:class:`torch.Tensor` and making the overall object compatible with TorchScript.

    :param system: any system supported by rascaline. If this is an iterable of system,
        this function converts them all and returns a list of converted systems.

    :param positions_requires_grad: The value of ``requires_grad`` on the output
        ``positions``. If :py:obj:`None`` and the positions of the input is already a
        :py:class:`torch.Tensor`, ``requires_grad`` is kept the same. Otherwise it is
        initialized to :py:obj:`False`.

    :param cell_requires_grad: The value of ``requires_grad`` on the output ``cell``. If
        :py:obj:`None` and the positions of the input is already a
        :py:class:`torch.Tensor`, ``requires_grad`` is kept the same. Otherwise it is
        initialized to :py:obj:`False`.
    """

    try:
        return _system_to_torch(systems, positions_requires_grad, cell_requires_grad)
    except TypeError:
        # try iterating over the systems
        return [
            _system_to_torch(system, positions_requires_grad, cell_requires_grad)
            for system in systems
        ]


def _system_to_torch(system, positions_requires_grad, cell_requires_grad):
    if not _is_torch_system(system):
        system = rascaline.systems.wrap_system(system)
        system = System(
            species=torch.tensor(system.types()),
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

    if version.parse(torch.__version__) >= version.parse("2.1"):
        return system._type().name() == "System"

    # For older torch version, we check that we have the right properties
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
