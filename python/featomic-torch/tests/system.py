import ase
import numpy as np
import torch

from featomic.torch import systems_to_torch


def test_system_conversion_from_ase():
    atoms = ase.Atoms(
        "CO",
        positions=[(0, 0, 0), (0, 0, 2)],
        cell=4 * np.eye(3),
        pbc=[True, True, True],
    )

    system = systems_to_torch(atoms)

    assert isinstance(system, torch.ScriptObject)

    assert isinstance(system.types, torch.Tensor)
    assert torch.all(system.types == torch.tensor([6, 8]))
    assert system.types.dtype == torch.int32
    assert not system.types.requires_grad

    assert isinstance(system.positions, torch.Tensor)
    assert torch.all(system.positions == torch.tensor([(0, 0, 0), (0, 0, 2)]))
    assert system.positions.dtype == torch.float64
    assert not system.positions.requires_grad

    assert isinstance(system.cell, torch.Tensor)
    assert torch.all(system.cell == 4 * torch.eye(3))
    assert system.cell.dtype == torch.float64
    assert not system.cell.requires_grad

    system = systems_to_torch(atoms, positions_requires_grad=True)

    assert system.positions.requires_grad
    assert not system.cell.requires_grad

    # we can send a torch System through this function, and change the requires_grad
    system = systems_to_torch(
        system,
        cell_requires_grad=True,
        positions_requires_grad=False,
    )

    assert not system.positions.requires_grad
    assert system.cell.requires_grad

    # test a list of ase.Atoms
    systems = systems_to_torch([atoms, atoms])
    assert isinstance(systems[0], torch.ScriptObject)
