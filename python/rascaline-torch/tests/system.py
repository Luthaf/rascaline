import ase
import numpy as np
import torch

from rascaline.torch import System, systems_to_torch


def test_system():
    # positional arguments
    system = System(
        torch.ones(4, dtype=torch.int32),
        torch.zeros((4, 3), dtype=torch.float64),
        torch.zeros((3, 3), dtype=torch.float64),
    )

    assert torch.all(system.species == torch.ones((4)))
    assert torch.all(system.positions == torch.zeros((4, 3)))
    assert torch.all(system.cell == torch.zeros((3, 3)))

    assert len(system) == 4
    assert str(system) == "System with 4 atoms, non periodic"

    # named arguments
    system = System(
        species=torch.ones(4, dtype=torch.int32),
        positions=torch.zeros((4, 3), dtype=torch.float64),
        cell=6 * torch.eye(3),
    )

    assert torch.all(system.species == torch.ones((4)))
    assert torch.all(system.positions == torch.zeros((4, 3)))
    assert torch.all(system.cell == 6 * torch.eye(3))

    assert len(system) == 4
    assert (
        str(system) == "System with 4 atoms, periodic cell: [6, 0, 0, 0, 6, 0, 0, 0, 6]"
    )


def test_system_conversion_from_ase():
    atoms = ase.Atoms(
        "CO",
        positions=[(0, 0, 0), (0, 0, 2)],
        cell=4 * np.eye(3),
        pbc=[True, True, True],
    )

    system = systems_to_torch(atoms)

    assert isinstance(system, torch.ScriptObject)

    assert isinstance(system.species, torch.Tensor)
    assert torch.all(system.species == torch.tensor([6, 8]))
    assert system.species.dtype == torch.int32
    assert not system.species.requires_grad

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


# define a wrapper class to make sure the types TorchScript uses for of all
# C-defined functions matches what we expect
class SystemWrap:
    def __init__(
        self,
        species: torch.Tensor,
        positions: torch.Tensor,
        cell: torch.Tensor,
    ):
        self._c = System(
            species=species,
            positions=positions,
            cell=cell,
        )

    def __str__(self) -> str:
        return self._c.__str__()

    def __repr__(self) -> str:
        return self._c.__repr__()

    def __len__(self) -> int:
        return self._c.__len__()

    def species(self) -> torch.Tensor:
        return self._c.species

    def positions(self) -> torch.Tensor:
        return self._c.positions

    def cell(self) -> torch.Tensor:
        return self._c.cell


def test_script():
    class TestModule(torch.nn.Module):
        def forward(self, x: SystemWrap) -> SystemWrap:
            return x

    module = TestModule()
    module = torch.jit.script(module)
