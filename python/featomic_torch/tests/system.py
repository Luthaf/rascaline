import ase
import numpy as np
import pytest
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("as_list", [False, True])
def test_conversion(dtype, as_list):
    atoms = ase.Atoms(
        "CO", positions=[(0, 0, 0), (0, 0, 2)], cell=4 * np.eye(3), pbc=True
    )
    result = systems_to_torch(
        [atoms, atoms] if as_list else atoms, dtype=dtype, positions_requires_grad=True
    )
    systems = result if as_list else [result]
    for s in systems:
        assert s.positions.dtype == dtype
        assert s.cell.dtype == dtype
        assert s.types.dtype == torch.int32
        assert s.pbc.dtype == torch.bool
        assert s.positions.requires_grad
        assert not s.cell.requires_grad
        torch.testing.assert_close(
            s.positions, torch.tensor(atoms.positions, dtype=dtype)
        )
        torch.testing.assert_close(s.cell, torch.tensor(atoms.cell.array, dtype=dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_existing(dtype):
    s = systems_to_torch(
        ase.Atoms("H", positions=[[1, 2, 3]], cell=4 * np.eye(3), pbc=True)
    )
    original_dtype = s.positions.dtype
    out = systems_to_torch(
        s, dtype=dtype, positions_requires_grad=True, cell_requires_grad=True
    )
    assert out.positions.dtype == dtype
    assert out.cell.dtype == dtype
    assert s.positions.dtype == original_dtype
    (out.positions.square().sum() + out.cell.square().sum()).backward()
    assert out.positions.grad is not None
    assert out.cell.grad is not None


def test_default_and_positional():
    s = systems_to_torch(ase.Atoms("H", positions=[[0, 0, 0]]), True, False)
    assert s.positions.dtype == torch.float64
    assert s.positions.requires_grad
    assert systems_to_torch(s) is s
