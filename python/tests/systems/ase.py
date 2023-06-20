import numpy as np
import pytest

from rascaline.systems import AseSystem


ase = pytest.importorskip("ase")


@pytest.fixture
def system():
    positions = [
        (0, 0, 0),
        (0, 0, 1.4),
        (0, 0, -1.6),
    ]
    atoms = ase.Atoms(
        "CO2",
        positions=positions,
    )
    atoms.pbc = [False, False, False]
    return AseSystem(atoms)


def test_system_implementation(system):
    assert system.size() == 3
    assert np.all(system.species() == [6, 8, 8])

    positions = [
        (0, 0, 0),
        (0, 0, 1.4),
        (0, 0, -1.6),
    ]
    assert np.all(system.positions() == positions)
    assert np.all(system.cell() == [[0, 0, 0], [0, 0, 0], [0, 0, 0]])


def test_pairs(system):
    system.compute_neighbors(1.5)
    pairs = system.pairs()

    assert len(pairs) == 1
    assert pairs[0][:2] == (0, 1)
    assert pairs[0][2] == 1.4
    assert np.all(pairs[0][3] == [0, 0, 1.4])

    system.compute_neighbors(2.5)
    pairs = system.pairs()
    assert len(pairs) == 2
    assert pairs[0][:2] == (0, 1)
    assert pairs[0][2] == 1.4
    assert np.all(pairs[0][3] == [0, 0, 1.4])

    assert pairs[1][:2] == (0, 2)
    assert pairs[1][2] == 1.6
    assert np.all(pairs[1][3] == [0, 0, -1.6])

    system.compute_neighbors(3.5)
    pairs = system.pairs()
    assert len(pairs) == 3
    assert pairs[0][:2] == (0, 1)
    assert pairs[0][2] == 1.4
    assert np.all(pairs[0][3] == [0, 0, 1.4])

    assert pairs[1][:2] == (0, 2)
    assert pairs[1][2] == 1.6
    assert np.all(pairs[1][3] == [0, 0, -1.6])

    assert pairs[2][:2] == (1, 2)
    assert pairs[2][2] == 3.0
    assert np.all(pairs[2][3] == [0, 0, -3.0])


def test_pairs_containing(system):
    system.compute_neighbors(1.5)
    pairs = system.pairs_containing(0)
    assert len(pairs) == 1
    assert pairs[0][:2] == (0, 1)

    pairs = system.pairs_containing(1)
    assert len(pairs) == 1
    assert pairs[0][:2] == (0, 1)

    pairs = system.pairs_containing(2)
    assert len(pairs) == 0

    system.compute_neighbors(3.5)
    pairs = system.pairs_containing(0)
    assert len(pairs) == 2
    assert pairs[0][:2] == (0, 1)
    assert pairs[1][:2] == (0, 2)

    pairs = system.pairs_containing(1)
    assert len(pairs) == 2
    assert pairs[0][:2] == (0, 1)
    assert pairs[1][:2] == (1, 2)

    pairs = system.pairs_containing(2)
    assert len(pairs) == 2
    assert pairs[0][:2] == (0, 2)
    assert pairs[1][:2] == (1, 2)


def test_pbc_no_cell():
    atoms = ase.Atoms("C", positions=[(0, 0, 0)])
    atoms.pbc = [True, True, True]

    message = (
        "periodic boundary conditions are enabled, but the cell "
        "matrix is zero everywhere. You should set pbc to `False`, "
        "or the cell to its value."
    )
    with pytest.raises(ValueError, match=message):
        AseSystem(atoms)

    atoms.cell = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    atoms.pbc = True

    with pytest.raises(ValueError, match=message):
        AseSystem(atoms)


def test_partial_pbc():
    atoms = ase.Atoms("C", positions=[(0, 0, 0)])

    atoms.cell = [[10, 0, 0], [0, 10, 0], [0, 0, 10]]
    atoms.pbc = [True, True, False]

    message = (
        "different periodic boundary conditions on different axis " "are not supported"
    )
    with pytest.raises(ValueError, match=message):
        AseSystem(atoms)


def test_no_pbc_cell():
    atoms = ase.Atoms("C", positions=[(0, 0, 0)])

    atoms.cell = [[10, 0, 0], [0, 10, 0], [0, 0, 10]]
    atoms.pbc = False

    message = (
        "periodic boundary conditions are disabled, but the cell matrix is "
        "not zero, we will set the cell to zero."
    )
    with pytest.warns(Warning, match=message):
        AseSystem(atoms)
