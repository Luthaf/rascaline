import numpy as np
import pytest

from rascaline.systems import AseSystem


pyscf = pytest.importorskip("pyscf")


@pytest.fixture
def system():
    atoms = pyscf.M(
        atom = "C 0 0 0; O 0 0 1.4; O 0 0 -1.6",
    )
    #atoms.pbc = [False, False, False]
    return PyscfSystem(atoms)


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


def test_pbc_data():
    #import pyscf.pbo
    atoms = pyscf.pbo.gto.Cell(
        atom = "H 0 0 0; H 1 1 1",
        a = np.array([[2,0,0],[0,2,1],[0,0,2]],dtype=float)
    )
    ras_sys = PyscfSystem(atoms)
    assert ras_sys.positions() == np.array([[0,0,0],[1,1,1]],dtype=float)
