import numpy as np
import pytest

from rascaline.systems import PyscfSystem


pyscf = pytest.importorskip("pyscf")


@pytest.fixture
def system():
    atoms = pyscf.M(
        atom="C 0 0 0; O 0 0 1.4; O 0 0 -1.6",
    )
    # atoms.pbc = [False, False, False]
    return PyscfSystem(atoms)


def test_system_implementation(system):
    assert system.size() == 3
    assert np.all(system.species() == [6, 8, 8])

    positions = np.array(
        [
            (0, 0, 0),
            (0, 0, 1.4),
            (0, 0, -1.6),
        ]
    )
    print(system.positions(), positions)
    assert np.allclose(system.positions(), positions, rtol=1e-14)
    assert np.all(system.cell() == [[0, 0, 0], [0, 0, 0], [0, 0, 0]])


def test_pbc_data():
    import pyscf.pbc

    atoms = pyscf.pbc.gto.Cell(
        atom="H 0 0 0; H 1 1 1",
        a=np.array([[2, 0, 0], [0, 2, 1], [0, 0, 2]], dtype=float),
    ).build()
    ras_sys = PyscfSystem(atoms)
    assert np.allclose(
        ras_sys.positions(),
        np.array([[0, 0, 0], [1, 1, 1]], dtype=float),
        rtol=1e-14,
    )
