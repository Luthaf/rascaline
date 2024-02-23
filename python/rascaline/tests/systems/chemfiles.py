import numpy as np
import pytest

from rascaline.calculators import DummyCalculator
from rascaline.status import RascalError
from rascaline.systems import ChemfilesSystem


chemfiles = pytest.importorskip("chemfiles")


def test_system_implementation():
    frame = chemfiles.Frame()
    frame.add_atom(chemfiles.Atom("C"), (0, 0, 0))
    frame.add_atom(chemfiles.Atom("CH3"), (0, 1, 0))
    frame.add_atom(chemfiles.Atom("X"), (0, 2, 0))
    frame.add_atom(chemfiles.Atom("Zn1", "Zn"), (0, 3, 0))

    system = ChemfilesSystem(frame)

    assert system.size() == 4
    assert np.all(system.types() == [6, 120, 121, 30])
    positions = [
        (0, 0, 0),
        (0, 1, 0),
        (0, 2, 0),
        (0, 3, 0),
    ]
    assert np.all(system.positions() == positions)


def test_cell():
    frame = chemfiles.Frame()

    system = ChemfilesSystem(frame)
    assert np.all(system.cell() == 0)

    frame.cell = chemfiles.UnitCell((3, 3, 3))
    assert np.allclose(system.cell(), np.diag([3, 3, 3]))

    frame.cell = chemfiles.UnitCell((3, 3, 3), (60, 60, 60))
    expected = [
        [3.0, 0.0, 0.0],
        [1.5, 2.59807621, 0.0],
        [1.5, 0.8660254, 2.44948974],
    ]
    assert np.allclose(system.cell(), expected)


def test_compute():
    frame = chemfiles.Frame()
    frame.add_atom(chemfiles.Atom("C"), (0, 0, 0))
    frame.add_atom(chemfiles.Atom("C"), (0, 1, 0))
    frame.add_atom(chemfiles.Atom("C"), (0, 2, 0))
    frame.add_atom(chemfiles.Atom("C"), (0, 3, 0))

    calculator = DummyCalculator(cutoff=3.4, delta=1, name="")

    message = (
        "error from external code \\(status -1\\): "
        "call to rascal_system_t.compute_neighbors failed"
    )
    with pytest.raises(RascalError, match=message) as cm:
        calculator.compute(frame, use_native_system=False)

    cause = "chemfiles systems can only be used with 'use_native_system=True'"
    assert cm.value.__cause__.args[0] == cause

    # use_native_system=True should work fine
    descriptor = calculator.compute(frame, use_native_system=True)
    assert len(descriptor.keys) == 1
