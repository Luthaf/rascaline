import pytest

from rascaline import RascalError
from rascaline.calculators import DummyCalculator
from rascaline.systems import SystemBase


class UnimplementedSystem(SystemBase):
    pass


def test_unimplemented():
    system = UnimplementedSystem()
    calculator = DummyCalculator(cutoff=3.2, delta=2, name="")

    message = (
        "error from external code \\(status -1\\): "
        "call to rascal_system_t.species failed"
    )
    with pytest.raises(RascalError, match=message) as cm:
        calculator.compute(system, use_native_system=False)

    assert cm.value.__cause__.args[0] == "System.species method is not implemented"
