import pytest

from featomic import FeatomicError
from featomic.calculators import DummyCalculator
from featomic.systems import SystemBase


class UnimplementedSystem(SystemBase):
    pass


def test_unimplemented():
    system = UnimplementedSystem()
    calculator = DummyCalculator(cutoff=3.2, delta=2, name="")

    message = (
        "error from external code \\(status -1\\): "
        "call to featomic_system_t.types failed"
    )
    with pytest.raises(FeatomicError, match=message) as cm:
        calculator.compute(system, use_native_system=False)

    assert cm.value.__cause__.args[0] == "System.types method is not implemented"
