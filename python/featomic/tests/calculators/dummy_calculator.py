import numpy as np
import pytest

from featomic import FeatomicError
from featomic.calculators import DummyCalculator

from ..test_systems import SystemForTests


def test_name():
    calculator = DummyCalculator(cutoff=3.2, delta=12, name="foo")
    expected = "dummy test calculator with cutoff: 3.2 - delta: 12 - name: foo"
    assert calculator.name == expected

    assert calculator.c_name == "dummy_calculator"

    # very long name, checking that we can pass large string back and forth
    name = "abc" * 2048
    calculator = DummyCalculator(cutoff=3.2, delta=12, name=name)
    expected = f"dummy test calculator with cutoff: 3.2 - delta: 12 - name: {name}"
    assert calculator.name == expected


def test_parameters():
    calculator = DummyCalculator(cutoff=3.2, delta=12, name="foo")
    assert calculator.parameters == '{"cutoff": 3.2, "delta": 12, "name": "foo"}'

    assert calculator.cutoffs == [3.2]


def test_bad_parameters():
    message = (
        'json error: invalid type: string "12", expected isize at line 1 column 29'
    )

    with pytest.raises(FeatomicError, match=message):
        _ = DummyCalculator(cutoff=3.2, delta="12", name="foo")


def test_compute():
    system = SystemForTests()
    calculator = DummyCalculator(cutoff=3.2, delta=2, name="")
    descriptor = calculator.compute(
        system, use_native_system=False, gradients=["positions"]
    )

    assert len(descriptor.keys) == 2
    assert descriptor.keys.names == ["center_type"]
    assert tuple(descriptor.keys[0]) == (1,)
    assert tuple(descriptor.keys[1]) == (8,)

    H_block = descriptor.block(center_type=1)
    assert H_block.values.shape == (2, 2)
    assert np.all(H_block.values[0] == (2, 11))
    assert np.all(H_block.values[1] == (3, 13))

    assert len(H_block.samples) == 2
    assert H_block.samples.names == ["system", "atom"]
    assert tuple(H_block.samples[0]) == (0, 0)
    assert tuple(H_block.samples[1]) == (0, 1)

    assert len(H_block.components) == 0

    assert len(H_block.properties) == 2
    assert H_block.properties.names == ["index_delta", "x_y_z"]
    assert tuple(H_block.properties[0]) == (1, 0)
    assert tuple(H_block.properties[1]) == (0, 1)

    gradient = H_block.gradient("positions")
    assert gradient.values.shape == (5, 3, 2)
    for i in range(gradient.values.shape[0]):
        assert np.all(gradient.values[i, 0, :] == (0, 1))
        assert np.all(gradient.values[i, 1, :] == (0, 1))
        assert np.all(gradient.values[i, 2, :] == (0, 1))

    assert len(gradient.samples) == 5
    assert gradient.samples.names == ["sample", "system", "atom"]
    assert tuple(gradient.samples[0]) == (0, 0, 0)
    assert tuple(gradient.samples[1]) == (0, 0, 1)
    assert tuple(gradient.samples[2]) == (1, 0, 0)
    assert tuple(gradient.samples[3]) == (1, 0, 1)
    assert tuple(gradient.samples[4]) == (1, 0, 2)

    assert len(gradient.components) == 1
    component = gradient.components[0]
    assert len(component) == 3
    assert component.names == ["xyz"]
    assert tuple(component[0]) == (0,)
    assert tuple(component[1]) == (1,)
    assert tuple(component[2]) == (2,)

    assert len(gradient.properties) == 2
    assert gradient.properties.names == ["index_delta", "x_y_z"]
    assert tuple(gradient.properties[0]) == (1, 0)
    assert tuple(gradient.properties[1]) == (0, 1)

    O_block = descriptor.block(center_type=8)
    assert O_block.values.shape == (2, 2)
    assert np.all(O_block.values[0] == (4, 6))
    assert np.all(O_block.values[1] == (5, 5))


def test_compute_multiple_systems():
    systems = [SystemForTests(), SystemForTests(), SystemForTests()]
    calculator = DummyCalculator(cutoff=3.2, delta=2, name="")
    descriptor = calculator.compute(systems, use_native_system=False)

    H_block = descriptor.block(center_type=1)
    assert H_block.values.shape == (6, 2)
    expected = np.array([(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)])
    np.testing.assert_equal(H_block.samples.values, expected)

    O_block = descriptor.block(center_type=8)
    assert O_block.values.shape == (6, 2)
