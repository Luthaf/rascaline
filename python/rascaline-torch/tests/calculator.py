from typing import List, Optional, Union

import pytest
import torch
from metatensor.torch import Labels, TensorMap
from metatensor.torch.atomistic import System

from rascaline.torch import CalculatorModule
from rascaline.torch.calculators import DummyCalculator


@pytest.fixture
def system():
    return System(
        species=torch.tensor([1, 1, 8, 8]),
        positions=torch.tensor([[0.0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]]),
        cell=torch.tensor([[10.0, 0, 0], [0, 10, 0], [0, 0, 10]]),
    )


def test_parameters():
    calculator = DummyCalculator(cutoff=3.2, delta=2, name="foo")

    expected = "dummy test calculator with cutoff: 3.2 - delta: 2 - name: foo"
    assert calculator.name == expected
    assert calculator.c_name == "dummy_calculator"

    expected = '{"cutoff": 3.2, "delta": 2, "name": "foo"}'
    assert calculator.parameters == expected
    assert calculator.cutoffs == [3.2]


def test_compute(system):
    calculator = DummyCalculator(cutoff=3.2, delta=2, name="")
    descriptor = calculator.compute(system, gradients=["positions"])

    assert len(descriptor.keys) == 2
    assert descriptor.keys.names == ["species_center"]
    assert torch.all(descriptor.keys.values == torch.tensor([[1], [8]]))

    H_block = descriptor.block({"species_center": 1})
    assert H_block.values.shape == (2, 2)
    assert torch.all(H_block.values[0] == torch.tensor([2, 6]))
    assert torch.all(H_block.values[1] == torch.tensor([3, 6]))

    assert len(H_block.samples) == 2
    assert H_block.samples.names == ["structure", "center"]
    assert tuple(H_block.samples[0]) == (0, 0)
    assert tuple(H_block.samples[1]) == (0, 1)

    assert len(H_block.components) == 0

    assert len(H_block.properties) == 2
    assert H_block.properties.names == ["index_delta", "x_y_z"]
    assert tuple(H_block.properties[0]) == (1, 0)
    assert tuple(H_block.properties[1]) == (0, 1)

    gradient = H_block.gradient("positions")
    assert gradient.values.shape == (8, 3, 2)
    for i in range(gradient.values.shape[0]):
        assert torch.all(gradient.values[i, 0, :] == torch.tensor([0, 1]))
        assert torch.all(gradient.values[i, 1, :] == torch.tensor([0, 1]))
        assert torch.all(gradient.values[i, 2, :] == torch.tensor([0, 1]))

    assert len(gradient.samples) == 8
    assert gradient.samples.names == ["sample", "structure", "atom"]
    assert tuple(gradient.samples[0]) == (0, 0, 0)
    assert tuple(gradient.samples[1]) == (0, 0, 1)
    assert tuple(gradient.samples[2]) == (0, 0, 2)
    assert tuple(gradient.samples[3]) == (0, 0, 3)
    assert tuple(gradient.samples[4]) == (1, 0, 0)
    assert tuple(gradient.samples[5]) == (1, 0, 1)
    assert tuple(gradient.samples[6]) == (1, 0, 2)
    assert tuple(gradient.samples[7]) == (1, 0, 3)

    assert len(gradient.components) == 1
    component = gradient.components[0]
    assert len(component) == 3
    assert component.names == ["direction"]
    assert tuple(component[0]) == (0,)
    assert tuple(component[1]) == (1,)
    assert tuple(component[2]) == (2,)

    assert len(gradient.properties) == 2
    assert gradient.properties.names == ["index_delta", "x_y_z"]
    assert tuple(gradient.properties[0]) == (1, 0)
    assert tuple(gradient.properties[1]) == (0, 1)

    O_block = descriptor.block({"species_center": 8})
    assert O_block.values.shape == (2, 2)
    assert torch.all(O_block.values[0] == torch.tensor([4, 6]))
    assert torch.all(O_block.values[1] == torch.tensor([5, 6]))


def test_compute_native_system_error_raise(system):
    calculator = DummyCalculator(cutoff=3.2, delta=2, name="")
    with pytest.raises(ValueError, match="only `use_native_system=True` is supported"):
        calculator.compute(system, gradients=["positions"], use_native_system=False)


def test_compute_multiple_systems(system):
    systems = [system, system, system]
    calculator = DummyCalculator(cutoff=3.2, delta=2, name="")
    descriptor = calculator.compute(systems)

    H_block = descriptor.block({"species_center": 1})
    assert H_block.values.shape == (6, 2)
    expected = torch.tensor([(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)])
    assert torch.all(H_block.samples.values == expected)

    O_block = descriptor.block({"species_center": 8})
    assert O_block.values.shape == (6, 2)


def test_key_selection(system):
    calculator = DummyCalculator(cutoff=3.2, delta=2, name="")
    descriptor = calculator.compute(
        system,
        selected_keys=Labels("species_center", torch.IntTensor([[12], [1]])),
    )
    assert torch.all(descriptor.keys.values == torch.tensor([[12], [1]]))

    H_block = descriptor.block({"species_center": 1})
    assert H_block.values.shape == (2, 2)

    missing_block = descriptor.block({"species_center": 12})
    assert missing_block.values.shape == (0, 2)


def test_samples_selection(system):
    calculator = DummyCalculator(cutoff=3.2, delta=2, name="")
    descriptor = calculator.compute(
        system,
        selected_samples=Labels("center", torch.IntTensor([[0], [2]])),
    )

    H_block = descriptor.block({"species_center": 1})
    assert torch.all(H_block.samples.values == torch.tensor([(0, 0)]))

    O_block = descriptor.block({"species_center": 8})
    assert torch.all(O_block.samples.values == torch.tensor([(0, 2)]))


def test_properties_selection(system):
    calculator = DummyCalculator(cutoff=3.2, delta=2, name="")
    descriptor = calculator.compute(
        system,
        selected_properties=Labels("index_delta", torch.IntTensor([[0]])),
    )

    H_block = descriptor.block({"species_center": 1})
    assert torch.all(H_block.properties.values == torch.tensor([(0, 1)]))

    O_block = descriptor.block({"species_center": 8})
    assert torch.all(O_block.properties.values == torch.tensor([(0, 1)]))


def test_base_classes():
    assert DummyCalculator.__module__ == "rascaline.torch.calculators"

    assert DummyCalculator.__bases__ == (CalculatorModule,)
    assert CalculatorModule.__bases__ == (torch.nn.Module,)


def test_script(tmpdir):
    class TestModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calculator = DummyCalculator(cutoff=3.2, delta=6, name="foo")

        def forward(
            self,
            systems: List[System],
            gradients: Optional[List[str]] = None,
            use_native_system: bool = True,
            selected_samples: Optional[Union[Labels, TensorMap]] = None,
            selected_properties: Optional[Union[Labels, TensorMap]] = None,
            selected_keys: Optional[Labels] = None,
        ) -> TensorMap:
            return self.calculator(
                systems=systems,
                gradients=gradients,
                use_native_system=use_native_system,
                selected_samples=selected_samples,
                selected_properties=selected_properties,
                selected_keys=selected_keys,
            )

    module = TestModule()
    module = torch.jit.script(module)

    with tmpdir.as_cwd():
        torch.jit.save(module, "test-save.torch")
        module = torch.jit.load("test-save.torch")
