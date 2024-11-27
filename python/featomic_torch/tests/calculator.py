import os
from typing import List, Optional, Union

import pytest
import torch
from metatensor.torch import Labels, TensorMap
from metatensor.torch.atomistic import System

from featomic.torch import CalculatorModule
from featomic.torch.calculators import DummyCalculator


@pytest.fixture
def system():
    return System(
        types=torch.tensor([1, 1, 8, 8]),
        positions=torch.tensor([[0.0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]]),
        cell=torch.tensor([[10.0, 0, 0], [0, 10, 0], [0, 0, 10]]),
        pbc=torch.tensor([True, True, True]),
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
    assert descriptor.keys.names == ["center_type"]
    assert torch.all(descriptor.keys.values == torch.tensor([[1], [8]]))

    H_block = descriptor.block({"center_type": 1})
    assert H_block.values.shape == (2, 2)
    assert torch.all(H_block.values[0] == torch.tensor([2, 6]))
    assert torch.all(H_block.values[1] == torch.tensor([3, 6]))

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
    assert gradient.values.shape == (8, 3, 2)
    for i in range(gradient.values.shape[0]):
        assert torch.all(gradient.values[i, 0, :] == torch.tensor([0, 1]))
        assert torch.all(gradient.values[i, 1, :] == torch.tensor([0, 1]))
        assert torch.all(gradient.values[i, 2, :] == torch.tensor([0, 1]))

    assert len(gradient.samples) == 8
    assert gradient.samples.names == ["sample", "system", "atom"]
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
    assert component.names == ["xyz"]
    assert tuple(component[0]) == (0,)
    assert tuple(component[1]) == (1,)
    assert tuple(component[2]) == (2,)

    assert len(gradient.properties) == 2
    assert gradient.properties.names == ["index_delta", "x_y_z"]
    assert tuple(gradient.properties[0]) == (1, 0)
    assert tuple(gradient.properties[1]) == (0, 1)

    O_block = descriptor.block({"center_type": 8})
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

    H_block = descriptor.block({"center_type": 1})
    assert H_block.values.shape == (6, 2)
    expected = torch.tensor([(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)])
    assert torch.all(H_block.samples.values == expected)

    O_block = descriptor.block({"center_type": 8})
    assert O_block.values.shape == (6, 2)


def test_key_selection(system):
    calculator = DummyCalculator(cutoff=3.2, delta=2, name="")
    descriptor = calculator.compute(
        system,
        selected_keys=Labels("center_type", torch.IntTensor([[12], [1]])),
    )
    assert torch.all(descriptor.keys.values == torch.tensor([[12], [1]]))

    H_block = descriptor.block({"center_type": 1})
    assert H_block.values.shape == (2, 2)

    missing_block = descriptor.block({"center_type": 12})
    assert missing_block.values.shape == (0, 2)


def test_samples_selection(system):
    calculator = DummyCalculator(cutoff=3.2, delta=2, name="")
    descriptor = calculator.compute(
        system,
        selected_samples=Labels("atom", torch.IntTensor([[0], [2]])),
    )

    H_block = descriptor.block({"center_type": 1})
    assert torch.all(H_block.samples.values == torch.tensor([(0, 0)]))

    O_block = descriptor.block({"center_type": 8})
    assert torch.all(O_block.samples.values == torch.tensor([(0, 2)]))


def test_properties_selection(system):
    calculator = DummyCalculator(cutoff=3.2, delta=2, name="")
    descriptor = calculator.compute(
        system,
        selected_properties=Labels("index_delta", torch.IntTensor([[0]])),
    )

    H_block = descriptor.block({"center_type": 1})
    assert torch.all(H_block.properties.values == torch.tensor([(0, 1)]))

    O_block = descriptor.block({"center_type": 8})
    assert torch.all(O_block.properties.values == torch.tensor([(0, 1)]))


def test_base_classes():
    assert DummyCalculator.__module__ == "featomic.torch.calculators"

    assert DummyCalculator.__bases__ == (CalculatorModule,)
    assert CalculatorModule.__bases__ == (torch.nn.Module,)


def test_different_device_dtype_errors(system):
    calculator = DummyCalculator(cutoff=3.2, delta=2, name="")

    message = "all systems should have the same dtype"
    with pytest.raises(TypeError, match=message):
        calculator.compute(
            [
                system.to(dtype=torch.float32),
                system.to(dtype=torch.float64),
            ]
        )

    message = "featomic only supports float64 and float32 data"
    with pytest.raises(TypeError, match=message):
        calculator.compute(system.to(dtype=torch.float16))

    # Different devices
    custom_device = None
    if can_use_mps_backend():
        custom_device = torch.device("mps:0")

    if torch.cuda.is_available():
        custom_device = torch.device("cuda:0")

    if custom_device is not None:
        device_system = system.to(device=custom_device)

        torch.set_warn_always(True)
        message = (
            "Systems data is on device .* but featomic only supports calculations "
            "on CPU. All the data will be moved to CPU and then back on device on "
            "your behalf"
        )
        with pytest.warns(match=message):
            calculator.compute(device_system)

        message = "all systems should have the same device"
        with pytest.raises(TypeError, match=message):
            calculator.compute([device_system, system])


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


def can_use_mps_backend():
    return (
        # Github Actions M1 runners don't have a GPU accessible
        os.environ.get("GITHUB_ACTIONS") is None
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    )
