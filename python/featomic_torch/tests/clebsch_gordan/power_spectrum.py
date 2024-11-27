import torch
from metatensor.torch.atomistic import System
from packaging import version

from featomic.torch.calculators import SphericalExpansion
from featomic.torch.clebsch_gordan import PowerSpectrum


def system():
    return System(
        types=torch.tensor([1, 1, 8, 8]),
        positions=torch.tensor([[0.0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]]),
        cell=torch.tensor([[10.0, 0, 0], [0, 10, 0], [0, 0, 10]]),
        pbc=torch.tensor([True, True, True]),
    )


def spherical_expansion_calculator():
    return SphericalExpansion(
        cutoff={
            "radius": 5.0,
            "smoothing": {"type": "ShiftedCosine", "width": 0.5},
        },
        density={
            "type": "Gaussian",
            "width": 0.3,
        },
        basis={
            "type": "TensorProduct",
            "max_angular": 4,
            "radial": {"type": "Gto", "max_radial": 5},
        },
    )


def check_operation(calculator):
    # this only runs basic checks functionality checks, and that the code produces
    # output with the right type

    descriptor = calculator.compute(system(), gradients=["positions"])

    assert isinstance(descriptor, torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert descriptor._type().name() == "TensorMap"


def test_operation_as_python():
    calculator = PowerSpectrum(spherical_expansion_calculator())

    check_operation(calculator)


def test_operation_as_torch_script():
    calculator = PowerSpectrum(spherical_expansion_calculator())
    scripted = torch.jit.script(calculator)
    check_operation(scripted)
