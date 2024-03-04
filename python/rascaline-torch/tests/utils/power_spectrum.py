import torch
from metatensor.torch.atomistic import System
from packaging import version

from rascaline.torch.calculators import SphericalExpansion
from rascaline.torch.utils import PowerSpectrum


def system():
    return System(
        types=torch.tensor([1, 1, 8, 8]),
        positions=torch.tensor([[0.0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]]),
        cell=torch.tensor([[10.0, 0, 0], [0, 10, 0], [0, 0, 10]]),
    )


def spherical_expansion_calculator():
    return SphericalExpansion(
        cutoff=5.0,
        max_radial=6,
        max_angular=4,
        atomic_gaussian_width=0.3,
        center_atom_weight=1.0,
        radial_basis={
            "Gto": {},
        },
        cutoff_function={
            "ShiftedCosine": {"width": 0.5},
        },
    )


def test_forward() -> None:
    """Test that forward results in the same as compute."""
    ps_compute = PowerSpectrum(spherical_expansion_calculator()).compute(system())
    ps_forward = PowerSpectrum(spherical_expansion_calculator()).forward(system())

    assert ps_compute.keys == ps_forward.keys


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
