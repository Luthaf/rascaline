# -*- coding: utf-8 -*-
import numpy as np
import pytest
from numpy.testing import assert_allclose

import rascaline
from rascaline.utils import PowerSpectrum

from ..test_systems import SystemForTests
from .test_utils import finite_differences_positions


ase = pytest.importorskip("ase")


HYPERS = hypers = {
    "cutoff": 5.0,
    "max_radial": 6,
    "max_angular": 4,
    "atomic_gaussian_width": 0.3,
    "center_atom_weight": 1.0,
    "radial_basis": {
        "Gto": {},
    },
    "cutoff_function": {
        "ShiftedCosine": {"width": 0.5},
    },
}
N_SPECIES = len(np.unique(SystemForTests().species()))


def soap_calculator():
    return rascaline.SphericalExpansion(**HYPERS)


def lode_calculator():
    hypers = HYPERS.copy()
    hypers.pop("cutoff_function")
    hypers["potential_exponent"] = 1

    return rascaline.LodeSphericalExpansion(**hypers)


def soap():
    return soap_calculator().compute(SystemForTests())


def power_spectrum():
    return PowerSpectrum(soap_calculator()).compute(SystemForTests())


@pytest.mark.parametrize("calculator", [soap_calculator(), lode_calculator()])
def test_power_spectrum(calculator) -> None:
    """Test that power spectrum works and that the shape is correct."""
    ps_python = PowerSpectrum(calculator).compute(SystemForTests())
    ps_python = ps_python.keys_to_samples(["species_center"])

    # Test the number of properties is correct
    n_props_actual = len(ps_python.block().properties)

    n_props_expected = (
        N_SPECIES**2 * HYPERS["max_radial"] ** 2 * (HYPERS["max_angular"] + 1)
    )

    assert n_props_actual == n_props_expected


def test_error_max_angular():
    """Test error raise if max_angular are different."""
    hypers_2 = HYPERS.copy()
    hypers_2.update(max_radial=3, max_angular=1)

    se_calculator2 = rascaline.SphericalExpansion(**hypers_2)

    msg = "'max_angular' of both calculators must be the same!"
    with pytest.raises(ValueError, match=msg):
        PowerSpectrum(soap_calculator(), se_calculator2)

    calculator = rascaline.SoapPowerSpectrum(**HYPERS)
    with pytest.raises(ValueError, match="are supported for calculator_1"):
        PowerSpectrum(calculator)


def test_wrong_calculator_1():
    """Test error raise for wrong calculator_1."""

    calculator = rascaline.SoapPowerSpectrum(**HYPERS)
    with pytest.raises(ValueError, match="are supported for calculator_1"):
        PowerSpectrum(calculator)


def test_wrong_calculator_2():
    """Test error raise for wrong calculator_2."""

    calculator = rascaline.SoapPowerSpectrum(**HYPERS)
    with pytest.raises(ValueError, match="are supported for calculator_2"):
        PowerSpectrum(soap_calculator(), calculator)


def test_power_spectrum_different_hypers() -> None:
    """Test that power spectrum works with different spherical expansions."""

    hypers_2 = HYPERS.copy()
    hypers_2.update(max_radial=3, max_angular=4)

    se_calculator2 = rascaline.SphericalExpansion(**hypers_2)

    PowerSpectrum(soap_calculator(), se_calculator2).compute(SystemForTests())


def test_power_spectrum_rust() -> None:
    """Test that the dot kernels of the rust and python version are the same."""

    power_spectrum_python = power_spectrum()
    power_spectrum_python = power_spectrum_python.keys_to_samples(["species_center"])
    kernel_python = np.dot(
        power_spectrum_python[0].values, power_spectrum_python[0].values.T
    )

    power_spectrum_rust = rascaline.SoapPowerSpectrum(**HYPERS).compute(
        SystemForTests()
    )
    power_spectrum_rust = power_spectrum_rust.keys_to_samples(["species_center"])
    power_spectrum_rust = power_spectrum_rust.keys_to_properties(
        ["species_neighbor_1", "species_neighbor_2"]
    )
    kernel_rust = np.dot(power_spectrum_rust[0].values, power_spectrum_rust[0].values.T)
    assert_allclose(kernel_python, kernel_rust)


def test_power_spectrum_gradients() -> None:
    """Test that gradients are correct using finite differences."""
    calculator = PowerSpectrum(soap_calculator())

    # An ASE atoms object with the same properties as SystemForTests()
    atoms = ase.Atoms(
        symbols="HHOO",
        positions=[[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]],
        pbc=True,
        cell=[[10, 0, 0], [0, 10, 0], [0, 0, 10]],
    )

    finite_differences_positions(calculator, atoms)


def test_power_spectrum_unknown_gradient() -> None:
    """Test error raise if an unknown gradient is present."""

    calculator = rascaline.SphericalExpansion(**HYPERS)

    msg = "PowerSpectrum currently only supports gradients w.r.t. to positions"
    with pytest.raises(NotImplementedError, match=msg):
        PowerSpectrum(calculator).compute(SystemForTests(), gradients=["cell"])


def test_fill_species_neighbor() -> None:
    """Test that ``species_center`` keys can be merged for different blocks."""

    frames = [
        ase.Atoms("H", positions=np.zeros([1, 3])),
        ase.Atoms("O", positions=np.zeros([1, 3])),
    ]

    calculator = PowerSpectrum(
        calculator_1=rascaline.SphericalExpansion(**HYPERS),
        calculator_2=rascaline.SphericalExpansion(**HYPERS),
    )

    descriptor = calculator.compute(frames)

    descriptor.keys_to_samples("species_center")
