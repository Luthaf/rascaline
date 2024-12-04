import copy

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

import featomic
from featomic.clebsch_gordan import PowerSpectrum

from ..test_systems import SystemForTests


ase = pytest.importorskip("ase")


SOAP_HYPERS = {
    "cutoff": {
        "radius": 5.0,
        "smoothing": {"type": "ShiftedCosine", "width": 0.5},
    },
    "density": {
        "type": "Gaussian",
        "width": 0.3,
    },
    "basis": {
        "type": "TensorProduct",
        "max_angular": 4,
        "radial": {"type": "Gto", "max_radial": 6},
    },
}

LODE_HYPERS = {
    "density": {
        "type": "SmearedPowerLaw",
        "smearing": 0.3,
        "exponent": 1,
    },
    "basis": {
        "type": "TensorProduct",
        "max_angular": 4,
        "radial": {
            "type": "Gto",
            "max_radial": 6,
            "radius": 5.0,
        },
    },
}

N_ATOMIC_TYPES = len(np.unique(SystemForTests().types()))


def soap_spx():
    return featomic.SphericalExpansion(**SOAP_HYPERS)


def soap_ps():
    return featomic.SoapPowerSpectrum(**SOAP_HYPERS)


def lode_spx():
    return featomic.LodeSphericalExpansion(**LODE_HYPERS)


def soap():
    return soap_spx().compute(SystemForTests())


def power_spectrum():
    return PowerSpectrum(soap_spx()).compute(SystemForTests())


@pytest.mark.parametrize("calculator", [soap_spx(), lode_spx()])
def test_power_spectrum(calculator) -> None:
    """Test that power spectrum works and that the shape is correct."""
    ps_python = PowerSpectrum(calculator).compute(SystemForTests())
    ps_python = ps_python.keys_to_samples(["center_type"])

    # Test the number of properties is correct
    n_props_actual = len(ps_python.block().properties)

    n_props_expected = (
        N_ATOMIC_TYPES**2
        * (SOAP_HYPERS["basis"]["radial"]["max_radial"] + 1) ** 2
        * (SOAP_HYPERS["basis"]["max_angular"] + 1)
    )

    assert n_props_actual == n_props_expected


def test_error_max_angular():
    """Test error raise if max_angular are different."""
    hypers_2 = copy.deepcopy(SOAP_HYPERS)
    hypers_2["basis"]["radial"]["max_radial"] = 3
    hypers_2["basis"]["max_angular"] = 2

    se_calculator2 = featomic.SphericalExpansion(**hypers_2)

    message = "'basis.max_angular' must be the same in both calculators"
    with pytest.raises(ValueError, match=message):
        PowerSpectrum(soap_spx(), se_calculator2)


def test_wrong_calculator():
    message = (
        "Only \\[lode_spherical_expansion, spherical_expansion\\] "
        "are supported for `calculator_1`, got 'soap_power_spectrum'"
    )
    with pytest.raises(ValueError, match=message):
        PowerSpectrum(soap_ps())

    message = (
        "Only \\[lode_spherical_expansion, spherical_expansion\\] "
        "are supported for `calculator_2`, got 'soap_power_spectrum'"
    )
    with pytest.raises(ValueError, match=message):
        PowerSpectrum(soap_spx(), soap_ps())


def test_power_spectrum_different_hypers() -> None:
    """Test that power spectrum works with different spherical expansions."""

    hypers_2 = copy.deepcopy(SOAP_HYPERS)
    hypers_2["basis"]["radial"]["max_radial"] = 3

    soap_spx_2 = featomic.SphericalExpansion(**hypers_2)

    PowerSpectrum(soap_spx(), soap_spx_2).compute(SystemForTests())


def test_power_spectrum_rust() -> None:
    """Test that the dot kernels of the rust and python version are the same."""

    power_spectrum_python = power_spectrum()
    power_spectrum_python = power_spectrum_python.keys_to_samples(["center_type"])
    kernel_python = np.dot(
        power_spectrum_python[0].values, power_spectrum_python[0].values.T
    )

    power_spectrum_rust = featomic.SoapPowerSpectrum(**SOAP_HYPERS).compute(
        SystemForTests()
    )
    power_spectrum_rust = power_spectrum_rust.keys_to_samples(["center_type"])
    power_spectrum_rust = power_spectrum_rust.keys_to_properties(
        ["neighbor_1_type", "neighbor_2_type"]
    )
    kernel_rust = np.dot(power_spectrum_rust[0].values, power_spectrum_rust[0].values.T)
    assert_allclose(kernel_python, kernel_rust)


def test_power_spectrum_gradients() -> None:
    """Test that gradients are correct using finite differences."""
    calculator = PowerSpectrum(soap_spx())

    # An ASE atoms object with the same properties as SystemForTests()
    atoms = ase.Atoms(
        symbols="HHOO",
        positions=[[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]],
        pbc=True,
        cell=[[10, 0, 0], [0, 10, 0], [0, 0, 10]],
    )

    _finite_differences_positions(calculator, atoms)


def test_power_spectrum_unknown_gradient() -> None:
    """Test error raise if an unknown gradient is present."""

    message = "PowerSpectrum currently only supports gradients w.r.t. to positions"
    with pytest.raises(NotImplementedError, match=message):
        PowerSpectrum(soap_spx()).compute(SystemForTests(), gradients=["strain"])


def test_fill_neighbor_type() -> None:
    """Test that ``center_type`` keys can be merged for different blocks."""

    frames = [
        ase.Atoms("H", positions=np.zeros([1, 3])),
        ase.Atoms("O", positions=np.zeros([1, 3])),
    ]

    calculator = PowerSpectrum(
        calculator_1=soap_spx(),
        calculator_2=soap_spx(),
    )

    descriptor = calculator.compute(frames)

    descriptor.keys_to_samples("center_type")


def test_fill_types_option() -> None:
    """Test that ``types`` options adds arbitrary atomic types."""

    frames = [
        ase.Atoms("H", positions=np.zeros([1, 3])),
        ase.Atoms("O", positions=np.zeros([1, 3])),
    ]

    types = [1, 8, 10]
    calculator = PowerSpectrum(calculator_1=soap_spx(), types=types)

    descriptor = calculator.compute(frames)

    assert_equal(np.unique(descriptor[0].properties["neighbor_1_type"]), types)
    assert_equal(np.unique(descriptor[0].properties["neighbor_2_type"]), types)


def _finite_differences_positions(
    calculator,
    system,
    displacement=1e-6,
    rtol=1e-5,
    atol=1e-16,
):
    """
    Check that analytical gradients with respect to positions agree with a finite
    difference calculation of the gradients.

    The implementation is simular to ``featomic/src/calculators/tests_utils.rs``.

    :param calculator: calculator used to compute the representation
    :param system: Atoms object
    :param displacement: distance each atom will be displaced in each direction when
        computing finite differences
    :param max_relative: Maximal relative error. ``10 * displacement`` is a good
        starting point
    :param atol: Threshold below which all values are considered zero. This should be
        very small (1e-16) to prevent false positives (if all values & gradients are
        below that threshold, tests will pass even with wrong gradients)
    :raises AssertionError: if the two gradients are not equal up to specified precision
    """
    reference = calculator.compute(system, gradients=["positions"])

    for atom_i in range(len(system)):
        for xyz in range(3):
            system_pos = system.copy()
            system_pos.positions[atom_i, xyz] += displacement / 2
            updated_pos = calculator.compute(system_pos)

            system_neg = system.copy()
            system_neg.positions[atom_i, xyz] -= displacement / 2
            updated_neg = calculator.compute(system_neg)

            assert updated_pos.keys == reference.keys
            assert updated_neg.keys == reference.keys

            for key, block in reference.items():
                gradients = block.gradient("positions")

                block_pos = updated_pos.block(key)
                block_neg = updated_neg.block(key)

                for gradient_i, (sample_i, _, atom) in enumerate(gradients.samples):
                    if atom != atom_i:
                        continue

                    # check that the sample is the same in both descriptors
                    assert block_pos.samples[sample_i] == block.samples[sample_i]
                    assert block_neg.samples[sample_i] == block.samples[sample_i]

                    value_pos = block_pos.values[sample_i]
                    value_neg = block_neg.values[sample_i]
                    gradient = gradients.values[gradient_i, xyz]

                    assert value_pos.shape == gradient.shape
                    assert value_neg.shape == gradient.shape

                    finite_difference = (value_pos - value_neg) / displacement

                    assert_allclose(finite_difference, gradient, rtol=rtol, atol=atol)
