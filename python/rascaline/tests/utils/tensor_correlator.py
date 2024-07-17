import os
from typing import List

import metatensor
import numpy as np
import pytest
from metatensor import Labels, TensorBlock, TensorMap

import rascaline
from rascaline.utils.clebsch_gordan import TensorCorrelator, _utils


# Try to import some modules
ase = pytest.importorskip("ase")
import ase.io  # noqa: E402


try:
    import metatensor.operations

    HAS_METATENSOR_OPERATIONS = True
except ImportError:
    HAS_METATENSOR_OPERATIONS = False
try:
    import sympy  # noqa: F401

    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False


try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    ARRAYS_BACKEND = ["numpy", "torch"]
else:
    ARRAYS_BACKEND = ["numpy"]

DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")

SPHEX_HYPERS = {
    "cutoff": 2.5,  # Angstrom
    "max_radial": 3,  # Exclusive
    "max_angular": 3,  # Inclusive
    "atomic_gaussian_width": 0.2,
    "radial_basis": {"Gto": {}},
    "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
    "center_atom_weight": 1.0,
}

SPHEX_HYPERS_SMALL = {
    "cutoff": 2.5,  # Angstrom
    "max_radial": 1,  # Exclusive
    "max_angular": 2,  # Inclusive
    "atomic_gaussian_width": 0.2,
    "radial_basis": {"Gto": {}},
    "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
    "center_atom_weight": 1.0,
}


def h2o_isolated():
    return [
        ase.Atoms(
            symbols=["O", "H", "H"],
            positions=[
                [2.56633400, 2.50000000, 2.50370100],
                [1.97361700, 1.73067300, 2.47063400],
                [1.97361700, 3.26932700, 2.47063400],
            ],
        )
    ]


def spherical_expansion(frames: List[ase.Atoms]):
    """Returns a rascaline SphericalExpansion"""
    calculator = rascaline.SphericalExpansion(**SPHEX_HYPERS)
    return calculator.compute(frames)


def spherical_expansion_small(frames: List[ase.Atoms]):
    """Returns a rascaline SphericalExpansion"""
    calculator = rascaline.SphericalExpansion(**SPHEX_HYPERS_SMALL)
    return calculator.compute(frames)


def tensor_correlator():
    """
    Constructs a tensor correlator with required CG coefficients to correlate two density
    produced with the SPHEX_HYPERS_SMALL hypers.
    """
    return TensorCorrelator(
        max_angular=SPHEX_HYPERS["max_angular"],
    )

def tensor_correlator_samll():
    """
    Constructs a tensor correlator with required CG coefficients to correlate two density
    produced with the SPHEX_HYPERS_SMALL hypers.
    """
    return TensorCorrelator(
        max_angular=SPHEX_HYPERS_SMALL["max_angular"],
    )


def test_match_keys():
    """
    Tests that the output tensor from these processes are equivalent:

    1. Computing SphericalExpansion, keeping "center_type" and "neighbor_type" in the
       keys so that they are matched, computing a CG tensor product (to give an
       equivariant power spectrum or "lambda-SOAP") then moving these keys to properties
    2. Moving both these keys to properties first, computing a full correlation of the
       over these properties, then maually doing the matching.

    Match keys are automatically detected as keys that have the same name in
    ``tensor_1`` and ``tensor_2``.
    """

    # Set up
    frames = h2o_isolated()
    calculator = tensor_correlator()
    selected_keys = Labels(names=["o3_lambda"], values=np.array([[1], [3]]))

    # Compute the density
    density = spherical_expansion(frames)

    # Compute the first lambda-SOAP with both "center_type" and "neighbor_type" in the
    # keys, so that they are matched. Then move "neighbor_type" to properties.
    density_1 = _utils._increment_property_name_suffices(density, 1)
    density_2 = _utils._increment_property_name_suffices(density_1, 1)
    lsoap_1 = calculator.compute(
        density_1,
        density_2,
        o3_lambda_1_name="l_1",
        o3_lambda_2_name="l_2",
        selected_keys=selected_keys,
    )
    lsoap_1 = lsoap_1.keys_to_properties("neighbor_type")

    # Compute the second lambda-SOAP with only "center_type" in the keys, so that only
    # "center_type" is matched.
    density = density.keys_to_properties("neighbor_type")
    density_1 = _utils._increment_property_name_suffices(density, 1)
    density_2 = _utils._increment_property_name_suffices(density_1, 1)
    lsoap_2 = calculator.compute(
        density_1,
        density_2,
        o3_lambda_1_name="l_1",
        o3_lambda_2_name="l_2",
        selected_keys=selected_keys,
    )

    # Now do manual matching by slicing the properties dimension of lsoap_2,
    # i.e. identifying where "neighbor_1_type_1" == "neighbor_2_type"
    lsoap_2 = metatensor.rename_dimension(
        lsoap_2, "properties", "neighbor_1_type", "neighbor_type"
    )
    print(lsoap_2, lsoap_2[0])

    lsoap_2 = metatensor.permute_dimensions(lsoap_2, "properties", [0, 2, 1, 3])
    new_blocks = []
    for block in lsoap_2:
        properties_filter = block.properties.column(
            "neighbor_type"
        ) == block.properties.column("neighbor_2_type")
        new_properties = Labels(
            names=block.properties.names,
            values=block.properties.values[properties_filter],
        )
        new_properties = new_properties.remove("neighbor_2_type")
        new_blocks.append(
            TensorBlock(
                values=block.values[:, :, properties_filter],
                samples=block.samples,
                components=block.components,
                properties=new_properties,
            )
        )
    lsoap_2 = TensorMap(lsoap_2.keys, new_blocks)

    # Check for equivalence. Sorting of metadata required here.
    assert metatensor.allclose(metatensor.sort(lsoap_1), metatensor.sort(lsoap_2))
