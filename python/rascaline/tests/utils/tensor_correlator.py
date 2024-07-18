import os
from typing import List

import metatensor
import numpy as np
import pytest
from metatensor import Labels, TensorBlock, TensorMap

import rascaline
from rascaline.utils import _dispatch
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

MAX_ANGULAR = 3
SPHEX_HYPERS = {
    "cutoff": 3.0,  # Angstrom
    "max_radial": 3,  # Exclusive
    "max_angular": MAX_ANGULAR,  # Inclusive
    "atomic_gaussian_width": 0.3,
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


def spherical_expansion_by_pair(frames: List[ase.Atoms]):
    """Returns a rascaline SphericalExpansionByPair"""
    calculator = rascaline.SphericalExpansionByPair(**SPHEX_HYPERS)
    return calculator.compute(frames)


def tensor_correlator():
    """
    Constructs a tensor correlator with required CG coefficients to perform a single CG
    tensor product of two densities computed with MAX_ANGULAR.
    """
    return TensorCorrelator(max_angular=MAX_ANGULAR * 2)


def test_keys_are_matched():
    """
    Tests that key dimensions named the same in two tensors are matched.
    """
    # Set up
    frames = h2o_isolated()
    calculator = tensor_correlator()

    # Compute lambda-SOAP
    density = spherical_expansion(frames)
    match_names = [
        name for name in density.keys.names if name not in ["o3_lambda", "o3_sigma"]
    ]
    lsoap = calculator.compute(
        _utils._increment_property_name_suffices(density, 1),
        _utils._increment_property_name_suffices(density, 2),
        o3_lambda_1_name="l_1",
        o3_lambda_2_name="l_2",
    )

    # Check that the keys are matched
    for name in match_names:
        assert name in lsoap.keys.names


def test_key_matching_versus_manual_property_matching():
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
    lsoap_1 = calculator.compute(
        _utils._increment_property_name_suffices(density, 1),
        _utils._increment_property_name_suffices(density, 2),
        o3_lambda_1_name="l_1",
        o3_lambda_2_name="l_2",
        selected_keys=selected_keys,
    )
    lsoap_1 = lsoap_1.keys_to_properties("neighbor_type")

    # Compute the second lambda-SOAP with only "center_type" in the keys, so that only
    # "center_type" is matched.
    density = density.keys_to_properties("neighbor_type")
    lsoap_2 = calculator.compute(
        _utils._increment_property_name_suffices(density, 1),
        _utils._increment_property_name_suffices(density, 2),
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


def test_error_same_name_properties():
    """
    Tests that a ValueError is raised if the properties of two tensors have the same
    name.
    """
    frames = h2o_isolated()
    calculator = tensor_correlator()
    density = spherical_expansion(frames)

    with pytest.raises(ValueError):
        calculator.compute(
            density,
            density,
            o3_lambda_1_name="l_1",
            o3_lambda_2_name="l_2",
        )


def test_sufficient_max_angular_with_angular_cutoff():
    """
    Tests that the max_angular is sufficient to correlate the two tensors when not large
    enough to cover MAX_ANGULAR, but when an angular cutoff is applied.
    """
    frames = h2o_isolated()

    # max_angular to be twice as big here if not using an angular cutoff
    calculator = TensorCorrelator(
        max_angular=MAX_ANGULAR,
    )
    density = spherical_expansion(frames)
    calculator.compute(  # no error
        _utils._increment_property_name_suffices(density, 1),
        _utils._increment_property_name_suffices(density, 2),
        o3_lambda_1_name="l_1",
        o3_lambda_2_name="l_2",
        angular_cutoff=MAX_ANGULAR,
    )


def test_error_insufficient_max_angular():
    """
    Tests that a ValueError is raised if the max_angular is insufficient to correlate
    the two tensors.
    """
    frames = h2o_isolated()
    # max_angular to be twice as big here if not using an angular cutoff
    calculator = TensorCorrelator(max_angular=MAX_ANGULAR)
    density = spherical_expansion(frames)

    with pytest.raises(ValueError):
        calculator.compute(
            _utils._increment_property_name_suffices(density, 1),
            _utils._increment_property_name_suffices(density, 2),
            o3_lambda_1_name="l_1",
            o3_lambda_2_name="l_2",
            angular_cutoff=None,
        )


def test_correlator_same_samples():
    """
    Tests that the samples of the TensorCorrelator.compute output is the same as both
    the input tensor with the most dimensions.
    """

    # Set up
    frames = h2o_isolated()
    calculator = tensor_correlator()

    # Compute the density
    density = spherical_expansion(frames)
    density = density.keys_to_properties("neighbor_type")
    density_1 = _utils._increment_property_name_suffices(density, 1)
    density_2 = _utils._increment_property_name_suffices(density, 2)

    for key in density_1.keys:
        assert density_1[key].samples == density_2[key].samples

    # Compute the lambda-SOAP
    new_o3_lambda_names = ["l_1", "l_2"]
    lsoap = calculator.compute(
        density_1,
        density_2,
        o3_lambda_1_name=new_o3_lambda_names[0],
        o3_lambda_2_name=new_o3_lambda_names[1],
        angular_cutoff=MAX_ANGULAR,
    )

    # Move everything but "o3_lambda" and "center_type" to properties
    density_1 = density_1.keys_to_properties(["o3_sigma"])
    lsoap = lsoap.keys_to_properties(["o3_sigma"] + new_o3_lambda_names)

    for key in lsoap.keys:
        assert lsoap[key].samples == density_1[key].samples


def test_correlator_different_samples():
    """
    Tests that the samples of the TensorCorrelator.compute output is the same as both
    the input tensor with the most dimensions.
    """

    # Set up
    frames = h2o_isolated()
    calculator = tensor_correlator()

    # Compute the density
    density = spherical_expansion(frames)
    density = density.keys_to_properties("neighbor_type")
    density = _utils._increment_property_name_suffices(density, 1)
    density_2 = _utils._increment_property_name_suffices(density, 2)

    for key in density_1.keys:
        assert density_1[key].samples == density_2[key].samples

    # Compute the lambda-SOAP
    new_o3_lambda_names = ["l_1", "l_2"]
    lsoap = calculator.compute(
        density_1,
        density_2,
        o3_lambda_1_name=new_o3_lambda_names[0],
        o3_lambda_2_name=new_o3_lambda_names[1],
        angular_cutoff=MAX_ANGULAR,
    )

    # Move everything but "o3_lambda" and "center_type" to properties
    density_1 = density_1.keys_to_properties(["o3_sigma"])
    lsoap = lsoap.keys_to_properties(["o3_sigma"] + new_o3_lambda_names)

    for key in lsoap.keys:
        assert lsoap[key].samples == density_1[key].samples