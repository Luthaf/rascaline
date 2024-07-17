import os
import warnings
from typing import List

import metatensor
import numpy as np
import pytest
from metatensor import Labels, TensorBlock, TensorMap

import rascaline
from rascaline.utils.clebsch_gordan import ClebschGordanProduct


# Try to import some modules
ase = pytest.importorskip("ase")
import ase.io  # noqa: E402, F811


try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

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


if HAS_TORCH:
    ARRAYS_BACKEND = ["numpy", "torch"]
else:
    ARRAYS_BACKEND = ["numpy"]


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


def test_keys_are_matched():
    """
    Tests that key dimensions named the same in two tensors are matched.
    """
    # Set up
    frames = h2o_isolated()
    calculator = ClebschGordanProduct(max_angular=MAX_ANGULAR * 2)

    # Compute lambda-SOAP
    density = spherical_expansion(frames)
    match_names = [
        name for name in density.keys.names if name not in ["o3_lambda", "o3_sigma"]
    ]
    lambda_soap = calculator.compute(
        metatensor.rename_dimension(density, "properties", "n", "n_1"),
        metatensor.rename_dimension(density, "properties", "n", "n_2"),
        o3_lambda_1_new_name="l_1",
        o3_lambda_2_new_name="l_2",
    )

    # Check that the keys are matched
    for name in match_names:
        assert name in lambda_soap.keys.names


def test_key_matching_versus_manual_property_matching():
    """
    Tests that the output tensor from these processes are equivalent:

    1. Computing SphericalExpansion, keeping "center_type" and "neighbor_type" in the
       keys so that they are matched, computing a CG tensor product (to give an
       equivariant power spectrum or "lambda-SOAP") then moving these keys to properties
    2. Moving both these keys to properties first, computing a full correlation of the
       over these properties, then manually doing the matching.

    Match keys are automatically detected as keys that have the same name in
    ``tensor_1`` and ``tensor_2``.
    """

    # Set up
    frames = h2o_isolated()
    calculator = ClebschGordanProduct(max_angular=MAX_ANGULAR * 2)
    selected_keys = Labels(names=["o3_lambda"], values=np.array([[1], [3]]))

    # Compute the density
    density = spherical_expansion(frames)

    # Compute the first lambda-SOAP with both "center_type" and "neighbor_type" in the
    # keys, so that they are matched. Then move "neighbor_type" to properties.
    lambda_soap_1 = calculator.compute(
        metatensor.rename_dimension(density, "properties", "n", "n_1"),
        metatensor.rename_dimension(density, "properties", "n", "n_2"),
        o3_lambda_1_new_name="l_1",
        o3_lambda_2_new_name="l_2",
        selected_keys=selected_keys,
    )
    lambda_soap_1 = lambda_soap_1.keys_to_properties("neighbor_type")

    # Compute the second lambda-SOAP with only "center_type" in the keys, so that only
    # "center_type" is matched.
    density = density.keys_to_properties("neighbor_type")
    lambda_soap_2 = calculator.compute(
        metatensor.rename_dimension(
            metatensor.rename_dimension(
                density, "properties", "neighbor_type", "neighbor_1_type"
            ),
            "properties",
            "n",
            "n_1",
        ),
        metatensor.rename_dimension(
            metatensor.rename_dimension(
                density, "properties", "neighbor_type", "neighbor_2_type"
            ),
            "properties",
            "n",
            "n_2",
        ),
        o3_lambda_1_new_name="l_1",
        o3_lambda_2_new_name="l_2",
        selected_keys=selected_keys,
    )

    # Now do manual matching by slicing the properties dimension of lsoap_2,
    # i.e. identifying where "neighbor_1_type_1" == "neighbor_2_type"
    lambda_soap_2 = metatensor.rename_dimension(
        lambda_soap_2, "properties", "neighbor_1_type", "neighbor_type"
    )

    lambda_soap_2 = metatensor.permute_dimensions(
        lambda_soap_2, "properties", [0, 2, 1, 3]
    )
    new_blocks = []
    for block in lambda_soap_2:
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
    lambda_soap_2 = TensorMap(lambda_soap_2.keys, new_blocks)

    # Check for equivalence. Sorting of metadata required here.
    assert metatensor.allclose(
        metatensor.sort(lambda_soap_1), metatensor.sort(lambda_soap_2)
    )


def test_error_same_name_properties():
    """
    Tests that a ValueError is raised if the properties of two tensors have the same
    name.
    """
    frames = h2o_isolated()
    calculator = ClebschGordanProduct(max_angular=MAX_ANGULAR * 2)
    density = spherical_expansion(frames)

    error_msg = (
        "property name `n` present in both input tensors."
        " As all property dimensions are combined, they must have"
        " different names in the two tensors. Use"
        " `metatensor.rename_dimension` and try again."
    )
    with pytest.raises(ValueError, match=error_msg):
        calculator.compute(
            density,
            density,
            o3_lambda_1_new_name="l_1",
            o3_lambda_2_new_name="l_2",
        )


def test_sufficient_max_angular_with_angular_cutoff():
    """
    Tests that the max_angular is sufficient to correlate the two tensors when not large
    enough to cover MAX_ANGULAR, but when an angular cutoff is applied.
    """
    frames = h2o_isolated()

    # max_angular to be twice as big here if not using an angular cutoff
    calculator = ClebschGordanProduct(max_angular=MAX_ANGULAR)
    density = spherical_expansion(frames)

    calculator.compute(  # no error
        metatensor.rename_dimension(density, "properties", "n", "n_1"),
        metatensor.rename_dimension(density, "properties", "n", "n_2"),
        o3_lambda_1_new_name="l_1",
        o3_lambda_2_new_name="l_2",
        selected_keys=Labels(
            names=["o3_lambda"],
            values=np.arange(MAX_ANGULAR + 1).reshape(-1, 1),
        ),
    )


def test_error_insufficient_max_angular():
    """
    Tests that a ValueError is raised if the max_angular is insufficient to correlate
    the two tensors.
    """
    frames = h2o_isolated()
    # max_angular to be twice as big here if not using an angular cutoff
    calculator = ClebschGordanProduct(max_angular=MAX_ANGULAR)
    density = spherical_expansion(frames)

    error_msg = (
        "the maximum angular momentum value found in key dimension"
        " `'o3_lambda'` in `selected_keys` exceeds `max_angular=3`"
        " used to calculate the CG coefficients in the constructor."
    )
    with pytest.raises(ValueError, match=error_msg):
        calculator.compute(
            metatensor.rename_dimension(density, "properties", "n", "n_1"),
            metatensor.rename_dimension(density, "properties", "n", "n_2"),
            o3_lambda_1_new_name="l_1",
            o3_lambda_2_new_name="l_2",
            selected_keys=Labels(
                names=["o3_lambda"],
                values=np.arange(MAX_ANGULAR + 2).reshape(-1, 1),
            ),
        )


def test_same_samples():
    """
    Tests that the samples of the ClebschGordanProduct.compute output is the same as
    both the input tensor with the most dimensions.
    """

    # Set up
    frames = h2o_isolated()
    calculator = ClebschGordanProduct(max_angular=MAX_ANGULAR * 2)

    # Compute the density
    density = spherical_expansion(frames)
    density = density.keys_to_properties("neighbor_type")
    density_1 = metatensor.rename_dimension(density, "properties", "n", "n_1")
    density_1 = metatensor.rename_dimension(
        density_1, "properties", "neighbor_type", "neighbor_1_type"
    )
    density_2 = metatensor.rename_dimension(density, "properties", "n", "n_2")
    density_2 = metatensor.rename_dimension(
        density_2, "properties", "neighbor_type", "neighbor_2_type"
    )

    for key in density_1.keys:
        assert density_1[key].samples == density_2[key].samples

    # Compute the lambda-SOAP
    new_o3_lambda_names = ["l_1", "l_2"]
    lsoap = calculator.compute(
        density_1,
        density_2,
        o3_lambda_1_new_name=new_o3_lambda_names[0],
        o3_lambda_2_new_name=new_o3_lambda_names[1],
    )

    # Move everything but "o3_lambda" and "center_type" to properties
    density_1 = density_1.keys_to_properties(["o3_sigma"])
    lsoap = lsoap.keys_to_properties(["o3_sigma"] + new_o3_lambda_names)

    for key in density_1.keys:
        assert lsoap[key].samples == density_1[key].samples


def test_different_samples():
    """
    Tests that the samples of the ClebschGordanProduct.compute output is the same as
    both the input tensor with the most dimensions.
    """

    # Set up
    frames = h2o_isolated()
    calculator = ClebschGordanProduct(max_angular=MAX_ANGULAR * 2)

    # Compute the density
    density = spherical_expansion(frames)
    density = metatensor.rename_dimension(
        density, "keys", "center_type", "first_atom_type"
    )
    density = metatensor.rename_dimension(
        density, "keys", "neighbor_type", "second_atom_type"
    )
    density = metatensor.rename_dimension(density, "samples", "atom", "first_atom")
    density = density.keys_to_properties("second_atom_type")
    density_2 = metatensor.rename_dimension(density, "properties", "n", "n_2")
    density_2 = metatensor.rename_dimension(
        density_2, "properties", "second_atom_type", "second_atom_2_type"
    )

    # Compute pair density
    pair_density = spherical_expansion_by_pair(frames)
    pair_density = pair_density.keys_to_properties("second_atom_type")
    density_1 = metatensor.rename_dimension(pair_density, "properties", "n", "n_1")
    density_1 = metatensor.rename_dimension(
        pair_density, "properties", "second_atom_type", "second_atom_1_type"
    )

    # Compute the power spectrum by pair
    new_o3_lambda_names = ["l_1", "l_2"]
    lsoap = calculator.compute(
        density_1,
        density_2,
        o3_lambda_1_new_name=new_o3_lambda_names[0],
        o3_lambda_2_new_name=new_o3_lambda_names[1],
    )

    # Move everything but "o3_lambda" and "center_type" to properties
    density_1 = metatensor.sort(density_1)
    lsoap = metatensor.sort(lsoap.keys_to_properties(new_o3_lambda_names))
    lsoap = metatensor.sum_over_samples(
        lsoap, ["second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"]
    )

    for i, key in enumerate(density_2.keys):
        assert lsoap[key].samples == density_2[key].samples, i


def can_use_mps_backend():
    import torch

    return (
        # Github Actions M1 runners don't have a GPU accessible
        os.environ.get("GITHUB_ACTIONS") is None
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    )


def available_dtype_devices():
    if not HAS_TORCH:
        return []

    options = [(torch.float32, "cpu"), (torch.float64, "cpu")]

    if can_use_mps_backend():
        options.append((torch.float32, "mps"))

    if torch.cuda.is_available():
        options.append((torch.float32, "cuda"))
        options.append((torch.float64, "cuda"))

    return options


@pytest.mark.skipif(not HAS_TORCH, reason="torch is not available")
@pytest.mark.parametrize("dtype, device", available_dtype_devices())
def test_device_dtype(dtype, device):
    warnings.filterwarnings(
        action="ignore",
        message="Blocks values and keys for this TensorMap are on different devices",
    )
    warnings.filterwarnings(
        action="ignore",
        message="Values and labels for this block are on different devices",
    )

    frames = h2o_isolated()
    density = spherical_expansion(frames).to(arrays="torch", dtype=dtype, device=device)

    calculator = ClebschGordanProduct(
        max_angular=6,
        arrays_backend="torch",
        dtype=dtype,
        device=device,
    )

    # just checking that the code runs without error
    calculator.compute(
        metatensor.rename_dimension(density, "properties", "n", "n_1"),
        metatensor.rename_dimension(density, "properties", "n", "n_2"),
        o3_lambda_1_new_name="l_1",
        o3_lambda_2_new_name="l_2",
    )
