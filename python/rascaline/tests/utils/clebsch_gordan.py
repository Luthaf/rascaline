# -*- coding: utf-8 -*-
import os
from typing import List

import ase.io
import numpy as np
import pytest
from numpy.testing import assert_allclose

import metatensor
import rascaline
from metatensor import Labels, TensorBlock, TensorMap
from rascaline.utils import clebsch_gordan, PowerSpectrum


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")

RASCAL_HYPERS = {
    "cutoff": 3.0,  # Angstrom
    "max_radial": 6,  # Exclusive
    "max_angular": 5,  # Inclusive
    "atomic_gaussian_width": 0.2,
    "radial_basis": {"Gto": {}},
    "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
    "center_atom_weight": 1.0,
}

RASCAL_HYPERS_SMALL = {
    "cutoff": 3.0,  # Angstrom
    "max_radial": 1,  # Exclusive
    "max_angular": 2,  # Inclusive
    "atomic_gaussian_width": 0.2,
    "radial_basis": {"Gto": {}},
    "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
    "center_atom_weight": 1.0,
}


# ============ Pytest fixtures ============

# ============ Helper functions ============


def h2_isolated():
    return ase.io.read(os.path.join(DATA_ROOT, "h2_isolated.xyz"), ":")


def h2o_isolated():
    return ase.io.read(os.path.join(DATA_ROOT, "h2o_isolated.xyz"), ":")


def h2o_periodic():
    return ase.io.read(os.path.join(DATA_ROOT, "h2o_periodic.xyz"), ":")


def wigners(lmax: int):
    return clebsch_gordan.WignerDReal(lmax=lmax)


def sphex(frames: List[ase.Atoms]):
    """Returns a rascaline SphericalExpansion"""
    calculator = rascaline.SphericalExpansion(**RASCAL_HYPERS)
    return calculator.compute(frames)


def sphex_small_features(frames: List[ase.Atoms]):
    """Returns a rascaline SphericalExpansion"""
    calculator = rascaline.SphericalExpansion(**RASCAL_HYPERS_SMALL)
    return calculator.compute(frames)


def powspec(frames: List[ase.Atoms]):
    """Returns a rascaline PowerSpectrum constructed from a
    SphericalExpansion"""
    return PowerSpectrum(rascaline.SphericalExpansion(**RASCAL_HYPERS)).compute(frames)


def powspec_small_features(frames: List[ase.Atoms]):
    """Returns a rascaline PowerSpectrum constructed from a
    SphericalExpansion"""
    return PowerSpectrum(rascaline.SphericalExpansion(**RASCAL_HYPERS_SMALL)).compute(
        frames
    )


# ============ Test equivariance ============


@pytest.mark.parametrize(
    "frames, nu_target, angular_cutoff, angular_selection, parity_selection",
    [
        (
            h2o_isolated(),
            2,
            5,
            [0],  # [0, 4, 5],
            [+1],  # [+1, -1],
        ),
        (
            h2o_isolated(),
            2,
            5,
            None,  # [0, 4, 5],
            None,  # [+1, -1],
        ),
        (
            h2o_periodic(),
            2,
            5,
            None,  # [0, 1, 2, 3, 4, 5],
            None,  # [-1],
        ),
    ],
)
def test_so3_equivariance(
    frames: List[ase.Atoms],
    nu_target: int,
    angular_cutoff: int,
    angular_selection: List[List[int]],
    parity_selection: List[List[int]],
):
    wig = wigners(nu_target * RASCAL_HYPERS["max_angular"])
    frames_so3 = [
        clebsch_gordan.transform_frame_so3(frame, wig.angles) for frame in frames
    ]

    nu_1 = sphex(frames)
    nu_1_so3 = sphex(frames_so3)

    nu_3 = clebsch_gordan.combine_single_center_to_body_order(
        nu_1_tensor=nu_1,
        target_body_order=nu_target,
        angular_cutoff=angular_cutoff,
        angular_selection=angular_selection,
        parity_selection=parity_selection,
    )
    nu_3_so3 = clebsch_gordan.combine_single_center_to_body_order(
        nu_1_tensor=nu_1_so3,
        target_body_order=nu_target,
        angular_cutoff=angular_cutoff,
        angular_selection=angular_selection,
        parity_selection=parity_selection,
    )

    nu_3_transf = wig.transform_tensormap_so3(nu_3)

    # assert metatensor.equal_metadata(nu_3_transf, nu_3_rot)
    assert metatensor.allclose(nu_3_transf, nu_3_so3)


@pytest.mark.parametrize(
    "frames, nu_target, angular_cutoff, angular_selection, parity_selection",
    [
        (
            h2o_isolated(),
            2,
            5,
            [0],  # [0, 4, 5],
            [+1],  # [+1, -1],
        ),
        (
            h2o_isolated(),
            2,
            5,
            None,  # [0, 4, 5],
            None,  # [+1, -1],
        ),
        (
            h2o_periodic(),
            2,
            5,
            None,  # [0, 1, 2, 3, 4, 5],
            None,  # [-1],
        ),
    ],
)
def test_o3_equivariance(
    frames: List[ase.Atoms],
    nu_target: int,
    angular_cutoff: int,
    angular_selection: List[List[int]],
    parity_selection: List[List[int]],
):
    wig = wigners(nu_target * RASCAL_HYPERS["max_angular"])
    frames_o3 = [
        clebsch_gordan.transform_frame_o3(frame, wig.angles) for frame in frames
    ]

    nu_1 = sphex(frames)
    nu_1_o3 = sphex(frames_o3)

    nu_3 = clebsch_gordan.combine_single_center_to_body_order(
        nu_1_tensor=nu_1,
        target_body_order=nu_target,
        angular_cutoff=angular_cutoff,
        angular_selection=angular_selection,
        parity_selection=parity_selection,
    )
    nu_3_o3 = clebsch_gordan.combine_single_center_to_body_order(
        nu_1_tensor=nu_1_o3,
        target_body_order=nu_target,
        angular_cutoff=angular_cutoff,
        angular_selection=angular_selection,
        parity_selection=parity_selection,
    )

    nu_3_transf = wig.transform_tensormap_o3(nu_3)

    # assert metatensor.equal_metadata(nu_3_transf, nu_3_rot)
    assert metatensor.allclose(nu_3_transf, nu_3_o3)


# ============ Test lambda-SOAP vs PowerSpectrum ============


@pytest.mark.parametrize("frames", [h2_isolated()])
def test_lambda_soap_vs_powerspectrum(frames):
    """
    Tests for exact equivalence between the invariant block of a generated
    lambda-SOAP equivariant and the Python implementation of PowerSpectrum in
    rascaline utils.
    """
    # Build a PowerSpectrum
    ps = powspec_small_features(frames)

    # Build a lambda-SOAP
    nu_1_tensor = sphex_small_features(frames)
    lsoap = clebsch_gordan.lambda_soap_vector(
        nu_1_tensor=nu_1_tensor,
        angular_selection=[0],
    )
    keys = lsoap.keys.remove(name="spherical_harmonics_l")
    lsoap = TensorMap(keys=keys, blocks=[b.copy() for b in lsoap.blocks()])

    # Manipulate metadata to match that of PowerSpectrum:
    # 1) remove components axis
    # 2) change "l1" and "l2" properties dimensions to just "l" (as l1 == l2)
    blocks = []
    for block in lsoap.blocks():
        n_samples, n_props = block.values.shape[0], block.values.shape[2]
        new_props = block.properties
        new_props = new_props.remove(name="l1")
        new_props = new_props.rename(old="l2", new="l")
        blocks.append(
            TensorBlock(
                values=block.values.reshape((n_samples, n_props)),
                samples=block.samples,
                components=[],
                properties=new_props,
            )
        )
    lsoap = TensorMap(keys=lsoap.keys, blocks=blocks)

    # Compare metadata
    assert metatensor.equal_metadata(lsoap, ps)

    # allclose on values
    assert metatensor.allclose(lsoap, ps)


# ============ Test norm preservation  ============

@pytest.mark.parametrize("frames", [h2o_isolated()])
@pytest.mark.parametrize("max_angular", [1])
@pytest.mark.parametrize("nu", [2, 3])
def test_combine_single_center_orthogonality(frames, max_angular, nu):
    """
    Checks \|ρ^\\nu\| =  \|ρ\|^\\nu
    For \\nu = 2 the tests passes but for \\nu = 3 it fails because we do not add the
    multiplicity when iterating multiple body-orders
    """
    rascal_hypers = {
        "cutoff": 3.0,  # Angstrom
        "max_radial": 6,  # Exclusive
        "max_angular": max_angular,  # Inclusive
        "atomic_gaussian_width": 0.2,
        "radial_basis": {"Gto": {}},
        "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
        "center_atom_weight": 1.0,
    }

    calculator = rascaline.SphericalExpansion(**rascal_hypers)
    nu1_tensor = calculator.compute(frames)

    # compute norm of the body order 2 tensor
    nu_tensor = clebsch_gordan.combine_single_center_to_body_order(
        nu1_tensor,
        nu,  # target_body_order
        angular_cutoff=None,
        angular_selection=None,
        parity_selection=None,
        use_sparse=True,
    )
    nu_tensor = nu_tensor.keys_to_properties(["inversion_sigma", "order_nu"])
    nu_tensor = nu_tensor.keys_to_samples(["species_center"])
    n_samples = nu_tensor[0].values.shape[0]

    #  compute norm of the body order 1 tensor
    nu1_tensor = nu1_tensor.keys_to_properties(["species_neighbor"])
    nu1_tensor = nu1_tensor.keys_to_samples(["species_center"])

    nu_tensor_values = np.hstack(
        [
            nu_tensor.block(
                Labels("spherical_harmonics_l", np.array([[l]]))
            ).values.reshape(n_samples, -1)
            for l in nu_tensor.keys["spherical_harmonics_l"]
        ]
    )
    nu_tensor_norm = np.linalg.norm(nu_tensor_values, axis=1)
    nu1_tensor_values = np.hstack(
        [
            nu1_tensor.block(
                Labels("spherical_harmonics_l", np.array([[l]]))
            ).values.reshape(n_samples, -1)
            for l in nu1_tensor.keys["spherical_harmonics_l"]
        ]
    )
    nu1_tensor_norm = np.linalg.norm(nu1_tensor_values, axis=1)

    # check if the norm is equal
    assert np.allclose(nu_tensor_norm, nu1_tensor_norm**nu)
