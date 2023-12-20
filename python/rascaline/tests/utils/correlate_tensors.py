# -*- coding: utf-8 -*-
import os
from typing import List

import metatensor
import numpy as np
import pytest
from metatensor import Labels, TensorBlock, TensorMap

import rascaline
from rascaline.utils import PowerSpectrum
from rascaline.utils.clebsch_gordan._cg_cache import ClebschGordanReal
from rascaline.utils.clebsch_gordan._clebsch_gordan import _standardize_keys
from rascaline.utils.clebsch_gordan.correlate_tensors import (
    _correlate_tensors,
    correlate_tensors,
    correlate_tensors_metadata,
)
from rascaline.utils.clebsch_gordan.correlate_density import correlate_density


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

if HAS_SYMPY:
    from .rotations import WignerDReal, transform_frame_o3, transform_frame_so3


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")

SPHEX_HYPERS = {
    "cutoff": 3.0,  # Angstrom
    "max_radial": 6,  # Exclusive
    "max_angular": 4,  # Inclusive
    "atomic_gaussian_width": 0.2,
    "radial_basis": {"Gto": {}},
    "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
    "center_atom_weight": 1.0,
}

SPHEX_HYPERS_SMALL = {
    "cutoff": 3.0,  # Angstrom
    "max_radial": 1,  # Exclusive
    "max_angular": 2,  # Inclusive
    "atomic_gaussian_width": 0.2,
    "radial_basis": {"Gto": {}},
    "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
    "center_atom_weight": 1.0,
}


# ============ Pytest fixtures ============


@pytest.fixture()
def cg_cache_sparse():
    return ClebschGordanReal(lambda_max=5, sparse=True)


@pytest.fixture()
def cg_cache_dense():
    return ClebschGordanReal(lambda_max=5, sparse=False)


# ============ Helper functions ============


# def h2_isolated():
#     return ase.io.read(os.path.join(DATA_ROOT, "h2_isolated.xyz"), ":")


def h2o_isolated():
    return ase.io.read(os.path.join(DATA_ROOT, "h2o_isolated.xyz"), ":")


# def h2o_periodic():
#     return ase.io.read(os.path.join(DATA_ROOT, "h2o_periodic.xyz"), ":")


# def wigner_d_matrices(lmax: int):
#     return WignerDReal(lmax=lmax)


# def spherical_expansion(frames: List[ase.Atoms]):
#     """Returns a rascaline SphericalExpansion"""
#     calculator = rascaline.SphericalExpansion(**SPHEX_HYPERS)
#     return calculator.compute(frames)


def spherical_expansion_small(frames: List[ase.Atoms]):
    """Returns a rascaline SphericalExpansion"""
    calculator = rascaline.SphericalExpansion(**SPHEX_HYPERS_SMALL)
    return calculator.compute(frames)


# def power_spectrum(frames: List[ase.Atoms]):
#     """Returns a rascaline PowerSpectrum constructed from a
#     SphericalExpansion"""
#     return PowerSpectrum(rascaline.SphericalExpansion(**SPHEX_HYPERS)).compute(frames)


# def power_spectrum_small(frames: List[ase.Atoms]):
#     """Returns a rascaline PowerSpectrum constructed from a
#     SphericalExpansion"""
#     return PowerSpectrum(rascaline.SphericalExpansion(**SPHEX_HYPERS_SMALL)).compute(
#         frames
#     )


# def get_norm(tensor: TensorMap):
#     """
#     Calculates the norm used in CG iteration tests. Assumes that the TensorMap
#     is sliced to a single sample.

#     For a given atomic sample, the norm is calculated for each feature vector,
#     as a sum over lambda, sigma, and m.
#     """
#     # Check that there is only one sample
#     assert (
#         len(
#             metatensor.unique_metadata(
#                 tensor, "samples", ["structure", "center", "species_center"]
#             ).values
#         )
#         == 1
#     )
#     norm = 0.0
#     for key, block in tensor.items():  # Sum over lambda and sigma
#         angular_l = key["spherical_harmonics_l"]
#         norm += np.sum(
#             [
#                 np.linalg.norm(block.values[0, m, :]) ** 2
#                 for m in range(-angular_l, angular_l + 1)
#             ]
#         )

#     return norm


# ============ Test equivalence with `correlate_density` ============

@pytest.mark.skipif(
    not HAS_METATENSOR_OPERATIONS, reason="metatensor-operations is not installed"
)
@pytest.mark.parametrize("frames", [h2o_isolated()])
@pytest.mark.parametrize("selected_keys", [None])
def test_correlate_tensors_correlate_density_equivalent(frames, selected_keys):
    """
    Tests that using `correlate_tensors` with two identical densities 
    """
    density = spherical_expansion_small(frames)

    # NOTE: testing the private function here so we can control the use of
    # sparse v dense CG cache
    correlated_density = correlate_density(density, correlation_order=2, selected_keys=selected_keys)
    correlated_tensor = correlate_tensors(
        density, density, selected_keys=selected_keys
    )
    correlated_tensor = correlated_tensor.keys_to_properties(keys_to_move=["l1", "l2"])

    assert metatensor.equal(correlated_density, correlated_tensor)
