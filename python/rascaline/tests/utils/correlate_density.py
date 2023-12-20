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
from rascaline.utils.clebsch_gordan.correlate_density import (
    _correlate_density,
    correlate_density,
    correlate_density_metadata,
)


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


# ============ Pytest fixtures ============


@pytest.fixture()
def cg_cache_sparse():
    return ClebschGordanReal(lambda_max=5, sparse=True)


@pytest.fixture()
def cg_cache_dense():
    return ClebschGordanReal(lambda_max=5, sparse=False)


# ============ Helper functions ============


def h2_isolated():
    return ase.io.read(os.path.join(DATA_ROOT, "h2_isolated.xyz"), ":")


def h2o_isolated():
    return ase.io.read(os.path.join(DATA_ROOT, "h2o_isolated.xyz"), ":")


def h2o_periodic():
    return ase.io.read(os.path.join(DATA_ROOT, "h2o_periodic.xyz"), ":")


def wigner_d_matrices(lmax: int):
    return WignerDReal(lmax=lmax)


def spherical_expansion(frames: List[ase.Atoms]):
    """Returns a rascaline SphericalExpansion"""
    calculator = rascaline.SphericalExpansion(**SPHEX_HYPERS)
    return calculator.compute(frames)


def spherical_expansion_small(frames: List[ase.Atoms]):
    """Returns a rascaline SphericalExpansion"""
    calculator = rascaline.SphericalExpansion(**SPHEX_HYPERS_SMALL)
    return calculator.compute(frames)


def power_spectrum(frames: List[ase.Atoms]):
    """Returns a rascaline PowerSpectrum constructed from a
    SphericalExpansion"""
    return PowerSpectrum(rascaline.SphericalExpansion(**SPHEX_HYPERS)).compute(frames)


def power_spectrum_small(frames: List[ase.Atoms]):
    """Returns a rascaline PowerSpectrum constructed from a
    SphericalExpansion"""
    return PowerSpectrum(rascaline.SphericalExpansion(**SPHEX_HYPERS_SMALL)).compute(
        frames
    )


def get_norm(tensor: TensorMap):
    """
    Calculates the norm used in CG iteration tests. Assumes that the TensorMap
    is sliced to a single sample.

    For a given atomic sample, the norm is calculated for each feature vector,
    as a sum over lambda, sigma, and m.
    """
    # Check that there is only one sample
    assert (
        len(
            metatensor.unique_metadata(
                tensor, "samples", ["structure", "center", "species_center"]
            ).values
        )
        == 1
    )
    norm = 0.0
    for key, block in tensor.items():  # Sum over lambda and sigma
        angular_l = key["spherical_harmonics_l"]
        norm += np.sum(
            [
                np.linalg.norm(block.values[0, m, :]) ** 2
                for m in range(-angular_l, angular_l + 1)
            ]
        )

    return norm


# ============ Test equivariance ============


@pytest.mark.skipif(
    not HAS_SYMPY or not HAS_METATENSOR_OPERATIONS,
    reason="SymPy or metatensor-operations are not installed",
)
@pytest.mark.parametrize(
    "frames, nu_target, angular_cutoff, selected_keys",
    [(h2o_periodic(), 2, 3, None)],
)
def test_so3_equivariance(frames, nu_target, angular_cutoff, selected_keys):
    """
    Tests that the output of :py:func:`correlate_density` is equivariant under
    SO(3) transformations.
    """
    wig = wigner_d_matrices(nu_target * SPHEX_HYPERS["max_angular"])
    frames_so3 = [transform_frame_so3(frame, wig.angles) for frame in frames]

    nu_1 = spherical_expansion(frames)
    nu_1_so3 = spherical_expansion(frames_so3)

    nu_3 = correlate_density(
        density=nu_1,
        correlation_order=nu_target,
        angular_cutoff=angular_cutoff,
        selected_keys=selected_keys,
    )
    nu_3_so3 = correlate_density(
        density=nu_1_so3,
        correlation_order=nu_target,
        angular_cutoff=angular_cutoff,
        selected_keys=selected_keys,
    )

    nu_3_transf = wig.transform_tensormap_so3(nu_3)
    assert metatensor.allclose(nu_3_transf, nu_3_so3)


@pytest.mark.skipif(
    not HAS_SYMPY or not HAS_METATENSOR_OPERATIONS,
    reason="SymPy or metatensor-operations are not installed",
)
@pytest.mark.parametrize(
    "frames, nu_target, angular_cutoff, selected_keys",
    [(h2_isolated(), 2, 3, None)],
)
def test_o3_equivariance(frames, nu_target, angular_cutoff, selected_keys):
    """
    Tests that the output of :py:func:`correlate_density` is equivariant under
    O(3) transformations.
    """
    wig = wigner_d_matrices(nu_target * SPHEX_HYPERS["max_angular"])
    frames_o3 = [transform_frame_o3(frame, wig.angles) for frame in frames]

    nu_1 = spherical_expansion(frames)
    nu_1_o3 = spherical_expansion(frames_o3)

    nu_3 = correlate_density(
        density=nu_1,
        correlation_order=nu_target,
        angular_cutoff=angular_cutoff,
        selected_keys=selected_keys,
    )
    nu_3_o3 = correlate_density(
        density=nu_1_o3,
        correlation_order=nu_target,
        angular_cutoff=angular_cutoff,
        selected_keys=selected_keys,
    )

    nu_3_transf = wig.transform_tensormap_o3(nu_3)
    assert metatensor.allclose(nu_3_transf, nu_3_o3)


# ============ Test lambda-SOAP vs PowerSpectrum ============


@pytest.mark.skipif(
    not HAS_METATENSOR_OPERATIONS, reason="metatensor-operations is not installed"
)
@pytest.mark.parametrize("frames", [h2_isolated()])
@pytest.mark.parametrize(
    "sphex_powspec",
    [(spherical_expansion, power_spectrum)],
)
def test_lambda_soap_vs_powerspectrum(frames, sphex_powspec):
    """
    Tests for exact equivalence between the invariant block of a generated
    lambda-SOAP equivariant and the Python implementation of PowerSpectrum in
    rascaline utils.
    """
    # Build a PowerSpectrum
    ps = sphex_powspec[1](frames)

    # Build a lambda-SOAP
    density = sphex_powspec[0](frames)
    lsoap = correlate_density(
        density=density,
        correlation_order=2,
        selected_keys=Labels(
            names=["spherical_harmonics_l"], values=np.array([0]).reshape(-1, 1)
        ),
    )
    keys = lsoap.keys
    keys = keys.remove(name="order_nu")
    keys = keys.remove(name="inversion_sigma")
    keys = keys.remove(name="spherical_harmonics_l")
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
    assert metatensor.allclose(lsoap, ps)


# ============ Test norm preservation  ============


@pytest.mark.skipif(
    not HAS_METATENSOR_OPERATIONS, reason="metatensor-operations is not installed"
)
@pytest.mark.parametrize("frames", [h2o_periodic()])
@pytest.mark.parametrize("correlation_order", [2, 4])
def test_correlate_density_norm(frames, correlation_order):
    """
    Checks \\|ρ^\\nu\\| =  \\|ρ\\|^\\nu in the case where l lists are not
    sorted. If l lists are sorted, thus saving computation of redundant block
    combinations, the norm check will not hold for target body order greater
    than 2.
    """

    # Build nu=1 SphericalExpansion
    nu1 = spherical_expansion_small(frames)

    # Build higher body order tensor without sorting the l lists
    nux = correlate_density(
        nu1,
        correlation_order=correlation_order,
        angular_cutoff=None,
        selected_keys=None,
        skip_redundant=False,
    )
    # Build higher body order tensor *with* sorting the l lists
    nux_sorted_l = correlate_density(
        nu1,
        correlation_order=correlation_order,
        angular_cutoff=None,
        selected_keys=None,
        skip_redundant=True,
    )

    # Standardize the features by passing through the CG combination code but with
    # no iterations (i.e. body order 1 -> 1)
    nu1 = _standardize_keys(nu1)

    # Make only lambda and sigma part of keys
    nu1 = nu1.keys_to_samples(["species_center"])
    nux = nux.keys_to_samples(["species_center"])
    nux_sorted_l = nux_sorted_l.keys_to_samples(["species_center"])

    # The norm shoudl be calculated for each sample. First find the unqiue
    # samples
    uniq_samples = metatensor.unique_metadata(
        nux, "samples", names=["structure", "center", "species_center"]
    )
    grouped_labels = [
        Labels(names=nux.sample_names, values=uniq_samples.values[i].reshape(1, 3))
        for i in range(len(uniq_samples))
    ]

    # Calculate norms
    norm_nu1 = 0.0
    norm_nux = 0.0
    norm_nux_sorted_l = 0.0
    for sample in grouped_labels:
        # Slice the TensorMaps
        nu1_sliced = metatensor.slice(nu1, "samples", labels=sample)
        nux_sliced = metatensor.slice(nux, "samples", labels=sample)
        nux_sorted_sliced = metatensor.slice(nux_sorted_l, "samples", labels=sample)

        # Calculate norms
        norm_nu1 += get_norm(nu1_sliced) ** correlation_order
        norm_nux += get_norm(nux_sliced)
        norm_nux_sorted_l += get_norm(nux_sorted_sliced)

    # Without sorting the l list we should get the same norm
    assert np.allclose(norm_nu1, norm_nux)

    # But with sorting the l list we should get a different norm
    assert not np.allclose(norm_nu1, norm_nux_sorted_l)


# ============ Test CG cache  ============


@pytest.mark.parametrize("l1, l2", [(1, 2), (2, 3), (0, 5)])
def test_clebsch_gordan_orthogonality(cg_cache_dense, l1, l2):
    """
    Test orthogonality relationships of cached dense CG coefficients.

    See
    https://en.wikipedia.org/wiki/Clebsch%E2%80%93Gordan_coefficients#Orthogonality_relations
    for details.
    """
    lam_min = abs(l1 - l2)
    lam_max = l1 + l2

    # We test lam dimension
    # \sum_{-m1 \leq l1 \leq m1, -m2 \leq l2 \leq m2}
    #           <λμ|l1m1,l2m2> <l1m1',l2m2'|λμ'> = δ_μμ'
    for lam in range(lam_min, lam_max):
        cg_mat = cg_cache_dense.coeffs[(l1, l2, lam)].reshape(-1, 2 * lam + 1)
        dot_product = cg_mat.T @ cg_mat
        diag_mask = np.zeros(dot_product.shape, dtype=np.bool_)
        diag_mask[np.diag_indices(len(dot_product))] = True
        assert np.allclose(
            dot_product[~diag_mask], np.zeros(dot_product.shape)[~diag_mask]
        )
        assert np.allclose(dot_product[diag_mask], dot_product[diag_mask][0])

    # We test l1 l2 dimension
    # \sum_{|l1-l2| \leq λ \leq l1+l2} \sum_{-μ \leq λ \leq μ}
    #            <l1m1,l2m2|λμ> <λμ|l1m1,l2m2> = δ_m1m1' δ_m2m2'
    l1l2_dim = (2 * l1 + 1) * (2 * l2 + 1)
    dot_product = np.zeros((l1l2_dim, l1l2_dim))
    for lam in range(lam_min, lam_max + 1):
        cg_mat = cg_cache_dense.coeffs[(l1, l2, lam)].reshape(-1, 2 * lam + 1)
        dot_product += cg_mat @ cg_mat.T
    diag_mask = np.zeros(dot_product.shape, dtype=np.bool_)
    diag_mask[np.diag_indices(len(dot_product))] = True

    assert np.allclose(dot_product[~diag_mask], np.zeros(dot_product.shape)[~diag_mask])
    assert np.allclose(dot_product[diag_mask], dot_product[diag_mask][0])


@pytest.mark.skipif(
    not HAS_METATENSOR_OPERATIONS, reason="metatensor-operations is not installed"
)
@pytest.mark.parametrize("frames", [h2o_periodic()])
def test_correlate_density_dense_sparse_agree(frames):
    """
    Tests for agreement between nu=3 tensors built using both sparse and dense
    CG coefficient caches.
    """
    density = spherical_expansion_small(frames)

    # NOTE: testing the private function here so we can control the use of
    # sparse v dense CG cache
    n_body_sparse = _correlate_density(
        density,
        correlation_order=2,
        compute_metadata_only=False,
        sparse=True,
    )
    n_body_dense = _correlate_density(
        density,
        correlation_order=2,
        compute_metadata_only=False,
        sparse=False,
    )

    assert metatensor.allclose(n_body_sparse, n_body_dense, atol=1e-8, rtol=1e-8)


# ============ Test metadata  ============


@pytest.mark.skipif(
    not HAS_METATENSOR_OPERATIONS, reason="metatensor-operations is not installed"
)
@pytest.mark.parametrize("frames", [h2o_isolated()])
@pytest.mark.parametrize("correlation_order", [3])
@pytest.mark.parametrize("skip_redundant", [True])
def test_correlate_density_metadata_agree(frames, correlation_order, skip_redundant):
    """
    Tests that the metadata of outputs from :py:func:`correlate_density` and
    :py:func:`correlate_density_metadata` agree.
    """
    for nu1 in [spherical_expansion_small(frames), spherical_expansion(frames)]:
        # Build higher body order tensor with CG computation
        nux = correlate_density(
            nu1,
            correlation_order=correlation_order,
            angular_cutoff=3,
            selected_keys=None,
            skip_redundant=skip_redundant,
        )
        # Build higher body order tensor without CG computation - i.e. metadata
        # only
        nux_metadata_only = correlate_density_metadata(
            nu1,
            correlation_order=correlation_order,
            angular_cutoff=3,
            selected_keys=None,
            skip_redundant=skip_redundant,
        )
        assert metatensor.equal_metadata(nux, nux_metadata_only)


@pytest.mark.parametrize("frames", [h2o_isolated()])
@pytest.mark.parametrize(
    "selected_keys",
    [
        None,
        Labels(
            names=["spherical_harmonics_l"], values=np.array([1, 3]).reshape(-1, 1)
        ),
    ],
)
@pytest.mark.parametrize("skip_redundant", [True, False])
def test_correlate_density_angular_selection(
    frames: List[ase.Atoms],
    selected_keys: Labels,
    skip_redundant: bool,
):
    """
    Tests that the correct angular channels are outputted based on the
    specified ``selected_keys``.
    """
    nu_1 = spherical_expansion(frames)

    nu_2 = correlate_density(
        density=nu_1,
        correlation_order=2,
        angular_cutoff=None,
        selected_keys=selected_keys,
        skip_redundant=skip_redundant,
    )

    if selected_keys is None:
        assert np.all(
            [
                angular in np.arange(SPHEX_HYPERS["max_angular"] * 2 + 1)
                for angular in np.unique(nu_2.keys.column("spherical_harmonics_l"))
            ]
        )

    else:
        assert np.all(
            np.sort(np.unique(nu_2.keys.column("spherical_harmonics_l")))
            == np.sort(selected_keys.column("spherical_harmonics_l"))
        )
