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

from .rotations import WignerDReal, transform_frame_so3, transform_frame_o3


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

# TODO: test a CG combination with LODE
LODE_HYPERS_SMALL = {}


# ============ Pytest fixtures ============


@pytest.fixture(scope="module")
def cg_cache_sparse():
    return clebsch_gordan.ClebschGordanReal(lambda_max=5, sparse=True)


@pytest.fixture(scope="module")
def cg_cache_dense():
    return clebsch_gordan.ClebschGordanReal(lambda_max=5, sparse=False)


# ============ Helper functions ============


def h2_isolated():
    return ase.io.read(os.path.join(DATA_ROOT, "h2_isolated.xyz"), ":")


def h2o_isolated():
    return ase.io.read(os.path.join(DATA_ROOT, "h2o_isolated.xyz"), ":")


def h2o_periodic():
    return ase.io.read(os.path.join(DATA_ROOT, "h2o_periodic.xyz"), ":")


def wigners(lmax: int):
    return WignerDReal(lmax=lmax)


def sphex(frames: List[ase.Atoms]):
    """Returns a rascaline SphericalExpansion"""
    calculator = rascaline.SphericalExpansion(**SPHEX_HYPERS)
    return calculator.compute(frames)


def sphex_small_features(frames: List[ase.Atoms]):
    """Returns a rascaline SphericalExpansion"""
    calculator = rascaline.SphericalExpansion(**SPHEX_HYPERS_SMALL)
    return calculator.compute(frames)


def powspec(frames: List[ase.Atoms]):
    """Returns a rascaline PowerSpectrum constructed from a
    SphericalExpansion"""
    return PowerSpectrum(rascaline.SphericalExpansion(**SPHEX_HYPERS)).compute(frames)


def powspec_small_features(frames: List[ase.Atoms]):
    """Returns a rascaline PowerSpectrum constructed from a
    SphericalExpansion"""
    return PowerSpectrum(rascaline.SphericalExpansion(**SPHEX_HYPERS_SMALL)).compute(
        frames
    )


def lode_small_features(frames: List[ase.Atoms]):
    """Returns a rascaline LODE SphericalExpansion"""
    return rascaline.LodeSphericalExpansion(**LODE_HYPERS_SMALL).compute(frames)


def get_norm(tensor: TensorMap):
    """
    Calculates the norm used in CG iteration tests. Assumes standardized
    metadata and that the TensorMap is sliced to a single sample.

    For a given atomic sample, the norm is calculated for each feature vector,
    as a sum over lambda, sigma, and m.
    """
    norm = 0.0
    for key, block in tensor.items():  # Sum over lambda and sigma
        l = key["spherical_harmonics_l"]
        norm += np.sum(
            [np.linalg.norm(block.values[0, m, :]) ** 2 for m in range(-l, l + 1)]
        )

    return norm


# ============ Test equivariance ============


@pytest.mark.parametrize(
    "frames, nu_target, angular_cutoff, angular_selection, parity_selection",
    [
        (h2_isolated(), 3, None, [0, 4, 5], [+1]),
        (h2o_isolated(), 2, 5, None, None),
        (h2o_periodic(), 2, 5, None, None),
    ],
)
def test_so3_equivariance(
    frames: List[ase.Atoms],
    nu_target: int,
    angular_cutoff: int,
    angular_selection: List[List[int]],
    parity_selection: List[List[int]],
):
    wig = wigners(nu_target * SPHEX_HYPERS["max_angular"])
    frames_so3 = [
        transform_frame_so3(frame, wig.angles) for frame in frames
    ]

    nu_1 = sphex(frames)
    nu_1_so3 = sphex(frames_so3)

    nu_3 = clebsch_gordan.single_center_combine_to_order(
        nu1_tensor=nu_1,
        correlation_order=nu_target,
        angular_cutoff=angular_cutoff,
        angular_selection=angular_selection,
        parity_selection=parity_selection,
    )
    nu_3_so3 = clebsch_gordan.single_center_combine_to_order(
        nu1_tensor=nu_1_so3,
        correlation_order=nu_target,
        angular_cutoff=angular_cutoff,
        angular_selection=angular_selection,
        parity_selection=parity_selection,
    )

    nu_3_transf = wig.transform_tensormap_so3(nu_3)
    assert metatensor.allclose(nu_3_transf, nu_3_so3)


@pytest.mark.parametrize(
    "frames, nu_target, angular_cutoff, angular_selection, parity_selection",
    [
        (h2_isolated(), 3, None, [0, 4, 5], None),
        (h2o_isolated(), 2, 5, None, None),
        (h2o_periodic(), 2, 5, None, None),
    ],
)
def test_o3_equivariance(
    frames: List[ase.Atoms],
    nu_target: int,
    angular_cutoff: int,
    angular_selection: List[List[int]],
    parity_selection: List[List[int]],
):
    wig = wigners(nu_target * SPHEX_HYPERS["max_angular"])
    frames_o3 = [
        transform_frame_o3(frame, wig.angles) for frame in frames
    ]

    nu_1 = sphex(frames)
    nu_1_o3 = sphex(frames_o3)

    nu_3 = clebsch_gordan.single_center_combine_to_order(
        nu1_tensor=nu_1,
        correlation_order=nu_target,
        angular_cutoff=angular_cutoff,
        angular_selection=angular_selection,
        parity_selection=parity_selection,
    )
    nu_3_o3 = clebsch_gordan.single_center_combine_to_order(
        nu1_tensor=nu_1_o3,
        correlation_order=nu_target,
        angular_cutoff=angular_cutoff,
        angular_selection=angular_selection,
        parity_selection=parity_selection,
    )

    nu_3_transf = wig.transform_tensormap_o3(nu_3)
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
    nu1_tensor = sphex_small_features(frames)
    lsoap = clebsch_gordan.single_center_combine_to_order(
        nu1_tensor=nu1_tensor,
        correlation_order=2,
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


@pytest.mark.parametrize("frames", [h2_isolated(), h2o_periodic()])
@pytest.mark.parametrize("correlation_order", [2, 3, 4])
def test_combine_single_center_norm(frames, correlation_order):
    """
    Checks \|ρ^\\nu\| =  \|ρ\|^\\nu in the case where l lists are not sorted. If
    l lists are sorted, thus saving computation of redundant block combinations,
    the norm check will not hold for target body order greater than 2.
    """

    # Build nu=1 SphericalExpansion
    nu1 = sphex_small_features(frames)

    # Build higher body order tensor without sorting the l lists
    nux = clebsch_gordan.single_center_combine_to_order(
        nu1,
        correlation_order=correlation_order,
        angular_cutoff=None,
        angular_selection=None,
        parity_selection=None,
        skip_redundant=False,
        use_sparse=True,
    )
    # Build higher body order tensor *with* sorting the l lists
    nux_sorted_l = clebsch_gordan.single_center_combine_to_order(
        nu1,
        correlation_order=correlation_order,
        angular_cutoff=None,
        angular_selection=None,
        parity_selection=None,
        skip_redundant=True,
        use_sparse=True,
    )

    # Standardize the features by passing through the CG combination code but with
    # no iterations (i.e. body order 1 -> 1)
    nu1 = clebsch_gordan.single_center_combine_to_order(
        nu1,
        correlation_order=1,
        angular_cutoff=None,
        angular_selection=None,
        parity_selection=None,
        skip_redundant=False,
        use_sparse=True,
    )

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
        norm_nu1 += get_norm(nu1) ** correlation_order
        norm_nux += get_norm(nux)
        norm_nux_sorted_l += get_norm(nux_sorted_l)

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


@pytest.mark.parametrize("frames", [h2_isolated(), h2o_isolated()])
def test_single_center_combine_to_correlation_order_dense_sparse_agree(frames):
    """
    Tests for agreement between nu=3 tensors built using both sparse and dense
    CG coefficient caches.
    """
    nu1_tensor = sphex_small_features(frames)
    n_body_sparse = clebsch_gordan.single_center_combine_to_order(
        nu1_tensor,
        correlation_order=3,
        use_sparse=True,
    )
    n_body_dense = clebsch_gordan.single_center_combine_to_order(
        nu1_tensor,
        correlation_order=3,
        use_sparse=False,
    )

    assert metatensor.operations.allclose(
        n_body_sparse, n_body_dense, atol=1e-8, rtol=1e-8
    )


# ============ Test metadata  ============


@pytest.mark.parametrize("frames", [h2o_isolated()])
@pytest.mark.parametrize("correlation_order", [2, 3])
@pytest.mark.parametrize("skip_redundant", [True, False])
def test_single_center_combine_to_correlation_order_metadata_agree(
    frames, correlation_order, skip_redundant
):
    """
    Tests that the metadata output from
    single_center_combine_metadata_to_order agrees with the metadata
    of the full tensor built using single_center_combine_to_order.
    """
    for nu1 in [sphex_small_features(frames), sphex(frames)]:
        # Build higher body order tensor with CG computation
        nux = clebsch_gordan.single_center_combine_to_order(
            nu1,
            correlation_order=correlation_order,
            angular_cutoff=None,
            angular_selection=None,
            parity_selection=None,
            skip_redundant=skip_redundant,
            use_sparse=True,
        )
        # Build higher body order tensor without CG computation - i.e. metadata
        # only
        nux_metadata_only = (
            clebsch_gordan.single_center_combine_metadata_to_order(
                nu1,
                correlation_order=correlation_order,
                angular_cutoff=None,
                angular_selection=None,
                parity_selection=None,
                skip_redundant=skip_redundant,
            )
        )
        assert metatensor.equal_metadata(nux, nux_metadata_only)


@pytest.mark.parametrize("frames", [h2o_isolated()])
@pytest.mark.parametrize("correlation_order", [2, 3])
@pytest.mark.parametrize("skip_redundant", [True, False])
def test_single_center_combine_to_correlation_order_metadata(
    frames, correlation_order, skip_redundant
):
    """
    Performs hard-coded tests on the metadata outputted from
    single_center_combine_metadata_to_order.

    TODO: finish!
    """
    # for nu1 in [sphex_small_features(frames), sphex(frames)]:
    #     # Build higher body order tensor without CG computation - i.e. metadata
    #     # only. This returns a list of the TensorMaps formed at each CG
    #     # iteration.
    #     nux_metadata_only = (
    #         clebsch_gordan.single_center_combine_metadata_to_order(
    #             nu1,
    #             correlation_order=correlation_order,
    #             angular_cutoff=None,
    #             angular_selection=None,
    #             parity_selection=None,
    #             skip_redundant=skip_redundant,
    #         )
    #     )


@pytest.mark.parametrize("frames", [h2o_isolated()])
@pytest.mark.parametrize("angular_selection", [None, [1, 2, 4]])
@pytest.mark.parametrize("skip_redundant", [True, False])
def test_single_center_combine_angular_selection(
    frames: List[ase.Atoms],
    angular_selection: List[List[int]],
    skip_redundant: bool,
):
    """Tests that the correct angular channels are outputted based on the
    specified ``angular_cutoff`` and ``angular_selection``."""
    nu_1 = sphex(frames)

    nu_2 = clebsch_gordan.single_center_combine_to_order(
        nu1_tensor=nu_1,
        correlation_order=2,
        angular_cutoff=None,
        angular_selection=angular_selection,
        parity_selection=None,
        skip_redundant=skip_redundant,
    )

    if angular_selection is None:
        assert np.all(
            [
                l in np.arange(SPHEX_HYPERS["max_angular"] * 2 + 1)
                for l in np.unique(nu_2.keys.column("spherical_harmonics_l"))
            ]
        )

    else:
        assert np.all(
            np.sort(np.unique(nu_2.keys.column("spherical_harmonics_l")))
            == np.sort(angular_selection)
        )


# ============ Test dispatch to torch/numpy  ============
