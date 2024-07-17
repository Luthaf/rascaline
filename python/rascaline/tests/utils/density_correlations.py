from typing import List

import metatensor
import numpy as np
import pytest
from metatensor import Labels, TensorBlock, TensorMap

import rascaline
from rascaline.utils import PowerSpectrum, _dispatch
from rascaline.utils.clebsch_gordan import ClebschGordanProduct, DensityCorrelations
from rascaline.utils.clebsch_gordan._coefficients import calculate_cg_coefficients


# Try to import some modules
ase = pytest.importorskip("ase")
import ase.io  # noqa: E402, F811


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


try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    ARRAYS_BACKEND = ["numpy", "torch"]
else:
    ARRAYS_BACKEND = ["numpy"]

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


# ============ Helper functions ============


def h2_isolated():
    return [
        ase.Atoms(
            symbols=["H", "H"],
            positions=[
                [1.97361700, 1.73067300, 2.47063400],
                [1.97361700, 3.26932700, 2.47063400],
            ],
        )
    ]


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


def h2o_periodic():

    return [
        ase.Atoms(
            symbols=["O", "H", "H"],
            positions=[
                [2.56633400, 2.50000000, 2.50370100],
                [1.97361700, 1.73067300, 2.47063400],
                [1.97361700, 3.26932700, 2.47063400],
            ],
            cell=[5, 5, 5],
            pbc=[True, True, True],
        )
    ]


def wigner_d_matrices(lmax: int):
    return WignerDReal(lmax=lmax)


def spherical_expansion(frames: List[ase.Atoms]):
    """Returns a rascaline SphericalExpansion"""
    calculator = rascaline.SphericalExpansion(**SPHEX_HYPERS)
    return calculator.compute(frames)


def spherical_expansion_small(frames: List[ase.Atoms]):
    """Returns a rascaline SphericalExpansion with smaller hypers"""
    calculator = rascaline.SphericalExpansion(**SPHEX_HYPERS_SMALL)
    return calculator.compute(frames)


def spherical_expansion_by_pair(frames: List[ase.Atoms]):
    """Returns a rascaline SphericalExpansionByPair"""
    calculator = rascaline.SphericalExpansionByPair(**SPHEX_HYPERS)
    return calculator.compute(frames)


def spherical_expansion_by_pair_small(frames: List[ase.Atoms]):
    """Returns a rascaline SphericalExpansionByPair with smaller hypers"""
    calculator = rascaline.SphericalExpansionByPair(**SPHEX_HYPERS_SMALL)
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
    samples = metatensor.unique_metadata(
        tensor, "samples", ["system", "atom", "center_type"]
    )
    assert len(samples) == 1

    norm = 0.0
    for key, block in tensor.items():  # Sum over lambda and sigma
        o3_sigma = key["o3_lambda"]
        norm += np.sum(
            [
                np.linalg.norm(block.values[0, m, :]) ** 2
                for m in range(-o3_sigma, o3_sigma + 1)
            ]
        )

    return norm


# ============ Test equivariance ============


@pytest.mark.skipif(
    not HAS_SYMPY or not HAS_METATENSOR_OPERATIONS,
    reason="SymPy or metatensor-operations are not installed",
)
def test_so3_equivariance():
    """
    Tests that the output of :py:func:`correlate_density` is equivariant under
    SO(3) transformations.
    """
    frames = h2o_periodic()
    n_correlations = 1

    wig = wigner_d_matrices((n_correlations + 1) * SPHEX_HYPERS["max_angular"])
    rotated_frames = [transform_frame_so3(frame, wig.angles) for frame in frames]

    # Generate density
    density = spherical_expansion(frames)
    density = density.keys_to_properties("neighbor_type")

    # Generate density_so3
    density_so3 = spherical_expansion(rotated_frames)
    density_so3 = density_so3.keys_to_properties("neighbor_type")

    calculator = DensityCorrelations(
        n_correlations=n_correlations,
        max_angular=SPHEX_HYPERS["max_angular"] * (n_correlations + 1),
    )
    nu_3 = calculator.compute(density)
    nu_3_so3 = calculator.compute(density_so3)

    nu_3_transf = wig.transform_tensormap_so3(nu_3)
    assert metatensor.allclose(nu_3_transf, nu_3_so3)


@pytest.mark.skipif(
    not HAS_SYMPY or not HAS_METATENSOR_OPERATIONS,
    reason="SymPy or metatensor-operations are not installed",
)
def test_o3_equivariance():
    """
    Tests that the output of :py:func:`correlate_density` is equivariant under
    O(3) transformations.
    """
    frames = h2_isolated()
    n_correlations = 1
    selected_keys = None
    wig = wigner_d_matrices((n_correlations + 1) * SPHEX_HYPERS["max_angular"])
    frames_o3 = [transform_frame_o3(frame, wig.angles) for frame in frames]

    # Generate density
    density = spherical_expansion(frames)
    density = density.keys_to_properties("neighbor_type")

    # Generate density_o3
    density_o3 = spherical_expansion(frames_o3)
    density_o3 = density_o3.keys_to_properties("neighbor_type")

    calculator = DensityCorrelations(
        n_correlations=n_correlations,
        max_angular=SPHEX_HYPERS["max_angular"] * (n_correlations + 1),
    )
    nu_3 = calculator.compute(density, selected_keys=selected_keys)
    nu_3_o3 = calculator.compute(density_o3, selected_keys=selected_keys)

    nu_3_transf = wig.transform_tensormap_o3(nu_3)
    assert metatensor.allclose(nu_3_transf, nu_3_o3)


# ============ Test lambda-SOAP vs PowerSpectrum ============


@pytest.mark.skipif(
    not HAS_METATENSOR_OPERATIONS, reason="metatensor-operations is not installed"
)
def test_lambda_soap_vs_powerspectrum():
    """
    Tests for exact equivalence between the invariant block of a generated
    lambda-SOAP equivariant and the Python implementation of PowerSpectrum in
    rascaline utils.
    """
    frames = h2_isolated()

    # Build a PowerSpectrum
    ps = power_spectrum(frames)

    # Build density
    density = spherical_expansion(frames)
    density = density.keys_to_properties("neighbor_type")

    n_correlations = 1
    calculator = DensityCorrelations(
        n_correlations=n_correlations,
        max_angular=SPHEX_HYPERS["max_angular"] * (n_correlations + 1),
    )
    lambda_soap = calculator.compute(
        density,
        selected_keys=Labels(names=["o3_lambda"], values=np.array([0]).reshape(-1, 1)),
    )
    lambda_soap = lambda_soap.keys_to_properties(["l_1", "l_2"])
    keys = lambda_soap.keys.remove(name="o3_lambda")
    keys = keys.remove("o3_sigma")

    # Manipulate metadata to match that of PowerSpectrum:
    # 1) remove components axis
    # 2) change "l1" and "l2" properties dimensions to just "l" (as l1 == l2)
    blocks = []
    for block in lambda_soap.blocks():
        n_samples, n_props = block.values.shape[0], block.values.shape[2]
        new_props = block.properties
        new_props = new_props.remove(name="l_1")
        new_props = new_props.rename(old="l_2", new="l")
        blocks.append(
            TensorBlock(
                values=block.values.reshape((n_samples, n_props)),
                samples=block.samples,
                components=[],
                properties=new_props,
            )
        )
    lambda_soap = TensorMap(keys=keys, blocks=blocks)

    assert metatensor.allclose(lambda_soap, ps)


# ============ Test norm preservation  ============


@pytest.mark.skipif(
    not HAS_METATENSOR_OPERATIONS, reason="metatensor-operations is not installed"
)
def test_correlate_density_norm():
    """
    Checks \\|\\rho^\\nu\\| =  \\|\\rho\\|^\\nu in the case where l lists are not
    sorted. If l lists are sorted, thus saving computation of redundant block
    combinations, the norm check will not hold for target body order greater than 2.
    """
    frames = h2o_periodic()
    n_correlations = 1

    # Build nu=1 SphericalExpansion
    density = spherical_expansion_small(frames)
    density = density.keys_to_properties("neighbor_type")

    # Build higher body order tensor without sorting the l lists
    calculator = DensityCorrelations(
        n_correlations=n_correlations,
        max_angular=SPHEX_HYPERS_SMALL["max_angular"] * (n_correlations + 1),
        skip_redundant=False,
    )
    ps = calculator.compute(density)

    # Build higher body order tensor *with* sorting the l lists
    calculator = DensityCorrelations(
        n_correlations=n_correlations,
        max_angular=SPHEX_HYPERS_SMALL["max_angular"] * (n_correlations + 1),
        skip_redundant=True,
    )
    ps_sorted = calculator.compute(density)

    # Make only lambda and sigma part of keys
    density = density.keys_to_samples(["center_type"])
    ps = ps.keys_to_samples(["center_type"])
    ps_sorted = ps_sorted.keys_to_samples(["center_type"])

    # The norm should be calculated for each sample. First find the unique
    # samples
    unique_samples = metatensor.unique_metadata(
        ps, "samples", names=["system", "atom", "center_type"]
    )
    grouped_labels = [
        Labels(names=ps.sample_names, values=unique_samples.values[i].reshape(1, 3))
        for i in range(len(unique_samples))
    ]

    # Calculate norms
    norm_nu1 = 0.0
    norm_ps = 0.0
    norm_ps_sorted = 0.0
    for sample in grouped_labels:
        # Slice the TensorMaps
        nu1_sliced = metatensor.slice(density, "samples", labels=sample)
        ps_sliced = metatensor.slice(ps, "samples", labels=sample)
        ps_sorted_sliced = metatensor.slice(ps_sorted, "samples", labels=sample)

        # Calculate norms
        norm_nu1 += get_norm(nu1_sliced) ** (n_correlations + 1)
        norm_ps += get_norm(ps_sliced)
        norm_ps_sorted += get_norm(ps_sorted_sliced)

    # Without sorting the l list we should get the same norm
    assert np.allclose(norm_nu1, norm_ps)

    # But with sorting the l list we should get a different norm
    assert not np.allclose(norm_nu1, norm_ps_sorted)


# ============ Test computation of CG coefficients  ============


@pytest.mark.parametrize("l1, l2", [(1, 2), (2, 3), (0, 5)])
@pytest.mark.parametrize("arrays_backend", ARRAYS_BACKEND)
def test_clebsch_gordan_orthogonality(l1, l2, arrays_backend):
    """
    Test orthogonality relationships of cached dense CG coefficients.

    See
    https://en.wikipedia.org/wiki/Clebsch%E2%80%93Gordan_coefficients#Orthogonality_relations
    for details.
    """
    if arrays_backend == "torch":
        int64_like = torch.empty(0, dtype=torch.int64)
        float64_like = torch.empty(0, dtype=torch.float64)
        bool_like = torch.empty(0, dtype=torch.bool)
        dtype = torch.float64
        device = "cpu"
    elif arrays_backend == "numpy":
        int64_like = np.empty(0, dtype=np.int64)
        float64_like = np.empty(0, dtype=np.float64)
        bool_like = np.empty(0, dtype=np.bool_)
        dtype = np.float64
        device = "cpu"
    else:
        raise ValueError(f"Not supported arrays backend {arrays_backend}.")

    cg_coeffs = calculate_cg_coefficients(
        lambda_max=5,
        cg_backend="python-dense",
        arrays_backend=arrays_backend,
        dtype=dtype,
        device=device,
    )

    lambda_min = abs(l1 - l2)
    lambda_max = l1 + l2

    # We test lambda dimension
    # \sum_{-m1 \leq l1 \leq m1, -m2 \leq l2 \leq m2}
    #           <λμ|l1m1,l2m2> <l1m1',l2m2'|λμ'> = δ_μμ'
    for o3_lambda in range(lambda_min, lambda_max):
        cg_mat = cg_coeffs.block(
            {"l1": l1, "l2": l2, "lambda": o3_lambda}
        ).values.reshape(-1, 2 * o3_lambda + 1)

        dot_product = cg_mat.T @ cg_mat
        diag_mask = _dispatch.zeros_like(bool_like, dot_product.shape)
        diag_indices = (
            _dispatch.int_range_like(0, len(dot_product), int64_like),
            _dispatch.int_range_like(0, len(dot_product), int64_like),
        )
        diag_mask[diag_indices] = True
        assert _dispatch.allclose(
            dot_product[~diag_mask],
            _dispatch.zeros_like(float64_like, dot_product.shape)[~diag_mask],
            rtol=1e-05,
            atol=1e-08,
        )
        assert _dispatch.allclose(
            dot_product[diag_mask], dot_product[diag_mask][0:1], rtol=1e-05, atol=1e-08
        )

    # We test l1 l2 dimension
    # \sum_{|l1-l2| \leq λ \leq l1+l2} \sum_{-μ \leq λ \leq μ}
    #            <l1m1,l2m2|λμ> <λμ|l1m1,l2m2> = δ_m1m1' δ_m2m2'
    l1l2_dim = (2 * l1 + 1) * (2 * l2 + 1)
    dot_product = _dispatch.zeros_like(float64_like, (l1l2_dim, l1l2_dim))
    for o3_lambda in range(lambda_min, lambda_max + 1):
        cg_mat = cg_coeffs.block(
            {"l1": l1, "l2": l2, "lambda": o3_lambda}
        ).values.reshape(-1, 2 * o3_lambda + 1)
        dot_product += cg_mat @ cg_mat.T
    diag_mask = _dispatch.zeros_like(bool_like, dot_product.shape)
    diag_indices = (
        _dispatch.int_range_like(0, len(dot_product), int64_like),
        _dispatch.int_range_like(0, len(dot_product), int64_like),
    )
    diag_mask[diag_indices] = True

    assert _dispatch.allclose(
        dot_product[~diag_mask],
        _dispatch.zeros_like(float64_like, dot_product.shape)[~diag_mask],
        rtol=1e-05,
        atol=1e-08,
    )
    assert _dispatch.allclose(
        dot_product[diag_mask], dot_product[diag_mask][0:1], rtol=1e-05, atol=1e-08
    )


@pytest.mark.skipif(
    not HAS_METATENSOR_OPERATIONS, reason="metatensor-operations is not installed"
)
def test_correlate_density_dense_sparse_agree():
    """
    Tests for agreement between nu=2 tensors built using both sparse and dense
    CG coefficient caches.
    """
    frames = h2o_periodic()
    density = spherical_expansion_small(frames)
    density = density.keys_to_properties("neighbor_type")

    n_correlations = 1
    calculator_sparse = DensityCorrelations(
        n_correlations=n_correlations,
        max_angular=SPHEX_HYPERS_SMALL["max_angular"] * (n_correlations + 1),
        cg_backend="python-sparse",
    )
    calculator_dense = DensityCorrelations(
        max_angular=SPHEX_HYPERS_SMALL["max_angular"] * (n_correlations + 1),
        n_correlations=n_correlations,
        cg_backend="python-dense",
    )

    n_body_sparse = calculator_sparse.compute(density)
    n_body_dense = calculator_dense.compute(density)

    assert metatensor.allclose(n_body_sparse, n_body_dense, atol=1e-8, rtol=1e-8)


# ============ Test metadata  ============


@pytest.mark.skipif(
    not HAS_METATENSOR_OPERATIONS, reason="metatensor-operations is not installed"
)
def test_correlate_density_metadata_agree():
    """
    Tests that the metadata of outputs from :py:func:`correlate_density` and
    :py:func:`correlate_density_metadata` agree.
    """
    frames = h2o_isolated()

    for max_angular, nu_1 in [
        (4, spherical_expansion_small(frames)),
        (6, spherical_expansion(frames)),
    ]:
        nu_1 = nu_1.keys_to_properties("neighbor_type")
        calculator = DensityCorrelations(
            n_correlations=1,
            max_angular=max_angular,
        )

        # Build higher body order tensor with CG computation
        nu_x = calculator.compute(nu_1)

        # Build higher body order tensor without CG computation - i.e. metadata
        # only
        nu_x_metadata_only = calculator.compute_metadata(nu_1)
        assert metatensor.equal_metadata(nu_x, nu_x_metadata_only)


@pytest.mark.parametrize(
    "selected_keys", [None, Labels(names=["o3_lambda"], values=np.array([[1], [3]]))]
)
@pytest.mark.parametrize("skip_redundant", [True, False])
@pytest.mark.parametrize("arrays_backend", ARRAYS_BACKEND + [None])
def test_correlate_density_angular_selection(
    selected_keys, skip_redundant, arrays_backend
):
    """
    Tests that the correct angular channels are output based on the specified
    ``selected_keys``.
    """
    frames = h2o_isolated()
    nu_1 = spherical_expansion(frames)
    nu_1 = nu_1.keys_to_properties("neighbor_type")

    n_correlations = 1
    calculator = DensityCorrelations(
        n_correlations=n_correlations,
        skip_redundant=skip_redundant,
        max_angular=SPHEX_HYPERS["max_angular"] * (n_correlations + 1),
        arrays_backend=arrays_backend,
        dtype=torch.float64 if arrays_backend == "torch" else None,
    )

    if arrays_backend is not None:
        nu_1 = nu_1.to(arrays=arrays_backend)

    nu_2 = calculator.compute(nu_1, selected_keys=selected_keys)

    if selected_keys is None:
        assert np.all(
            [
                angular in np.arange(SPHEX_HYPERS["max_angular"] * 2 + 1)
                for angular in np.unique(nu_2.keys.column("o3_lambda"))
            ]
        )

    else:
        assert np.all(
            np.sort(np.unique(nu_2.keys.column("o3_lambda")))
            == np.sort(selected_keys.column("o3_lambda"))
        )


def test_summed_powerspectrum_by_pair_equals_powerspectrum():
    """
    Tests that these two processes give equivalent outputs:

    1. Perform a single density correlation of a SphericalExpansion to produce an
       equivariant power spectrum, (i.e. lambda-SOAP)
    2. Perform a single density correlation of a SphericalExpansion with a
       SphericalExpansionByPair to produce an equivariant power spectrum by pair, then
       sum over neighbors to reduce to a single-center equivariant power spectrum.

    Provided both the density and pair density are generated with the same hypers.
    """
    frames = h2o_isolated()
    density_correlations = DensityCorrelations(
        n_correlations=1,
        max_angular=SPHEX_HYPERS["max_angular"] * 2,
        skip_redundant=False,
    )
    cg_product = ClebschGordanProduct(
        max_angular=SPHEX_HYPERS["max_angular"] * 2,
    )

    # Generate density and rename dimensions ready for correlation
    density = spherical_expansion_small(frames)
    density = metatensor.rename_dimension(
        density, "keys", "center_type", "first_atom_type"
    )
    density = metatensor.rename_dimension(
        density, "keys", "neighbor_type", "second_atom_type"
    )
    density = metatensor.rename_dimension(density, "samples", "atom", "first_atom")
    density = density.keys_to_properties("second_atom_type")

    # Calculate power spectrum with DensityCorrelations
    power_spectrum = density_correlations(density)

    # Rename dimensions ready for manual CG tensor product by ClebschGordanProduct
    density = metatensor.rename_dimension(density, "properties", "n", "n_1")
    density = metatensor.rename_dimension(
        density, "properties", "second_atom_type", "second_atom_1_type"
    )

    # Generate pair density
    pair_density = spherical_expansion_by_pair_small(frames)
    pair_density = pair_density.keys_to_properties("second_atom_type")
    pair_density = metatensor.rename_dimension(pair_density, "properties", "n", "n_2")
    pair_density = metatensor.rename_dimension(
        pair_density, "properties", "second_atom_type", "second_atom_2_type"
    )

    # Compute power spectrum by pair via ClebschGordanProduct
    power_spectrum_by_pair = cg_product(
        tensor_1=density,
        tensor_2=pair_density,
        o3_lambda_1_new_name="l_1",
        o3_lambda_2_new_name="l_2",
    )

    # Sum, sort, check equivalence
    power_spec_by_pair_reduced = metatensor.sum_over_samples(
        power_spectrum_by_pair,
        ["second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"],
    )

    # Compare
    metatensor.allclose_raise(
        metatensor.sort(power_spectrum),
        metatensor.sort(power_spec_by_pair_reduced),
    )


def test_angular_cutoff():
    """
    Tests that the angular channels present in the output of a DensityCorrelations
    calculator are consistent with passing the `angular_cutoff` parameter, and the
    angular order of the input density.
    """
    frames = h2o_isolated()

    # Initialize the calculator with only max_angular = SPHEX_HYPERS["max_angular"] * 2.
    # We will cutoff off the angular channels at 3 for all intermediate iterations, and
    # only on the final iteration do the full product, doubling the max angular order.
    n_correlations = 2
    calculator = DensityCorrelations(
        n_correlations=n_correlations,
        max_angular=SPHEX_HYPERS_SMALL["max_angular"] * 2,
    )

    # Generate density
    density = spherical_expansion_small(frames)

    # Perform 3 iterations of DensityCorrelations with `angular_cutoff`
    nu_4 = calculator.compute(
        density,
        angular_cutoff=SPHEX_HYPERS_SMALL["max_angular"],
        selected_keys=None,
    )

    assert np.all(
        np.sort(np.unique(nu_4.keys.column("o3_lambda")))
        == np.arange(SPHEX_HYPERS_SMALL["max_angular"] + 1)
    )


def test_angular_cutoff_with_selected_keys():
    """
    Tests that the angular channels present in the output of a DensityCorrelations
    calculator are consistent with passing both the `angular_cutoff` and `selected_keys`
    parameters, and the angular order of the input density.
    """
    frames = h2o_isolated()

    # Initialize the calculator with only max_angular = SPHEX_HYPERS["max_angular"] * 2.
    # We will cutoff off the angular channels at 3 for all intermediate iterations, and
    # only on the final iteration do the full product, doubling the max angular order.
    calculator = DensityCorrelations(
        n_correlations=2,
        max_angular=SPHEX_HYPERS_SMALL["max_angular"] * 2,
    )

    # Generate density
    density = spherical_expansion_small(frames)

    # Perform 3 iterations of DensityCorrelations with `angular_cutoff`
    nu_4 = calculator.compute(
        density,
        angular_cutoff=SPHEX_HYPERS_SMALL[
            "max_angular"
        ],  # applies to all intermediate steps as selected_keys is specified
        selected_keys=Labels(
            names=["o3_lambda"],
            values=np.arange(5).reshape(-1, 1),
        ),
    )

    assert np.all(np.sort(np.unique(nu_4.keys.column("o3_lambda"))) == np.arange(5))


def test_no_error_with_correct_angular_selection():
    """
    Tests that initializing a DensityCorrelations calculator with a certain
    ``max_angular`` raises no error when passing the same ``angular_cutoff`` and
    ``selected_keys`` to the compute function.
    """
    frames = h2o_isolated()
    nu_1 = spherical_expansion(frames)

    # Initialize the calculator with only max_angular = SPHEX_HYPERS["max_angular"]
    max_angular = SPHEX_HYPERS["max_angular"]
    density_correlations = DensityCorrelations(
        n_correlations=2,
        max_angular=max_angular,
    )

    # If `angular_cutoff` and `selected_keys` were not passed, this should error as
    # max_angular = SPHEX_HYPERS["max_angular"] * 3 would be required.
    density_correlations.compute(
        nu_1,
        angular_cutoff=max_angular,
        selected_keys=Labels(
            names=["o3_lambda"],
            values=np.arange(max_angular).reshape(-1, 1),
        ),
    )
