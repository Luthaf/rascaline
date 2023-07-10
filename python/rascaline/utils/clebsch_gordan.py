"""
Module for computing Clebsch-gordan iterations with equistore TensorMaps.
"""
import itertools
from typing import Optional, Sequence

import ase
import numpy as np

import equistore
from equistore.core import Labels, TensorBlock, TensorMap
import rascaline

import wigners


# TODO:
# - [ ] Add support for dense operation
# - [ ] Account for body-order multiplicity


# ===== Class for calculating Clebsch-Gordan coefficients =====


class ClebschGordanReal:
    """
    Class for computing Clebsch-Gordan coefficients for real spherical
    harmonics.
    """

    def __init__(self, l_max: int):
        self.l_max = l_max
        self.coeffs = ClebschGordanReal.build_coeff_dict(self.l_max)

    @staticmethod
    def build_coeff_dict(l_max: int):
        """
        Builds a dictionary of Clebsch-Gordan coefficients for all possible
        combination of l1 and l2, up to l_max.
        """
        # real-to-complex and complex-to-real transformations as matrices
        r2c = {}
        c2r = {}
        coeff_dict = {}
        for L in range(0, l_max + 1):
            r2c[L] = _real2complex(L)
            c2r[L] = np.conjugate(r2c[L]).T

        for l1 in range(l_max + 1):
            for l2 in range(l_max + 1):
                for L in range(max(l1, l2) - min(l1, l2), min(l_max, (l1 + l2)) + 1):
                    complex_cg = _complex_clebsch_gordan_matrix(l1, l2, L)

                    real_cg = (r2c[l1].T @ complex_cg.reshape(2 * l1 + 1, -1)).reshape(
                        complex_cg.shape
                    )

                    real_cg = real_cg.swapaxes(0, 1)
                    real_cg = (r2c[l2].T @ real_cg.reshape(2 * l2 + 1, -1)).reshape(
                        real_cg.shape
                    )
                    real_cg = real_cg.swapaxes(0, 1)

                    real_cg = real_cg @ c2r[L].T

                    if (l1 + l2 + L) % 2 == 0:
                        rcg = np.real(real_cg)
                    else:
                        rcg = np.imag(real_cg)

                    new_cg = []
                    for M in range(2 * L + 1):
                        cg_nonzero = np.where(np.abs(rcg[:, :, M]) > 1e-15)
                        cg_M = np.zeros(
                            len(cg_nonzero[0]),
                            dtype=[("m1", ">i4"), ("m2", ">i4"), ("cg", ">f8")],
                        )
                        cg_M["m1"] = cg_nonzero[0]
                        cg_M["m2"] = cg_nonzero[1]
                        cg_M["cg"] = rcg[cg_nonzero[0], cg_nonzero[1], M]
                        new_cg.append(cg_M)

                    coeff_dict[(l1, l2, L)] = new_cg

        return coeff_dict


def _real2complex(L: int) -> np.ndarray:
    """
    Computes a matrix that can be used to convert from real to complex-valued
    spherical harmonics(coefficients) of order L.
    It's meant to be applied to the left, ``real2complex @ [-L..L]``.
    """
    result = np.zeros((2 * L + 1, 2 * L + 1), dtype=np.complex128)

    I_SQRT_2 = 1.0 / np.sqrt(2)

    for m in range(-L, L + 1):
        if m < 0:
            result[L - m, L + m] = I_SQRT_2 * 1j * (-1) ** m
            result[L + m, L + m] = -I_SQRT_2 * 1j

        if m == 0:
            result[L, L] = 1.0

        if m > 0:
            result[L + m, L + m] = I_SQRT_2 * (-1) ** m
            result[L - m, L + m] = I_SQRT_2

    return result


def _complex_clebsch_gordan_matrix(l1, l2, L):
    r"""clebsch-gordan matrix
    Computes the Clebsch-Gordan (CG) matrix for
    transforming complex-valued spherical harmonics.
    The CG matrix is computed as a 3D array of elements
        < l1 m1 l2 m2 | L M >
    where the first axis loops over m1, the second loops over m2,
    and the third one loops over M. The matrix is real.
    For example, using the relation:
        | l1 l2 L M > = \sum_{m1, m2} <l1 m1 l2 m2 | L M > | l1 m1 > | l2 m2 >
    (https://en.wikipedia.org/wiki/Clebsch–Gordan_coefficients, section
    "Formal definition of Clebsch-Gordan coefficients", eq 2)
    one can obtain the spherical harmonics L from two sets of
    spherical harmonics with l1 and l2 (up to a normalization factor).
    E.g.:
    Args:
        l1: l number for the first set of spherical harmonics
        l2: l number for the second set of spherical harmonics
        L: l number For the third set of spherical harmonics
    Returns:
        cg: CG matrix for transforming complex-valued spherical harmonics
    >>> from scipy.special import sph_harm
    >>> import numpy as np
    >>> import wigners
    ...
    >>> C_112 = _complex_clebsch_gordan_matrix(1, 1, 2)
    >>> comp_sph_1 = np.array([
    ... sph_harm(m, 1, 0.2, 0.2) for m in range(-1, 1+1)
    ... ])
    >>> comp_sph_2 = np.array([sph_harm(m, 1, 0.2, 0.2) for m in range(-1, 1+1)])
    >>> # obtain the (unnormalized) spherical harmonics
    >>> # with l = 2 by contraction over m1 and m2
    >>> comp_sph_2_u = np.einsum("ijk,i,j->k", C_112, comp_sph_1, comp_sph_2)
    >>> # we can check that they differ from the spherical harmonics
    >>> # by a constant factor
    >>> comp_sph_2 = np.array([sph_harm(m, 2, 0.2, 0.2) for m in range(-2, 2+1)])
    >>> ratio = comp_sph_2 / comp_sph_2_u
    >>> np.allclose(ratio[0], ratio)
    True
    """
    if np.abs(l1 - l2) > L or np.abs(l1 + l2) < L:
        return np.zeros((2 * l1 + 1, 2 * l2 + 1, 2 * L + 1), dtype=np.double)
    else:
        return wigners.clebsch_gordan_array(l1, l2, L)


# ===== Methods for performing CG combinations =====


# ===== Fxns for combining multi center descriptors =====


def _combine_multi_centers(
    tensor_1: TensorMap,
    tensor_2: TensorMap,
    lambdas: Sequence[int],
    cg_cache,
    use_sparse: bool = True,
):
    """ """


def _combine_multi_centers_block_pair(
    block_1: TensorBlock,
    block_2: TensorBlock,
    lamb: int,
    cg_cache,
    use_sparse: bool = True,
):
    """ """


# ===== Fxns for combining single center descriptors =====


def n_body_iteration(
    frames: Sequence[ase.Atoms],
    rascal_hypers: dict,
    nu: int,
    lambdas: int,
    cg_cache,
    use_sparse: bool = True,
    intermediate_lambda_max: Optional[int] = None,
    selected_samples: Optional[Labels] = None,
) -> TensorMap:
    """
    Based on the passed ``rascal_hypers``, generates a rascaline
    SphericalExpansion (i.e. nu = 1 body order descriptor) and combines it
    iteratively to generate a descriptor of order ``nu``.

    The returned TensorMap will only contain blocks with angular channels of
    target order lambda corresponding to those passed in ``lambdas``.

    Passing ``intermediate_lambda_max`` will place a maximum on the angular
    order of blocks created by combination at each CG combination step.
    """
    # Generate rascaline SphericalExpansion
    calculator = rascaline.SphericalExpansion(**rascal_hypers)
    nu1 = calculator.compute(frames, selected_samples=selected_samples)
    nu1 = nu1.keys_to_properties("species_neighbor")

    # Standardize metadata
    nu1 = _add_nu_sigma_to_key_names(nu1)

    return _combine_single_center_to_body_order(
        nu1, nu1, lambdas, nu_target=2, cg_cache=cg_cache, use_sparse=use_sparse
    )


def _combine_single_center_to_body_order(
    tensor_1: TensorMap,
    tensor_2: TensorMap,
    lambdas: Sequence[int],
    nu_target: int,
    cg_cache,
    use_sparse: bool = True,
) -> TensorMap:
    """
    Assumes standradized metadata:

    Key names are:

    ["order_nu", "inversion_sigma", "spherical_harmonics_l", "species_center",
    "l1", "l2", ...]

    Samples of pairs of blocks corresponding to the same chemical species are
    equivalent in the two TensorMaps. Samples names are ["structure", "center"]

    Components names are [["spherical_harmonics_m"],] for each block.

    Property names are ["n1", "n2", ..., "species_neighbor_1",
    "species_neighbor_2", ...] for each block.
    """
    assert nu_target == 2  # only implemented for nu = 2

    combined_tensor = _combine_single_center(
        tensor_1, tensor_2, lambdas, cg_cache, use_sparse
    )

    # Now we have reached the desired body order:

    # TODO: Account for body-order multiplicity
    combined_tensor = _apply_body_order_corrections(combined_tensor)

    # Move the [l1, l2, ...] keys to the properties
    combined_tensor = combined_tensor.keys_to_properties(
        ["l" + str(i) for i in range(1, len(combined_tensor.keys.names) - 3)]
    )

    return combined_tensor


def _combine_single_center(
    tensor_1: TensorMap,
    tensor_2: TensorMap,
    lambdas: Sequence[int],
    cg_cache,
    use_sparse: bool = True,
) -> TensorMap:
    """
    For 2 TensorMaps with body orders nu and eta respectively, combines their
    blocks to form a new TensorMap with body order (nu + eta). Returns blocks
    only with angular orders corresponding to those passed in ``lambdas``.
    """
    # Check metadata
    if not equistore.equal_metadata(tensor_1, tensor_2, check=["samples"]):
        raise ValueError(
            "TensorMap pair to combine must have equal samples in the same order"
        )
    # Get the correct keys for the combined TensorMap
    combined_keys = _create_combined_keys(tensor_1.keys, tensor_2.keys, lambdas)

    # Iterate over pairs of blocks and combine
    combined_blocks = []
    for combined_key in combined_keys:
        # Extract the pair of blocks to combine
        block_1 = tensor_1.block(
            spherical_harmonics_l=combined_key["l1"],
            species_center=combined_key["species_center"],
        )
        block_2 = tensor_1.block(
            spherical_harmonics_l=combined_key["l2"],
            species_center=combined_key["species_center"],
        )

        # Extract the desired lambda value
        lamb = combined_key["spherical_harmonics_l"]

        # Combine the blocks into a new TensorBlock
        combined_blocks.append(
            _combine_single_center_block_pair(
                block_1, block_2, lamb, cg_cache, use_sparse
            )
        )
    # Construct our combined TensorMap
    combined_tensor = TensorMap(combined_keys, combined_blocks)

    return combined_tensor


def _combine_single_center_block_pair(
    block_1: TensorBlock,
    block_2: TensorBlock,
    lamb: int,
    cg_cache,
    use_sparse: bool = True,
) -> TensorBlock:
    """
    For a given pair of TensorBlocks and desired lambda value, combines the
    values arrays and returns in a new TensorBlock.
    """
    # Check metadata
    if block_1.properties.names != block_2.properties.names:
        raise ValueError(
            "TensorBlock pair to combine must have equal properties in the same order"
        )

    # Do the CG combination - no shape pre-processing required
    # print(block_1.values.shape, block_2.values.shape, lamb)
    combined_values = _clebsch_gordan_combine(
        block_1.values, block_2.values, lamb, cg_cache
    )

    # Create a TensorBlock
    combined_block = TensorBlock(
        values=combined_values,
        samples=block_1.samples,
        components=[
            Labels(
                names=["spherical_harmonics_m"],
                values=np.arange(-lamb, lamb + 1).reshape(-1, 1),
            ),
        ],
        properties=Labels(
            names=["n1", "n2", "species_neighbor_1", "species_neighbor_2"],
            values=np.array(
                [
                    [n1, n2, neighbor_1, neighbor_2]
                    for (n2, neighbor_2) in block_2.properties.values
                    for (n1, neighbor_1) in block_1.properties.values
                ]
            ),
        ),
    )

    return combined_block


# ===== Mathematical manipulation fxns


def _clebsch_gordan_combine(
    arr_1: np.ndarray,
    arr_2: np.ndarray,
    lamb: int,
    cg_cache,
    use_sparse: bool = True,
) -> np.ndarray:
    """
    Couples arrays corresponding to the irreducible spherical components of 2
    angular channels l1 and l2 using the appropriate Clebsch-Gordan
    coefficients. As l1 and l2 can be combined to form multiple lambda channels,
    this function returns the coupling to a single specified channel `lambda`.

    `arr_1` has shape (n_i, 2 * l1 + 1, n_p) and `arr_2` has shape (n_i, 2 * l2
    + 1, n_q). n_i is the number of samples, n_p and n_q are the number of
    properties in each array. The number of samples in each array must be the
    same.

    The ouput array has shape (n_i, 2 * lambda + 1, n_p * n_q), where lambda is
    the input parameter `lamb`.

    The Clebsch-Gordan coefficients are cached in `cg_cache`. Currently, these
    must be produced by the ClebschGordanReal class in this module.

    Either performs the operation in a dense or sparse manner, depending on the
    value of `sparse`.
    """
    # Check the first dimension of the arrays are the same (i.e. same samples)
    if use_sparse:
        return _clebsch_gordan_combine_sparse(arr_1, arr_2, lamb, cg_cache)
    return _clebsch_gordan_dense(arr_1, arr_2, lamb, cg_cache)


def _clebsch_gordan_combine_sparse(
    arr_1: np.ndarray,
    arr_2: np.ndarray,
    lamb: int,
    cg_cache,
) -> np.ndarray:
    """
    TODO: docstring
    """
    # Samples dimensions must be the same
    assert arr_1.shape[0] == arr_2.shape[0]

    # Define other useful dimensions
    n_i = arr_1.shape[0]  # number of samples
    n_p = arr_1.shape[2]  # number of properties in arr_1
    n_q = arr_2.shape[2]  # number of properties in arr_2

    # Infer l1 and l2 from the len of the lenght of axis 1 of each tensor
    l1 = (arr_1.shape[1] - 1) // 2
    l2 = (arr_2.shape[1] - 1) // 2

    # Get the corresponding Clebsch-Gordan coefficients
    cg_coeffs = cg_cache.coeffs[(l1, l2, lamb)]

    # Initialise output array
    arr_out = np.zeros((n_i, 2 * lamb + 1, n_p * n_q))

    # Fill in each mu component of the output array in turn
    for mu in range(2 * lamb + 1):
        # Iterate over the Clebsch-Gordan coefficients for this mu
        for m1, m2, cg_coeff in cg_coeffs[mu]:
            # Broadcast arrays, multiply together and with CG coeff
            arr_out[:, mu, :] = (
                arr_1[:, m1, :, None] * arr_2[:, m2, None, :] * cg_coeff
            ).reshape(n_i, n_p * n_q)

    return arr_out


def _clebsch_gordan_combine_dense(
    arr_1: np.ndarray,
    arr_2: np.ndarray,
    lamb: int,
    cg_cache,
) -> np.ndarray:
    """
    arr_1,#: Array[samples, 2 * l1 + 1, q_properties], # mu values for l1
    arr_2,#: Array[samples, 2 * l2 + 1, p_properties], # mu values for l2
    lamb: int,
    cg_cache,#: Array[(2 * l1 +1) * (2 * l2 +1), (2 * lamb + 1)]
    ) -> None: #Array[samples, 2 * lamb + 1, q_properties * p_properties]:

    :param arr_1: array with the mu values for l1 with shape [samples, 2 * l1 + 1, q_properties]
    :param arr_2: array with the mu values for l1 with shape [samples, 2 * l2 + 1, p_properties]
    :param lamb: int resulting coupled channel
    :param cg_cache: array of shape [(2 * l1 +1) * (2 * l2 +1), (2 * lamb + 1)]
    :returns lam_mu_values: array of shape [samples, (2 * l1 + 1)* (2 * l2 + 1), q_properties, p_properties]

    >>> N_SAMPLES = 30
    >>> N_Q_PROPERTIES = 10
    >>> N_P_PROPERTIES = 8
    >>> L1 = 2
    >>> L2 = 3
    >>> LAM = 2
    >>> arr_1 = np.random.rand(N_SAMPLES, 2*L1+1, N_Q_PROPERTIES)
    >>> arr_2 = np.random.rand(N_SAMPLES, 2*L2+1, N_P_PROPERTIES)
    >>> cg_cache = np.random.rand(2*L1+1, 2*L2+1, 2*LAM+1)
    >>> out1 = _clebsch_gordan_dense(arr_1, arr_2, LAM, cg_cache)
    >>> out2 = np.einsum("slq, skp, lkL -> sLqp", arr_1, arr_2, cg_cache).reshape(arr_1.shape[0], 2*LAM+1, -1)
    >>> print(np.allclose(out1, out2))
    True
    """
    # l1_mu q, samples l2_mu p -> samples l2_mu p l1_mu q
    # we broadcast it in this way so we only need to do one swapaxes in the next step
    out = arr_1[:, None, None, :, :] * arr_2[:, :, :, None, None]
    # samples l2_mu p l1_mu q -> samples q p l1_mu l2_mu
    out = out.swapaxes(1, 4)
    # samples q p l1_mu l2_mu ->  samples (q p) (l1_mu l2_mu)
    out = out.reshape(
        -1, arr_1.shape[2] * arr_2.shape[2], arr_1.shape[1] * arr_2.shape[1]
    )
    # l1_mu l2_mu lam_mu -> (l1_mu l2_mu) lam_mu
    cg_cache = cg_cache.reshape(-1, 2 * lamb + 1)
    # samples (q p) (l1_mu l2_mu), (l1_mu l2_mu) lam_mu -> samples (q p) lam_mu
    out = out @ cg_cache.reshape(-1, 2 * lamb + 1)
    # samples (q p) lam_mu -> samples lam_mu (q p)
    return out.swapaxes(1, 2)


def _apply_body_order_corrections(tensor: TensorMap) -> TensorMap:
    """
    Applies the appropriate prefactors to the block values of the output
    TensorMap (i.e. post-CG combination) according to its body order.
    """
    return tensor


# Commented out but left as reference. Not needed as we are just writing an
# end-to-end pipeline for SphericalExpansion -> NICE.
# def _check_nu_combination_valid() -> bool:
#     """ """
#     #     # Check "order_nu" of each TM to see that it can iteratively add to nu
#     #     nu1 = np.unique(tensor_1.keys.column("order_nu"))
#     #     nu2 = np.unique(tensor_2.keys.column("order_nu"))
#     #     assert len(nu1) != 1
#     #     assert len(nu2) != 1
#     #     nu1, nu2 = nu1[0], nu2[0]
#     #     assert _check_nu_combination_valid(nu1, nu2, nu)
#     return True


# ===== Fxns to manipulate metadata of TensorMaps =====


def _create_combined_keys(
    keys_1: Labels, keys_2: Labels, lambdas: Sequence[int]
) -> Sequence[Labels]:
    """
    Given the keys of 2 TensorMaps and a list of desired lambda values, creates
    the correct keys for the TensorMap returned after one CG combination.

    The input keys `keys_1` and `keys_2` must follow the key name convention:

    ["order_nu", "inversion_sigma", "spherical_harmonics_l", "species_center",
    "l1", "l2", ...]

    Returned is a Labels object corresponding to the appropriate keys of the
    output TensorMap created by a CG combination step.
    """
    # Check key names match internal convention
    for keys in [keys_1, keys_2]:
        assert np.all(
            keys.names[:4]
            == [
                "order_nu",
                "inversion_sigma",
                "spherical_harmonics_l",
                "species_center",
            ]
        )
    # Iteratively check the remaining key names are "l1", "l2", ... We will need
    # these counters later to name the output keys.
    l_counter_1, l_counter_2 = 0, 0

    # First do for keys_1
    for name in keys_1.names[4:]:
        assert name[0] == "l"
        l_counter_1 += 1
        assert int(name[1:]) == l_counter_1

    # Then do for keys_2
    for name in keys_2.names[4:]:
        assert name[0] == "l"
        l_counter_2 += 1
        assert int(name[1:]) == l_counter_2

    # Find the pair product of the keys
    combined_vals = set()
    for key_1, key_2 in itertools.product(keys_1, keys_2):
        # Unpack relevant key values
        nu1, sig1, lam1, a = key_1.values[:4]
        nu2, sig2, lam2, a2 = key_2.values[:4]

        # Only combine blocks of the same chemical species
        if a != a2:
            continue

        # Skip redundant lower triangle of (lam1, lam2) combinations
        if lam1 < lam2:
            continue

        # Get the list of previous l values from each key
        l_list_1, l_list_2 = key_1.values[4:].tolist(), key_2.values[4:].tolist()

        # Calculate new nu
        nu = nu1 + nu2

        # Only combine to create blocks of desired lambda values
        nonzero_lams = np.arange(abs(lam1 - lam2), abs(lam1 + lam2) + 1)
        for lam in nonzero_lams:
            if lam not in lambdas:
                continue

            # Calculate new sigma
            sig = sig1 * sig2 * (-1) ** (lam1 + lam2 + lam)

            # Create a key tuple, ordering the "l1", "l2", etc values
            l_list = [lam1] + l_list_1 + [lam2] + l_list_2
            key_tuple = (nu, sig, lam, a, *l_list)
            combined_vals.add(key_tuple)

    # Create the output key names
    new_names = [
        "order_nu",
        "inversion_sigma",
        "spherical_harmonics_l",
        "species_center",
    ] + [f"l{i}" for i in range(1, l_counter_1 + l_counter_2 + 3)]

    return Labels(
        names=new_names,
        values=np.array(list(combined_vals)),
    )


def _add_nu_sigma_to_key_names(tensor: TensorMap) -> TensorMap:
    """
    Prepends key names "order_nu" and "inversion_sigma" respectively to the key
    names of ``tensor``.

    For instance, if `tensor.keys.names` is ["spherical_harmonics_l",
    "species_center"], the returned tensor will have keys with names
    ["order_nu", "inversion_sigma", "spherical_harmonics_l", "species_center"].
    """
    keys = tensor.keys
    prepend_list = []
    if "inversion_sigma" in keys.names:
        assert keys.names.index("inversion_sigma") == 1
    else:
        prepend_list = [1] + prepend_list
    if "order_nu" in keys.names:
        assert keys.names.index("order_nu") == 0
    else:
        prepend_list = [1] + prepend_list

    new_keys = Labels(
        names=["order_nu", "inversion_sigma"] + keys.names,
        values=np.array([prepend_list + key_list for key_list in keys.values.tolist()]),
    )
    new_blocks = [block.copy() for block in tensor]

    return TensorMap(keys=new_keys, blocks=new_blocks)
