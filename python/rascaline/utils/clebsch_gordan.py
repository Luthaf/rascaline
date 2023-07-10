"""
Module for computing Clebsch-gordan iterations with equistore TensorMaps.
"""
import itertools
from typing import Optional, Sequence, Tuple

import ase
import numpy as np

import equistore
from equistore.core import Labels, TensorBlock, TensorMap
import rascaline

import wigners


# TODO:
# - [ ] Add support for dense operation
# - [ ] Account for body-order multiplicity
# - [ ] Gradients?


# ===== Class for calculating Clebsch-Gordan coefficients =====


class ClebschGordanReal:
    """
    Class for computing Clebsch-Gordan coefficients for real spherical
    harmonics.

    Stores the coefficients in a dictionary in the `self.coeffs` attribute,
    which is built at initialization. This dictionary has the form:

    {
        (l1, l2, lambda): [
            np.ndarray([m1, m2, cg]),
            ...
            for m1 in range(-l1, l1 + 1),
            for m2 in range(-l2, l2 + 1),
        ],
        ...
        for lambda in range(0, l)
    }

    where `cg`, i.e. the third value in each array, is the Clebsch-Gordan
    coefficient that describes the combination of the `m1` irreducible
    component of the `l1` angular channel and the `m2` irreducible component of
    the `l2` angular channel into the irreducible tensor of order `lambda`.
    """

    def __init__(self, lambda_max: int):
        self.lambda_max = lambda_max
        self.coeffs = ClebschGordanReal.build_coeff_dict(self.lambda_max)

    @staticmethod
    def build_coeff_dict(lambda_max: int):
        """
        Builds a dictionary of Clebsch-Gordan coefficients for all possible
        combination of l1 and l2, up to lambda_max.
        """
        # real-to-complex and complex-to-real transformations as matrices
        r2c = {}
        c2r = {}
        coeff_dict = {}
        for lam in range(0, lambda_max + 1):
            r2c[lam] = _real2complex(lam)
            c2r[lam] = np.conjugate(r2c[lam]).T

        for l1 in range(lambda_max + 1):
            for l2 in range(lambda_max + 1):
                for lam in range(
                    max(l1, l2) - min(l1, l2), min(lambda_max, (l1 + l2)) + 1
                ):
                    complex_cg = _complex_clebsch_gordan_matrix(l1, l2, lam)

                    real_cg = (r2c[l1].T @ complex_cg.reshape(2 * l1 + 1, -1)).reshape(
                        complex_cg.shape
                    )

                    real_cg = real_cg.swapaxes(0, 1)
                    real_cg = (r2c[l2].T @ real_cg.reshape(2 * l2 + 1, -1)).reshape(
                        real_cg.shape
                    )
                    real_cg = real_cg.swapaxes(0, 1)

                    real_cg = real_cg @ c2r[lam].T

                    if (l1 + l2 + lam) % 2 == 0:
                        rcg = np.real(real_cg)
                    else:
                        rcg = np.imag(real_cg)

                    new_cg = []
                    for mu in range(2 * lam + 1):
                        cg_nonzero = np.where(np.abs(rcg[:, :, mu]) > 1e-15)
                        cg_M = np.zeros(
                            len(cg_nonzero[0]),
                            dtype=[("m1", ">i4"), ("m2", ">i4"), ("cg", ">f8")],
                        )
                        cg_M["m1"] = cg_nonzero[0]
                        cg_M["m2"] = cg_nonzero[1]
                        cg_M["cg"] = rcg[cg_nonzero[0], cg_nonzero[1], mu]
                        new_cg.append(cg_M)

                    coeff_dict[(l1, l2, lam)] = new_cg

        return coeff_dict


def _real2complex(lam: int) -> np.ndarray:
    """
    Computes a matrix that can be used to convert from real to complex-valued
    spherical harmonics(coefficients) of order ``lam``.

    It's meant to be applied to the left, ``real2complex @ [-lam, ..., +lam]``.
    """
    result = np.zeros((2 * lam + 1, 2 * lam + 1), dtype=np.complex128)

    I_SQRT_2 = 1.0 / np.sqrt(2)

    for m in range(-lam, lam + 1):
        if m < 0:
            result[lam - m, lam + m] = I_SQRT_2 * 1j * (-1) ** m
            result[lam + m, lam + m] = -I_SQRT_2 * 1j

        if m == 0:
            result[lam, lam] = 1.0

        if m > 0:
            result[lam + m, lam + m] = I_SQRT_2 * (-1) ** m
            result[lam - m, lam + m] = I_SQRT_2

    return result


def _complex_clebsch_gordan_matrix(l1, l2, lam):
    r"""clebsch-gordan matrix
    Computes the Clebsch-Gordan (CG) matrix for
    transforming complex-valued spherical harmonics.
    The CG matrix is computed as a 3D array of elements
        < l1 m1 l2 m2 | lam mu >
    where the first axis loops over m1, the second loops over m2,
    and the third one loops over mu. The matrix is real.
    For example, using the relation:
        | l1 l2 lam mu > = \sum_{m1, m2} <l1 m1 l2 m2 | lam mu > | l1 m1 > | l2 m2 >
    (https://en.wikipedia.org/wiki/Clebsch–Gordan_coefficients, section
    "Formal definition of Clebsch-Gordan coefficients", eq 2)
    one can obtain the spherical harmonics lam from two sets of
    spherical harmonics with l1 and l2 (up to a normalization factor).
    E.g.:
    Args:
        l1: l number for the first set of spherical harmonics
        l2: l number for the second set of spherical harmonics
        lam: l number For the third set of spherical harmonics
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
    if np.abs(l1 - l2) > lam or np.abs(l1 + l2) < lam:
        return np.zeros((2 * l1 + 1, 2 * l2 + 1, 2 * lam + 1), dtype=np.double)
    else:
        return wigners.clebsch_gordan_array(l1, l2, lam)


# ===== Methods for performing CG combinations =====


# ===== Fxns for combining multi center descriptors =====


def _combine_multi_centers(
    tensor_1: TensorMap,
    tensor_2: TensorMap,
    lambdas: Sequence[int],
    cg_cache,
    use_sparse: bool = True,
) -> TensorMap:
    """ """
    raise NotImplementedError


def _combine_multi_centers_block_pair(
    block_1: TensorBlock,
    block_2: TensorBlock,
    lam: int,
    cg_cache,
    use_sparse: bool = True,
) -> TensorMap:
    """ """
    raise NotImplementedError


# ===== Fxns for combining single center descriptors =====


def n_body_iteration_single_center(
    frames: Sequence[ase.Atoms],
    rascal_hypers: dict,
    nu_target: int,
    lambdas: Sequence[int],
    use_sparse: bool = True,
    intermediate_lambda_max: Optional[int] = None,
    selected_samples: Optional[Labels] = None,
    species_neighbors: Optional[Sequence[int]] = None,
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
    if nu_target > 2:  # only implemented for nu = 2
        raise NotImplementedError(
            "currently CG iterations only implemented for max body order nu=2"
        )
    if intermediate_lambda_max is None:
        # TODO: figure out a better default? Setting this to the maximum
        # possible value by default (as is currently done here) can blow up the
        # memory.
        intermediate_lambda_max = nu_target * np.max(rascal_hypers["max_angular"])
    else:
        # The maximum value intermediate_lambda_max value is dictated by the
        # "max_angular" value in the rascal_hypers and the target body order.
        # Check the passed value is valid.
        if intermediate_lambda_max > nu_target * np.max(rascal_hypers["max_angular"]):
            raise ValueError(
                "intermediate_lambda_max must be less than nu_target * max_angular"
            )

    # Generate a rascaline SphericalExpansion, for only the selected samples if
    # applicable
    calculator = rascaline.SphericalExpansion(**rascal_hypers)
    nu1 = calculator.compute(frames, selected_samples=selected_samples)

    # Move the "species_neighbor" key to the properties. If species_neighbors is
    # passed as a list of int, sparsity can be created in the properties for
    # these species.
    if species_neighbors is None:
        keys_to_move = "species_neighbor"
    else:
        keys_to_move = Labels(
            names=["species_neighbor"],
            values=np.array(species_neighbors).reshape(-1, 1),
        )
    nu1 = nu1.keys_to_properties(keys_to_move=keys_to_move)

    # Standardize the key names metadata
    nu1 = _add_nu_sigma_to_key_names(nu1)

    # Define the cached CG coefficients - currently only sparse CG matrices implemented
    if use_sparse:
        cg_cache = ClebschGordanReal(lambda_max=intermediate_lambda_max)
    else:
        raise NotImplementedError(
            "currently CG iterations only implemented for use_sparse=True"
        )

    # TODO: Combine to the desired body order iteratively. Currently only a
    # single CG iteration to body order nu = 2 is implemented.
    combined_tensor = _combine_single_center(nu1, nu1, lambdas, cg_cache, use_sparse)

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
    For 2 TensorMaps, ``tensor_1`` and ``tensor_2``, with body orders nu and eta
    respectively, combines their blocks to form a new TensorMap with body order
    (nu + eta). Returns blocks only with angular orders corresponding to those
    passed in ``lambdas``.

    Assumes the metadata of the two TensorMaps are standaradized as follows.

    Key names are: ["order_nu", "inversion_sigma", "spherical_harmonics_l",
    "species_center", "l1", "l2", ...].

    Samples of pairs of blocks corresponding to the same chemical species are
    equivalent in the two TensorMaps. Samples names are ["structure", "center"]

    Components names are [["spherical_harmonics_m"],] for each block.

    Property names are ["n1", "n2", ..., "species_neighbor_1",
    "species_neighbor_2", ...] for each block.
    """
    # Check metadata
    if not equistore.equal_metadata(tensor_1, tensor_2, check=["samples"]):
        raise ValueError(
            "TensorMaps `tensor_1` and `tensor_2` to combine must have equivalent keys "
            "(order agnostic), and equal samples (same order)"
        )

    # Get the correct keys for the combined TensorMap
    combined_keys, combination_infos = _create_combined_keys(
        tensor_1.keys, tensor_2.keys, lambdas
    )

    # Iterate over pairs of blocks and combine
    combined_blocks = []
    for combined_key, combination_info in zip(combined_keys, combination_infos):
        # Extract the pair of blocks to combine. The lambda values of the block
        # pair being combined are stored in `combination_info`.
        block_1 = tensor_1.block(
            spherical_harmonics_l=combination_info[0],
            species_center=combined_key["species_center"],
        )
        block_2 = tensor_2.block(
            spherical_harmonics_l=combination_info[1],
            species_center=combined_key["species_center"],
        )

        # Combine the blocks into a new TensorBlock of the correct lambda order.
        # Pass the correction factor accounting for the redundancy of "lx"
        # combinations.
        combined_blocks.append(
            _combine_single_center_block_pair(
                block_1,
                block_2,
                combined_key["spherical_harmonics_l"],
                cg_cache,
                use_sparse,
                correction_factor=combination_info[2],
            )
        )

    return TensorMap(combined_keys, combined_blocks)


def _combine_single_center_block_pair(
    block_1: TensorBlock,
    block_2: TensorBlock,
    lam: int,
    cg_cache,
    use_sparse: bool = True,
    correction_factor: float = 1.0,
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
    # print(block_1.values.shape, block_2.values.shape, lam)
    combined_values = _clebsch_gordan_combine(
        block_1.values, block_2.values, lam, cg_cache
    )

    # Create a TensorBlock
    combined_block = TensorBlock(
        values=combined_values * correction_factor,
        samples=block_1.samples,
        components=[
            Labels(
                names=["spherical_harmonics_m"],
                values=np.arange(-lam, lam + 1).reshape(-1, 1),
            ),
        ],
        # TODO: account for more "species_neighbor_x" and "nx", i.e. for higher
        # body order
        properties=Labels(
            names=["n1", "n2", "species_neighbor_1", "species_neighbor_2"],
            values=np.array(
                [
                    [n1, n2, neighbor_1, neighbor_2]
                    for (neighbor_2, n2) in block_2.properties.values
                    for (neighbor_1, n1) in block_1.properties.values
                ]
            ),
        ),
    )

    return combined_block


# ===== Mathematical manipulation fxns


def _clebsch_gordan_combine(
    arr_1: np.ndarray,
    arr_2: np.ndarray,
    lam: int,
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
    the input parameter `lam`.

    The Clebsch-Gordan coefficients are cached in `cg_cache`. Currently, these
    must be produced by the ClebschGordanReal class in this module.

    Either performs the operation in a dense or sparse manner, depending on the
    value of `sparse`.
    """
    # Check the first dimension of the arrays are the same (i.e. same samples)
    if use_sparse:
        return _clebsch_gordan_combine_sparse(arr_1, arr_2, lam, cg_cache)
    raise NotImplementedError
    # return _clebsch_gordan_dense(arr_1, arr_2, lam, cg_cache)


def _clebsch_gordan_combine_sparse(
    arr_1: np.ndarray,
    arr_2: np.ndarray,
    lam: int,
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

    # Infer l1 and l2 from the len of the length of axis 1 of each tensor
    l1 = (arr_1.shape[1] - 1) // 2
    l2 = (arr_2.shape[1] - 1) // 2

    # Get the corresponding Clebsch-Gordan coefficients
    cg_coeffs = cg_cache.coeffs[(l1, l2, lam)]

    # Initialise output array
    arr_out = np.zeros((n_i, 2 * lam + 1, n_p * n_q))

    # Fill in each mu component of the output array in turn
    for mu in range(2 * lam + 1):
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
    lam: int,
    cg_cache,
) -> np.ndarray:
    """
    arr_1,#: Array[samples, 2 * l1 + 1, q_properties], # mu values for l1
    arr_2,#: Array[samples, 2 * l2 + 1, p_properties], # mu values for l2
    lam: int,
    cg_cache,#: Array[(2 * l1 +1) * (2 * l2 +1), (2 * lam + 1)]
    ) -> None: #Array[samples, 2 * lam + 1, q_properties * p_properties]:

    :param arr_1: array with the mu values for l1 with shape [samples, 2 * l1 + 1, q_properties]
    :param arr_2: array with the mu values for l1 with shape [samples, 2 * l2 + 1, p_properties]
    :param lam: int resulting coupled channel
    :param cg_cache: array of shape [(2 * l1 +1) * (2 * l2 +1), (2 * lam + 1)]
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
    raise NotImplementedError
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
    cg_cache = cg_cache.reshape(-1, 2 * lam + 1)
    # samples (q p) (l1_mu l2_mu), (l1_mu l2_mu) lam_mu -> samples (q p) lam_mu
    out = out @ cg_cache.reshape(-1, 2 * lam + 1)
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
) -> Tuple[Labels, Sequence[Sequence[int]]]:
    """
    Given the keys of 2 TensorMaps and a list of desired lambda values, creates
    the correct keys for the TensorMap returned after one CG combination.

    The input keys `keys_1` and `keys_2` must follow the key name convention:

    ["order_nu", "inversion_sigma", "spherical_harmonics_l", "species_center",
    "l1", "l2", ...]

    Returned is a tuple.

    The first element is to the Labels object for the keys of the output
    TensorMap created by a CG combination step.

    The second element is a list of list of ints. Each sublist corresponds to
    [lam1, lam2, correction_factor terms]. lam1 and lam2 tracks the lambda
    values of the blocks that combine to form the block indexed by the
    corresponding key. The correction_factor terms are the prefactors that
    account for the redundancy in the CG combination.
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
    combined_vals = dict()
    for key_1, key_2 in itertools.product(keys_1, keys_2):
        # Unpack relevant key values
        nu1, sig1, lam1, a = key_1.values[:4]
        nu2, sig2, lam2, a2 = key_2.values[:4]

        # Only combine blocks of the same chemical species
        if a != a2:
            continue

        # Skip redundant lower triangle of (lam1, lam2) combinations
        # if lam1 < lam2:
        #     continue

        # Get the list of previous l values from each key
        l_list_1 = key_1.values[4:].tolist()
        l_list_2 = key_2.values[4:].tolist()

        # Calculate new nu
        nu = nu1 + nu2

        # Only combine to create blocks of desired lambda values
        nonzero_lams = np.arange(abs(lam1 - lam2), abs(lam1 + lam2) + 1)
        for lam in nonzero_lams:
            if lam not in lambdas:
                continue

            # Calculate new sigma
            sig = sig1 * sig2 * (-1) ** (lam1 + lam2 + lam)

            # Create a list of the "l1", "l2", ... etc values, in ascending
            # order such that l1 <= l2 <= l3 < ...
            l_list = [lam1] + [lam2] + l_list_1 + l_list_2
            l_list.sort()

            # Define the values for the new key and the lamda values of the
            # block pair that created it.
            key_tuple = (nu, sig, lam, a, *l_list)

            # The number of times this key combination has been made needs to
            # been counted for correction later.
            if combined_vals.get(key_tuple) is None:
                correction_factor = 1  # combination not seen before
            else:  # combination seen before; increment factor by 1
                correction_factor = combined_vals[key_tuple][2] + 1

            # Store in a dict such that duplicates are discarded, along with the
            # combination info.
            combined_vals[key_tuple] = [lam1, lam2, correction_factor]

    # Create the output key names
    new_names = [
        "order_nu",
        "inversion_sigma",
        "spherical_harmonics_l",
        "species_center",
    ] + [f"l{i}" for i in range(1, l_counter_1 + l_counter_2 + 3)]

    new_key_vals, combination_infos = [], []
    for new_key_val, combination_info in combined_vals.items():
        new_key_vals.append(new_key_val)
        combination_infos.append(combination_info)

    return (
        Labels(
            names=new_names,
            values=np.array(list(new_key_vals)),
        ),
        combination_infos,
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
