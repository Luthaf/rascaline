"""
Module for computing Clebsch-gordan iterations with metatensor TensorMaps.
"""
import itertools
from typing import Optional, Sequence, Tuple, Union

import ase
import numpy as np

import metatensor
from metatensor import Labels, TensorBlock, TensorMap
import rascaline

import wigners


# TODO:
# - [ ] Add support for dense operation
# - [ ] Account for body-order multiplicity
# - [ ] Gradients?
# - [ ] Unit tests
# - [ ] Check the combined keys - potential bug (!)


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

    def __init__(self, lambda_max: int, sparse: bool = True):
        self._lambda_max = lambda_max
        self._sparse = sparse
        self._coeffs = ClebschGordanReal.build_coeff_dict(
            self._lambda_max, self._sparse
        )

    @property
    def lambda_max(self):
        return self._lambda_max

    @property
    def sparse(self):
        return self._sparse

    @property
    def coeffs(self):
        return self._coeffs

    @staticmethod
    def build_coeff_dict(lambda_max: int, sparse: bool):
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
                        cg_l1l2lam = np.real(real_cg)
                    else:
                        cg_l1l2lam = np.imag(real_cg)

                    if sparse:
                        # if sparse we make a dictionary out of the matrix
                        nonzeros_cg_coeffs_idx = np.where(np.abs(cg_l1l2lam) > 1e-15)
                        cg_l1l2lam = {
                            (m1, m2, mu): cg_l1l2lam[m1, m2, mu]
                            for m1, m2, mu in zip(*nonzeros_cg_coeffs_idx)
                        }
                    coeff_dict[(l1, l2, lam)] = cg_l1l2lam

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


def lambda_soap_vector(
    frames: Sequence[ase.Atoms],
    rascal_hypers: dict,
    lambda_filter: Optional[Union[None, int, Sequence[int]]] = None,
    sigma_filter: Optional[Union[None, int, Sequence[int]]] = None,
    lambda_cut: Optional[int] = None,
    selected_samples: Optional[Labels] = None,
    species_neighbors: Optional[Sequence[int]] = None,
) -> TensorMap:
    """
    A higher-level wrapper for the :py:func:`n_body_iteration_single_center`
    function specifically for generating lambda-SOAP vectors in the metatensor
    format, with some added metadata manipulation.

    The hyperparameters `rascal_hypers` are used to generate a nu=1
    SphericalExpansion object with rascaline, and these are then combined with a
    single Clebsch-Gordan iteration step to form the nu=2 lambda-SOAP
    descriptor. Only the target spherical channels given in `lambdas` are
    calculated and returned.

    `lambda_cut` can be set to reduce the memory overhead of the calculation, at
    the cost of loss of information. The theoretical maximum (and default) value
    is nu_target * rascal_hypers["max_angular"], though a lower value can be
    set. `nu_target` is the target body-order of the descriptor (by definition
    nu=2 for lambda-SOAP). Using the default (and theoretical maximum) value can
    lead to memory blow-up for large systems and hgih body-orders, so this value
    needs to be tailored for the computation and system. Note that truncating
    this value to less than the default will lead to some information loss.

    If `sigmas` is passed, then only the specified sigmas are
    returned in the output TensorMap. For instance, passing as an int +1 means
    only blocks with even sigmas will be returned. If a dict, the sigmas kept
    for each target lambda can be specified. Any lambdasIf false, all blocks of both odd
    and even sigmas are returned. In the latter case, the output TensorMap will
    have a key dimension "inversion_sigma" that tracks the sigmas.
    """
    # Generate lambda-SOAP using rascaline.utils
    lsoap = n_body_iteration_single_center(
        frames,
        rascal_hypers=rascal_hypers,
        nu_target=2,
        lambda_filter=lambda_filter,
        sigma_filter=sigma_filter,
        lambda_cut=lambda_cut,
        selected_samples=selected_samples,
        species_neighbors=species_neighbors,
        use_sparse=True,
    )

    # Drop the redundant key name "order_nu". This is by definition 2 for all
    # lambda-SOAP blocks.
    lsoap = metatensor.remove_dimension(lsoap, axis="keys", name="order_nu")

    # If a single sigmas is requested, drop the now redundant "inversion_sigma"
    # key name
    if len(np.unique(lsoap.keys.column("inversion_sigma"))) == 1:
        lsoap = metatensor.remove_dimension(lsoap, axis="keys", name="inversion_sigma")

    return lsoap


def _parse_sigma_filter(
    nu_target: int,
    sigma_filter: Union[None, int, Sequence[int], Sequence[Sequence[int]]],
) -> Sequence[Sequence[int]]:
    """
    Returns parity filters for each CG combination step of a nu=1 tensor with
    itself up to the target body order.

    If a filter isn't specified by the user with `sigma_filter=None`, then no
    filter is applied, i.e. [-1, +1] is used at every iteration.

    If a single sequence of int is specified, then this is used for the last
    iteration only, and [-1, +1] is used for all intermediate iterations. For
    example, if `nu_target=4` and `sigma_filter=[+1]`, then the filter [[-1,
    +1], [-1, +1], [+1]] is returned.

    If a sequence of sequences of int is specified, then this is assumed to be
    the desired filter for each iteration and is only checked for validity
    without modification.
    """
    if nu_target < 2:
        raise ValueError("`nu_target` must be > 1")

    # No filter specified: use [-1, +1] for all iterations
    if sigma_filter is None:
        sigma_filter = [[-1, +1] for _ in range(nu_target - 1)]

    # Parse user-defined filter: assume passed as Sequence[int] or
    # Sequence[Sequence[int]]
    else:
        if isinstance(sigma_filter, int):
            sigma_filter = [sigma_filter]
        if not isinstance(sigma_filter, Sequence):
            raise TypeError(
                "`sigma_filter` must be an int, Sequence[int], or Sequence[Sequence[int]]"
            )
        # Single filter: apply on last iteration only, use both sigmas for
        # intermediate iterations
        if np.all([isinstance(sigma, int) for sigma in sigma_filter]):
            sigma_filter = [[-1, +1] for _ in range(nu_target - 2)] + [sigma_filter]

        else:
            # Assume filter explicitly defined for each iteration (checked below)
            pass

    # Check sigma_filter
    assert isinstance(sigma_filter, Sequence)
    assert len(sigma_filter) == nu_target - 1
    assert np.all([isinstance(filt, Sequence) for filt in sigma_filter])
    assert np.all([np.all([s in [-1, +1] for s in filt]) for filt in sigma_filter])

    return sigma_filter


def _parse_lambda_filter(
    nu_target: int,
    rascal_max_l: int,
    lambda_filter: Union[None, int, Sequence[int], Sequence[Sequence[int]]],
    lambda_cut: Union[None, int],
) -> Sequence[Sequence[int]]:
    """
    Returns parity filters for each CG combination step of a nu=1 tensor with
    itself up to the target body order.

    If a filter isn't specified by the user with `lambda_filter=None`, then no
    filter is applied. In this case all possible lambda channels are retained at
    each iteration. For example, if `nu_target=4`, `rascal_max_l=5`, and
    `lambda_cut=None`, then the returned filter is [[0, ..., 10], [0, ..., 15],
    [0, ..., 20]]. If `nu_target=4`, `rascal_max_l=5`, and `lambda_cut=10`, then
    the returned filter is [[0, ..., 10], [0, ..., 10], [0, ..., 10]].

    If `lambda_filter` is passed a single sequence of int, then this is used for
    the last iteration only, and all possible combinations of lambda are used in
    intermediate iterations. For instance, if `lambda_filter=[0, 1, 2]`, the
    returned filters for the 2 examples above, respectively, would be [[0, ...,
    10], [0, ..., 15], [0, 1, 2]] and [[0, ..., 10], [0, ..., 10], [0, 1, 2]].

    If a sequence of sequences of int is specified, then this is assumed to be
    the desired filter for each iteration and is only checked for validity
    without modification.
    """
    if nu_target < 2:
        raise ValueError("`nu_target` must be > 1")

    # Check value of lambda_cut
    if lambda_cut is not None:
        if not (rascal_max_l <= lambda_cut <= nu_target * rascal_max_l):
            raise ValueError(
                "`lambda_cut` must be >= `rascal_hypers['max_angular']` and <= `nu_target`"
                " * `rascal_hypers['max_angular']`"
            )

    # No filter specified: retain all possible lambda channels for every
    # iteration, up to lambda_cut (if specified)
    if lambda_filter is None:
        if lambda_cut is None:
            # Use the full range of possible lambda channels for each iteration.
            # This is dependent on the itermediate body order.
            lambda_filter = [
                [lam for lam in range(0, (nu * rascal_max_l) + 1)]
                for nu in range(2, nu_target + 1)
            ]
        else:
            # Use the full range of possible lambda channels for each iteration,
            # but only up to lambda_cut, independent of the intermediate body
            # order.
            lambda_filter = [
                [lam for lam in range(0, lambda_cut + 1)]
                for nu in range(2, nu_target + 1)
            ]

    # Parse user-defined filter: assume passed as Sequence[int] or
    # Sequence[Sequence[int]]
    else:
        if isinstance(lambda_filter, int):
            lambda_filter = [lambda_filter]
        if not isinstance(lambda_filter, Sequence):
            raise TypeError(
                "`lambda_filter` must be an int, Sequence[int], or Sequence[Sequence[int]]"
            )
        # Single filter: apply on last iteration only, use all possible lambdas for
        # intermediate iterations (up to lambda_cut, if specified)
        if np.all([isinstance(filt, int) for filt in lambda_filter]):
            if lambda_cut is None:
                # Use the full range of possible lambda channels for each iteration.
                # This is dependent on the itermediate body order.
                lambda_filter = [
                    [lam for lam in range(0, (nu * rascal_max_l) + 1)]
                    for nu in range(2, nu_target)
                ] + [lambda_filter]

            else:
                # Use the full range of possible lambda channels for each iteration,
                # but only up to lambda_cut, independent of the intermediate body
                # order.
                lambda_filter = [
                    [lam for lam in range(0, lambda_cut + 1)]
                    for nu in range(2, nu_target)
                ] + [lambda_filter]

        else:
            # Assume filter explicitly defined for each iteration (checked below)
            pass

    # Check lambda_filter
    if not isinstance(lambda_filter, Sequence):
        raise TypeError(
            "`lambda_filter` must be an int, Sequence[int], or Sequence[Sequence[int]]"
        )
    if len(lambda_filter) != nu_target - 1:
        raise ValueError(
            "`lambda_filter` must have length `nu_target` - 1, i.e. the number of CG"
            " iterations required to reach `nu_target`"
        )
    if not np.all([isinstance(filt, Sequence) for filt in lambda_filter]):
        raise TypeError(
            "`lambda_filter` must be an int, Sequence[int], or Sequence[Sequence[int]]"
        )
    # Check the lambda values are within the possible range, based on each
    # intermediate body order
    if not np.all(
        [
            np.all([0 <= lam <= nu * rascal_max_l for lam in filt])
            for nu, filt in enumerate(lambda_filter, start=2)
        ]
    ):
        raise ValueError(
            "All lambda values in `lambda_filter` must be >= 0 and <= `nu` *"
            " `rascal_hypers['max_angular']`, where `nu` is the body"
            " order created in the intermediate CG combination step"
        )
    # Now check that at each iteration the lambda values can actually be created
    # from combination at the previous iteration
    for filt_i, filt in enumerate(lambda_filter):
        if filt_i == 0:
            # Assume that the original nu=1 tensors to be combined have all l up
            # to and including `rascal_max_l`
            allowed_lams = np.arange(0, (2 * rascal_max_l) + 1)
        else:
            allowed_lams = []
            for l1, l2 in itertools.product(lambda_filter[filt_i - 1], repeat=2):
                for lam in range(abs(l1 - l2), abs(l1 + l2) + 1):
                    allowed_lams.append(lam)

            allowed_lams = np.unique(allowed_lams)

        if not np.all([lam in allowed_lams for lam in filt]):
            raise ValueError(
                f"invalid lambda values in `lambda_filter` for iteration {filt_i + 1}."
                f" {filt} cannot be created by combination of previous lambda values"
                f" {lambda_filter[filt_i - 1]}"
            )

    return lambda_filter


def n_body_iteration_single_center(
    frames: Sequence[ase.Atoms],
    rascal_hypers: dict,
    nu_target: int,
    lambda_filter: Optional[
        Union[None, int, Sequence[int], Sequence[Sequence[int]]]
    ] = None,
    sigma_filter: Optional[
        Union[None, int, Sequence[int], Sequence[Sequence[int]]]
    ] = None,
    lambda_cut: Optional[int] = None,
    selected_samples: Optional[Labels] = None,
    species_neighbors: Optional[Sequence[int]] = None,
    use_sparse: bool = True,
    debug: bool = False,
) -> TensorMap:
    """
    Based on the passed ``rascal_hypers``, generates a rascaline
    SphericalExpansion (i.e. nu = 1 body order descriptor) and combines it
    iteratively to generate a descriptor of order ``nu_target``.
    """
    # Generate a rascaline SphericalExpansion, for only the selected samples if
    # applicable
    calculator = rascaline.SphericalExpansion(**rascal_hypers)
    nu1_tensor = calculator.compute(frames, selected_samples=selected_samples)

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
    nu1_tensor = nu1_tensor.keys_to_properties(keys_to_move=keys_to_move)

    # Add "order_nu" and "inversion_sigma" key dimensions, both with values 1
    nu1_tensor = metatensor.insert_dimension(
        nu1_tensor, axis="keys", name="order_nu", values=np.array([1]), index=0
    )
    nu1_tensor = metatensor.insert_dimension(
        nu1_tensor, axis="keys", name="inversion_sigma", values=np.array([1]), index=1
    )

    # If the desired body order is 1, return the spherical expansion with
    # standardized metadata
    if nu_target == 1:
        return nu1_tensor

    # Otherwise, perform CG iterations. First, construct explicit sigma and
    # lambda filters for each iteration. Basic checks are performed here and
    # errors raised if invalid filters are passed.
    sigma_filter = _parse_sigma_filter(nu_target, sigma_filter)
    lambda_filter = _parse_lambda_filter(
        nu_target, rascal_hypers["max_angular"], lambda_filter, lambda_cut
    )
    if debug:
        print("sigma_filter: ", sigma_filter)
        print("lambda_filter: ", lambda_filter)

    # Create a copy of the nu = 1 tensor to combine with itself and store its
    # keys.
    nux_tensor = nu1_tensor.copy()
    nux_keys = nux_tensor.keys
    nu1_keys = nu1_tensor.keys

    # Pre-compute all the information needed to combined tensors at every
    # iteration. This includes the keys of the TensorMaps produced at each
    # iteration, the keys of the blocks combined to make them, and block
    # multiplicities.
    combine_info = []
    for iteration in range(1, nu_target):
        info = _create_combined_keys(
            nux_keys,
            nu1_keys,
            lambda_filter[iteration - 1],
            sigma_filter[iteration - 1],
        )
        combine_info.append(info)
        nux_keys = info[0]

    if debug:
        print("Num. keys at each step: ", [len(c[0]) for c in combine_info])
        print([nu1_keys] + [c[0] for c in combine_info])

    if np.any([len(c[0]) == 0 for c in combine_info]):
        raise ValueError(
            "invalid filters: one or more iterations produce no valid combinations."
            f" Number of keys at each iteration: {[len(c[0]) for c in combine_info]}."
            " Check the `lambda_filter` and `sigma_filter` arguments."
        )

    # Define the cached CG coefficients, either as sparse dicts or dense arrays
    lambda_max = max(
        rascal_hypers["max_angular"],
        np.max(np.concatenate(lambda_filter).flatten()),
    )
    # TODO: we know the lambda combinations in advance, so a more cleverly
    # constructed CG cache could be used to reduce memory overhead
    cg_cache = ClebschGordanReal(lambda_max, use_sparse)

    # Now combine block values until the target body order is reached
    for iteration in range(1, nu_target):
        if debug:
            print(f"CG iteration {iteration}")

        # Combine pairs of blocks into new TensorBlocks of the correct lambda.
        # Pass the correction factor accounting for the redundancy of "lx"
        # combinations.
        nux_keys = combine_info[iteration - 1][0]
        nux_blocks = []
        for nux_key, key_1, key_2, multi in zip(*combine_info[iteration - 1]):
            nux_blocks.append(
                _combine_single_center_block_pair(
                    nux_tensor[key_1],
                    nu1_tensor[key_2],
                    nux_key["spherical_harmonics_l"],
                    cg_cache,
                    correction_factor=np.sqrt(multi),
                )
            )

        nux_tensor = TensorMap(nux_keys, nux_blocks)

    # TODO: Account for body-order multiplicity and normalize block values
    # nux_tensor = _apply_body_order_corrections(nux_tensor)
    # nux_tensor = _normalize_blocks(nux_tensor)

    # Move the [l1, l2, ...] keys to the properties
    if nu_target > 1:
        nux_tensor = nux_tensor.keys_to_properties(
            [f"l{i}" for i in range(1, nu_target + 1)]
            + [f"k{i}" for i in range(2, nu_target)]
        )

    return nux_tensor


# def _combine_single_center(
#     tensor_1: TensorMap,
#     tensor_2: TensorMap,
#     lambdas: Sequence[int],
#     sigmas: Sequence[int],
#     cg_cache,
# ) -> TensorMap:
#     """
#     For 2 TensorMaps, ``tensor_1`` and ``tensor_2``, with body orders nu and 1
#     respectively, combines their blocks to form a new TensorMap with body order
#     (nu + 1).

#     Returns blocks only indexed by keys .

#     Assumes the metadata of the two TensorMaps are standardized as follows.

#     The keys of `tensor_1`  must follow the key name convention:

#     ["order_nu", "inversion_sigma", "spherical_harmonics_l", "species_center",
#     "l1", "l2", ..., f"l{`nu`}", "k2", ..., f"k{`nu`-1}"]. The "lx" columns
#     track the l values of the nu=1 blocks that were previously combined. The
#     "kx" columns tracks the intermediate lambda values of nu > 1 blocks that
#     haev been combined.

#     For instance, a TensorMap of body order nu=4 will have key names
#     ["order_nu", "inversion_sigma", "spherical_harmonics_l", "species_center",
#     "l1", "l2", "l3", "l4", "k2", "k3"]. Two nu=1 TensorMaps with blocks of
#     order "l1" and "l2" were combined to form a nu=2 TensorMap with blocks of
#     order "k2". This was combined with a nu=1 TensorMap with blocks of order
#     "l3" to form a nu=3 TensorMap with blocks of order "k3". Finally, this was
#     combined with a nu=1 TensorMap with blocks of order "l4" to form a nu=4.

#     .. math ::

#         \bra{ n_1 l_1 ; n_2 l_2 k_2 ; ... ; n{\nu-1} l_{\nu-1} k_{\nu-1} ;
#         n{\nu} l_{\nu} k_{\nu}; \lambda } \ket{ \rho^{\otimes \nu}; \lambda M }

#     The keys of `tensor_2` must follow the key name convention:

#     ["order_nu", "inversion_sigma", "spherical_harmonics_l", "species_center"]

#     Samples of pairs of blocks corresponding to the same chemical species are
#     equivalent in the two TensorMaps. Samples names are ["structure", "center"]

#     Components names are [["spherical_harmonics_m"],] for each block.

#     Property names are ["n1", "n2", ..., "species_neighbor_1",
#     "species_neighbor_2", ...] for each block.
#     """

#     # Get the correct keys for the combined output TensorMap
#     (
#         nux_keys,
#         keys_1_entries,
#         keys_2_entries,
#         multiplicity_list,
#     ) = _create_combined_keys(tensor_1.keys, tensor_2.keys, lambdas, sigmas)

#     # Iterate over pairs of blocks and combine
#     nux_blocks = []
#     for nux_key, key_1, key_2, multi in zip(
#         nux_keys, keys_1_entries, keys_2_entries, multiplicity_list
#     ):
#         # Retrieve the blocks
#         block_1 = tensor_1[key_1]
#         block_2 = tensor_2[key_2]

#         # Combine the blocks into a new TensorBlock of the correct lambda order.
#         # Pass the correction factor accounting for the redundancy of "lx"
#         # combinations.
#         nux_blocks.append(
#             _combine_single_center_block_pair(
#                 block_1,
#                 block_2,
#                 nux_key["spherical_harmonics_l"],
#                 cg_cache,
#                 correction_factor=np.sqrt(multi),
#             )
#         )

#     return TensorMap(nux_keys, nux_blocks)


def _combine_single_center_block_pair(
    block_1: TensorBlock,
    block_2: TensorBlock,
    lam: int,
    cg_cache,
    correction_factor: float = 1.0,
) -> TensorBlock:
    """
    For a given pair of TensorBlocks and desired lambda value, combines the
    values arrays and returns in a new TensorBlock.
    """

    # Do the CG combination - single center so no shape pre-processing required
    combined_values = _clebsch_gordan_combine(
        block_1.values, block_2.values, lam, cg_cache
    )

    # Infer the new nu value: block 1's properties are nu pairs of
    # "species_neighbor_x" and "nx".
    combined_nu = int((len(block_1.properties.names) / 2) + 1)

    # Define the new property names for "nx" and "species_neighbor_x"
    n_names = [f"n{i}" for i in range(1, combined_nu + 1)]
    neighbor_names = [f"species_neighbor_{i}" for i in range(1, combined_nu + 1)]
    prop_names = [item for i in zip(neighbor_names, n_names) for item in i]

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
        properties=Labels(
            names=prop_names,
            values=np.array(
                [
                    np.concatenate((b2, b1))
                    for b2 in block_2.properties.values
                    for b1 in block_1.properties.values
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
    if cg_cache.sparse:
        return _clebsch_gordan_combine_sparse(arr_1, arr_2, lam, cg_cache)
    return _clebsch_gordan_combine_dense(arr_1, arr_2, lam, cg_cache)


def _clebsch_gordan_combine_sparse(
    arr_1: np.ndarray,
    arr_2: np.ndarray,
    lam: int,
    cg_cache,
) -> np.ndarray:
    """
    TODO: finish docstring.

    Performs a Clebsch-Gordan combination step on 2 arrays using sparse
    operations.

    :param arr_1: array with the m values for l1 with shape [n_samples, 2 * l1 +
        1, n_q_properties]
    :param arr_2: array with the m values for l2 with shape [n_samples, 2 * l2 +
        1, n_p_properties]
    :param lam: int value of the resulting coupled channel
    :param cg_cache: sparse dictionary with keys (m1, m2, mu) and array values
        being sparse blocks of shape <TODO: fill out>

    :returns: array of shape [n_samples, (2*lam+1), q_properties * p_properties]
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
    for m1, m2, mu in cg_coeffs.keys():
        # Broadcast arrays, multiply together and with CG coeff
        arr_out[:, mu, :] += (
            arr_1[:, m1, :, None] * arr_2[:, m2, None, :] * cg_coeffs[(m1, m2, mu)]
        ).reshape(n_i, n_p * n_q)

    return arr_out


def _clebsch_gordan_combine_dense(
    arr_1: np.ndarray,
    arr_2: np.ndarray,
    lam: int,
    cg_cache,
) -> np.ndarray:
    """
    Performs a Clebsch-Gordan combination step on 2 arrays using a dense
    operation.

    :param arr_1: array with the m values for l1 with shape [n_samples, 2 * l1 +
        1, n_q_properties]
    :param arr_2: array with the m values for l2 with shape [n_samples, 2 * l2 +
        1, n_p_properties]
    :param lam: int value of the resulting coupled channel
    :param cg_cache: dense array of shape [(2 * l1 +1) * (2 * l2 +1), (2 * lam +
        1)]

    :returns: array of shape [n_samples, (2*lam+1), q_properties * p_properties]

    TODO: do we need this example here? Could it be moved to a test?

    >>> N_SAMPLES = 30
    >>> N_Q_PROPERTIES = 10
    >>> N_P_PROPERTIES = 8
    >>> L1 = 2
    >>> L2 = 3
    >>> LAM = 2
    >>> arr_1 = np.random.rand(N_SAMPLES, 2*L1+1, N_Q_PROPERTIES)
    >>> arr_2 = np.random.rand(N_SAMPLES, 2*L2+1, N_P_PROPERTIES)
    >>> cg_cache = {(L1, L2, LAM): np.random.rand(2*L1+1, 2*L2+1, 2*LAM+1)}
    >>> out1 = _clebsch_gordan_dense(arr_1, arr_2, LAM, cg_cache)
    >>> # (samples l1_m  q_features) (samples l2_m p_features),
    >>> #   (l1_m  l2_m  lambda_mu)
    >>> # --> (samples, lambda_mu q_features p_features)
    >>> # in einsum l1_m is l, l2_m is k, lambda_mu is L
    >>> out2 = np.einsum("slq, skp, lkL -> sLqp", arr_1, arr_2, cg_cache[(L1, L2, LAM)])
    >>> # --> (samples lambda_mu (q_features p_features))
    >>> out2 = out2.reshape(arr_1.shape[0], 2*LAM+1, -1)
    >>> print(np.allclose(out1, out2))
    True
    """
    # Infer l1 and l2 from the len of the length of axis 1 of each tensor
    l1 = (arr_1.shape[1] - 1) // 2
    l2 = (arr_2.shape[1] - 1) // 2
    cg_coeffs = cg_cache.coeffs[(l1, l2, lam)]

    # (samples None None l1_mu q) * (samples l2_mu p None None) -> (samples l2_mu p l1_mu q)
    # we broadcast it in this way so we only need to do one swapaxes in the next step
    arr_out = arr_1[:, None, None, :, :] * arr_2[:, :, :, None, None]

    # (samples l2_mu p l1_mu q) -> (samples q p l1_mu l2_mu)
    arr_out = arr_out.swapaxes(1, 4)

    # samples (q p l1_mu l2_mu) -> (samples (q p) (l1_mu l2_mu))
    arr_out = arr_out.reshape(
        -1, arr_1.shape[2] * arr_2.shape[2], arr_1.shape[1] * arr_2.shape[1]
    )

    # (l1_mu l2_mu lam_mu) -> ((l1_mu l2_mu) lam_mu)
    cg_coeffs = cg_coeffs.reshape(-1, 2 * lam + 1)

    # (samples (q p) (l1_mu l2_mu)) @ ((l1_mu l2_mu) lam_mu) -> samples (q p) lam_mu
    arr_out = arr_out @ cg_coeffs

    # (samples (q p) lam_mu) -> (samples lam_mu (q p))
    return arr_out.swapaxes(1, 2)


def _apply_body_order_corrections(tensor: TensorMap) -> TensorMap:
    """
    Applies the appropriate prefactors to the block values of the output
    TensorMap (i.e. post-CG combination) according to its body order.
    """
    return tensor


def _normalize_blocks(tensor: TensorMap) -> TensorMap:
    """
    Applies corrections to the block values based on their 'leaf' l-values, such
    that the norm is preserved.
    """
    return tensor


# ===== Fxns to manipulate metadata of TensorMaps =====


def _create_combined_keys(
    keys_1: Labels,
    keys_2: Labels,
    lambdas: Sequence[int],
    sigmas: Sequence[int],
) -> Tuple[Labels, Sequence[Sequence[int]]]:
    """
    Given the keys of 2 TensorMaps and a list of desired lambda values, creates
    the correct keys for the TensorMap returned after one CG combination step.

    Assumes that `keys_1` corresponds to a TensorMap with arbitrary body order,
    while `keys_2` corresponds to a TensorMap with body order 1.

    `keys_1`  must follow the key name convention:

    ["order_nu", "inversion_sigma", "spherical_harmonics_l", "species_center",
    "l1", "l2", ..., f"l{`nu`}", "k2", ..., f"k{`nu`-1}"]. The "lx" columns
    track the l values of the nu=1 blocks that were previously combined. The
    "kx" columns tracks the intermediate lambda values of nu > 1 blocks that
    have been combined.

    For instance, a TensorMap of body order nu=4 will have key names
    ["order_nu", "inversion_sigma", "spherical_harmonics_l", "species_center",
    "l1", "l2", "l3", "l4", "k2", "k3"]. Two nu=1 TensorMaps with blocks of
    order "l1" and "l2" were combined to form a nu=2 TensorMap with blocks of
    order "k2". This was combined with a nu=1 TensorMap with blocks of order
    "l3" to form a nu=3 TensorMap with blocks of order "k3". Finally, this was
    combined with a nu=1 TensorMap with blocks of order "l4" to form a nu=4.

    .. math ::

        \bra{ n_1 l_1 ; n_2 l_2 k_2 ; ... ; n_{\nu-1} l_{\nu-1} k_{\nu-1} ;
        n_{\nu} l_{\nu} k_{\nu}; \lambda } \ket{ \rho^{\otimes \nu}; \lambda M }

    `keys_2` must follow the key name convention:

    ["order_nu", "inversion_sigma", "spherical_harmonics_l", "species_center"]

    Returned is a tuple.

    The first element is to the Labels object for the keys of the output
    TensorMap created by a CG combination step.

    The second element is a list of list of ints. Each sublist corresponds to
    [lam1, lam2, correction_factor terms]. lam1 and lam2 tracks the lambda
    values of the blocks that combine to form the block indexed by the
    corresponding key. The correction_factor terms are the prefactors that
    account for the redundancy in the CG combination.

    The `sigmas` argument can be used to return only keys with certain
    sigmas. This must be passed as a list with elements +1 and/or -1.
    """
    # Get the body order of the first TensorMap.
    nu1 = np.unique(keys_1.column("order_nu"))[0]

    # Define nu value of output TensorMap
    nu = nu1 + 1

    # Check the body order of the first TensorMap.
    assert np.all(keys_1.column("order_nu") == nu1)

    # The second by convention should be nu = 1.
    assert np.all(keys_2.column("order_nu") == 1)

    # If nu = 1, the key names don't yet have any "lx" columns
    if nu1 == 1:
        l_list_names = []
        new_l_list_names = ["l1", "l2"]
    else:
        l_list_names = [f"l{l}" for l in range(1, nu1 + 1)]
        new_l_list_names = l_list_names + [f"l{nu}"]

    # Check key names
    assert np.all(
        keys_1.names
        == ["order_nu", "inversion_sigma", "spherical_harmonics_l", "species_center"]
        + l_list_names
        + [f"k{k}" for k in range(2, nu1)]
    )
    assert np.all(
        keys_2.names
        == ["order_nu", "inversion_sigma", "spherical_harmonics_l", "species_center"]
    )

    # Check `sigmas` argument
    assert isinstance(sigmas, Sequence)
    assert np.all([s in [-1, +1] for s in sigmas])

    # Define key names of output Labels (i.e. for combined TensorMap)
    new_names = (
        ["order_nu", "inversion_sigma", "spherical_harmonics_l", "species_center"]
        + new_l_list_names
        + [f"k{k}" for k in range(2, nu)]
    )

    new_key_values = []
    keys_1_entries = []
    keys_2_entries = []
    for key_1, key_2 in itertools.product(keys_1, keys_2):
        # Unpack relevant key values
        sig1, lam1, a = key_1.values[1:4]
        sig2, lam2, a2 = key_2.values[1:4]

        # Only combine blocks of the same chemical species
        if a != a2:
            continue

        # Only combine to create blocks of desired lambda values
        nonzero_lams = np.arange(abs(lam1 - lam2), abs(lam1 + lam2) + 1)
        for lam in nonzero_lams:
            if lam not in lambdas:
                continue

            # Calculate new sigma
            sig = sig1 * sig2 * (-1) ** (lam1 + lam2 + lam)

            # Skip keys that don't give the desired sigmas
            if sig not in sigmas:
                continue

            # Extract the l and k lists from keys_1
            l_list = key_1.values[4 : 4 + nu1].tolist()
            k_list = key_1.values[4 + nu1 :].tolist()

            # Build the new keys values. l{nu} is `lam2`` (i.e.
            # "spherical_harmonics_l" of the key from `keys_2`. k{nu-1} is
            # `lam1` (i.e. "spherical_harmonics_l" of the key from `keys_1`).
            new_vals = [nu, sig, lam, a] + l_list + [lam2] + k_list + [lam1]
            new_key_values.append(new_vals)
            keys_1_entries.append(key_1)
            keys_2_entries.append(key_2)

    # Define new keys as the full product of keys_1 and keys_2
    nux_keys = Labels(names=new_names, values=np.array(new_key_values))

    # Now account for multiplicty
    key_idxs_to_keep = []
    mult_dict = {}
    for key_idx, key in enumerate(nux_keys):
        # Get the important key values. This is all of the keys, excpet the k
        # list
        key_vals_slice = key.values[: 4 + (nu + 1)].tolist()
        first_part, l_list = key_vals_slice[:4], key_vals_slice[4:]

        # Sort the l list
        l_list_sorted = sorted(l_list)

        # Compare the sliced key with the one recreated when the l list is
        # sorted. If they are identical, this is the key of the block that we
        # want to compute a CG combination for.
        key_slice_tuple = tuple(first_part + l_list)
        key_slice_sorted_tuple = tuple(first_part + l_list_sorted)
        if np.all(key_slice_tuple == key_slice_sorted_tuple):
            key_idxs_to_keep.append(key_idx)

        # Now count the multiplicity of each sorted l_list
        if mult_dict.get(key_slice_sorted_tuple) is None:
            mult_dict[key_slice_sorted_tuple] = 1
        else:
            mult_dict[key_slice_sorted_tuple] += 1

    # Build a reduced Labels object for the combined keys, with redundancies removed
    combined_keys_red = Labels(
        names=new_names,
        values=np.array([nux_keys[idx].values for idx in key_idxs_to_keep]),
    )

    # Create a of LabelsEntry objects that correspond to the original keys in
    # `keys_1` and `keys_2` that combined to form the combined key
    keys_1_entries_red = [keys_1_entries[idx] for idx in key_idxs_to_keep]
    keys_2_entries_red = [keys_2_entries[idx] for idx in key_idxs_to_keep]

    # Define the multiplicity of each key
    mult_list = [
        mult_dict[tuple(nux_keys[idx].values[: 4 + (nu + 1)].tolist())]
        for idx in key_idxs_to_keep
    ]

    return combined_keys_red, keys_1_entries_red, keys_2_entries_red, mult_list
