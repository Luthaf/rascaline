"""
Module that stores the ClebschGordanReal class for computing and caching Clebsch
Gordan coefficients for use in CG combinations.
"""

import math
from typing import Dict, List

import numpy as np
import wigners

from .. import _dispatch
from .._backend import Array, Labels, TensorBlock, TensorMap


try:
    import torch
    from torch import Tensor as TorchTensor

    torch_dtype = torch.dtype
    torch_device = torch.device

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

    class TorchTensor:
        pass

    class torch_dtype:
        pass

    class torch_device:
        pass


UNKNOWN_ARRAY_TYPE = (
    "unknown array type, only numpy arrays and torch tensors are supported"
)


def calculate_cg_coefficients(
    lambda_max: int,
    sparse: bool = True,
    use_torch: bool = False,
) -> TensorMap:
    """
    Calculates the Clebsch-Gordan coefficients for all possible combination of l1 and
    l2, up to ``lambda_max``. Returns them in :py:class:`TensorMap` format.

    The output data structure of the output :py:class:`TensorMap` depends on whether the
    backend used to perform CG tensor products uses sparse or dense operations.

    Dense:
        - samples: `_`, i.e. a dummy sample.
        - components: `[(m1,), (m2,), (mu,)]`, i.e. on separate components axes,
          where `m1` and `m2` are the m component values for the two arrays
          being combined and `mu` is the m component value for the resulting
          array.
        - properties: `cg_coefficient`

    Sparse:
        - samples: `(m1, m2, mu)`, where `m1` and `m2` are the m component
          values for the two arrays being combined and `mu` is the m component
          value for the resulting array.
        - components: `[]`, i.e. no components axis.
        - properties: `cg_coefficient`


    :param lambda_max: maximum angular momentum value to compute CG coefficients for.
    :param sparse: whether to store the CG coefficients in sparse format.
    :param use_torch: whether torch tensor or numpy arrays should be used for the cg
        coeffs

    :returns: :py:class:`TensorMap` of the Clebsch-Gordan coefficients.
    """
    # Build some 'like' arrays for the backend dispatch
    if use_torch:
        complex_like = torch.empty(0, dtype=torch.complex128)
        double_like = torch.empty(0, dtype=torch.double)
        if isinstance(Labels, torch.ScriptClass):
            labels_values_like = torch.empty(0, dtype=torch.double)
        else:
            labels_values_like = np.empty(0, dtype=np.double)
    else:
        complex_like = np.empty(0, dtype=np.complex128)
        double_like = np.empty(0, dtype=np.double)
        labels_values_like = np.empty(0, dtype=np.double)

    # Calculate the CG coefficients, stored as a dict of dense arrays. This is the
    # starting point for the conversion to a TensorMap of different formats depending on
    # the backend options.
    cg_coeff_dict = _build_dense_cg_coeff_dict(
        lambda_max, complex_like, double_like, labels_values_like
    )

    # Build the CG cache depending on whether the CG backend is sparse or dense. The
    # dispatching of the arrays backends are accounted for by `double_like` and
    # `labels_values_like`.
    if sparse:
        return _cg_coeff_dict_to_tensormap_sparse(
            cg_coeff_dict, double_like, labels_values_like
        )

    return _cg_coeff_dict_to_tensormap_dense(
        cg_coeff_dict, double_like, labels_values_like
    )


def _build_dense_cg_coeff_dict(
    lambda_max: int, complex_like: Array, double_like: Array, labels_values_like: Array
) -> Dict[int, Array]:
    """
    Calculates the CG coefficients and stores them as dense arrays in a dictionary.

    Each dictionary entry is a dense array with shape (2 * l1 + 1, 2 * l2 + 1, 2 *
    lambda + 1).

    {
        (l1, l2, lambda):
            array(
                cg_{m1, m2, mu}^{l1, l2, lambda}
                ...
                for m1 in range(-l1, l1 + 1),
                for m2 in range(-l2, l2 + 1),
                for mu in range(-lambda, lambda + 1),

                shape=(2 * l1 + 1, 2 * l2 + 1, 2 * lambda + 1),
            )
        ...
        for l1 in range(0, l1_list)
        for l2 in range(0, l2_list)
        for lambda in range(0, range(|l1 - l2|, ..., |l1 + l2|))
    }

    where `cg_{m1, m2, mu}^{l1, l2, lambda}` is the Clebsch-Gordan coefficient that
    describes the combination of the `m1` irreducible component of the `l1` angular
    channel and the `m2` irreducible component of the `l2` angular channel into the
    irreducible tensor of order `lambda`. In all cases, these correspond to the non-zero
    CG coefficients, i.e. those in the range |-l, ..., +l| for each angular order l in
    {l1, l2, lambda}.

    :param lambda_max: maximum angular momentum  value to compute CG coefficients for.
    :param complex_like: an empty array of dtype complex, used for dispatching
        operations
    :param double_like: an empty array of dtype double, used for dispatching operations
    :param labels_values_like: an empty array of dtype int32, used for dispatching
        operations

    :returns: dictionary of dense CG coefficients.
    """
    # real-to-complex and complex-to-real transformations as matrices
    r2c: Dict[int, Array] = {}
    c2r: Dict[int, Array] = {}
    coeff_dict = {}

    for o3_lambda in range(0, lambda_max + 1):
        c2r[o3_lambda] = _complex2real(o3_lambda, like=complex_like)
        r2c[o3_lambda] = _real2complex(o3_lambda, like=complex_like)

    for l1 in range(lambda_max + 1):
        for l2 in range(lambda_max + 1):
            for o3_lambda in range(
                max(l1, l2) - min(l1, l2), min(lambda_max, (l1 + l2)) + 1
            ):
                complex_cg = _complex_clebsch_gordan_matrix(
                    l1, l2, o3_lambda, complex_like
                )

                real_cg = (r2c[l1].T @ complex_cg.reshape(2 * l1 + 1, -1)).reshape(
                    complex_cg.shape
                )

                real_cg = real_cg.swapaxes(0, 1)
                real_cg = (r2c[l2].T @ real_cg.reshape(2 * l2 + 1, -1)).reshape(
                    real_cg.shape
                )
                real_cg = real_cg.swapaxes(0, 1)

                real_cg = real_cg @ c2r[o3_lambda].T

                if (l1 + l2 + o3_lambda) % 2 == 0:
                    cg_l1l2lam_dense = _dispatch.real(real_cg)
                else:
                    cg_l1l2lam_dense = _dispatch.imag(real_cg)

                coeff_dict[(l1, l2, o3_lambda)] = cg_l1l2lam_dense

    return coeff_dict


def _cg_coeff_dict_to_tensormap_dense(
    coeff_dict: Dict, double_like: Array, labels_values_like: Array
) -> TensorMap:
    """
    Converts the dictionary of dense CG coefficients coefficients to
    :py:class:`TensorMap` format, specifically for performing CG tensor products with
    the "python-dense" backend.
    """
    keys = Labels(
        ["l1", "l2", "lambda"],
        _dispatch.int_array_like(list(coeff_dict.keys()), labels_values_like),
    )
    blocks = []

    for l1l2lam_values in coeff_dict.values():
        # extending shape by samples and properties
        block_value_shape = (1,) + l1l2lam_values.shape + (1,)
        blocks.append(
            TensorBlock(
                values=_dispatch.contiguous(l1l2lam_values.reshape(block_value_shape)),
                samples=Labels.range("_", 1),
                components=[
                    Labels(
                        ["m1"],
                        _dispatch.int_range_like(
                            0, l1l2lam_values.shape[0], labels_values_like
                        ).reshape(-1, 1),
                    ),
                    Labels(
                        ["m2"],
                        _dispatch.int_range_like(
                            0, l1l2lam_values.shape[1], labels_values_like
                        ).reshape(-1, 1),
                    ),
                    Labels(
                        ["mu"],
                        _dispatch.int_range_like(
                            0, l1l2lam_values.shape[2], labels_values_like
                        ).reshape(-1, 1),
                    ),
                ],
                properties=Labels.range("cg_coefficient", 1),
            )
        )

    return TensorMap(keys, blocks)


def _cg_coeff_dict_to_tensormap_sparse(
    coeff_dict: Dict, double_like: Array, labels_values_like: Array
) -> TensorMap:
    """
    Converts the dictionary of dense CG coefficients coefficients to
    :py:class:`TensorMap` format, specifically for performing CG tensor products with
    the "python-sparse" backend.
    """
    dict_keys = list(coeff_dict.keys())
    keys = Labels(
        ["l1", "l2", "lambda"],
        _dispatch.int_array_like(list(dict_keys), labels_values_like),
    )
    blocks = []

    # For each (l1, l2, lambda) combination, build a TensorBlock of non-zero CG coeffs
    for l1, l2, o3_lambda in dict_keys:
        cg_l1l2lam_dense = coeff_dict[(l1, l2, o3_lambda)]

        # Find the dense indices of the non-zero CG coeffs
        nonzeros_cg_coeffs_idx = _dispatch.where(
            _dispatch.abs(cg_l1l2lam_dense) > 1e-15
        )

        # Create a sparse dictionary indexed by  of the non-zero CG coeffs
        cg_l1l2lam_sparse = {}
        for i in range(len(nonzeros_cg_coeffs_idx[0])):
            m1 = nonzeros_cg_coeffs_idx[0][i]
            m2 = nonzeros_cg_coeffs_idx[1][i]
            mu = nonzeros_cg_coeffs_idx[2][i]
            cg_l1l2lam_sparse[(m1, m2, mu)] = cg_l1l2lam_dense[m1, m2, mu]

        l1l2lam_sample_values = []
        for m1m2mu_key in cg_l1l2lam_sparse.keys():
            l1l2lam_sample_values.append(m1m2mu_key)
        # extending shape by samples and properties
        values = _dispatch.double_array_like(
            [*cg_l1l2lam_sparse.values()], double_like
        ).reshape(-1, 1)
        l1l2lam_sample_values = _dispatch.int_array_like(
            l1l2lam_sample_values, labels_values_like
        )
        # we have to move put the m1 m2 m3 inside a block so we can access it easier
        # inside cg combine function,
        blocks.append(
            TensorBlock(
                values=_dispatch.contiguous(values),
                samples=Labels(["m1", "m2", "mu"], l1l2lam_sample_values),
                components=[],
                properties=Labels.range("cg_coefficient", 1),
            )
        )

    return TensorMap(keys, blocks)


# ============================
# ===== Helper functions
# ============================


def _real2complex(o3_lambda: int, like: Array) -> Array:
    """
    Computes a matrix that can be used to convert from real to complex-valued spherical
    harmonics(coefficients) of order ``o3_lambda``.

    This is meant to be applied to the left:
    ``real2complex @ [-o3_lambda, ..., +o3_lambda]``.

    See https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form for details on the
    convention for how these tranformations are defined.

    Operations are dispatched to the corresponding array type given by ``like``
    """
    result = _dispatch.zeros_like(like, shape=(2 * o3_lambda + 1, 2 * o3_lambda + 1))
    inv_sqrt_2 = 1.0 / math.sqrt(2.0)
    i_sqrt_2 = 1.0j / complex(math.sqrt(2.0))

    for m in range(-o3_lambda, o3_lambda + 1):
        if m < 0:
            # Positve part
            result[o3_lambda + m, o3_lambda + m] = i_sqrt_2
            # Negative part
            result[o3_lambda - m, o3_lambda + m] = -i_sqrt_2 * ((-1) ** m)

        if m == 0:
            result[o3_lambda, o3_lambda] = 1.0

        if m > 0:
            # Negative part
            result[o3_lambda - m, o3_lambda + m] = inv_sqrt_2
            # Positive part
            result[o3_lambda + m, o3_lambda + m] = inv_sqrt_2 * ((-1) ** m)

    return result


def _complex2real(o3_lambda: int, like) -> Array:
    """
    Converts from complex to real spherical harmonics. This is just given by the
    conjugate tranpose of the real->complex transformation matrices.

    Operations are dispatched to the corresponding array type given by ``like``
    """
    return _dispatch.conjugate(_real2complex(o3_lambda, like)).T


def _complex_clebsch_gordan_matrix(l1: int, l2: int, o3_lambda: int, like: Array):
    r"""clebsch-gordan matrix
    Computes the Clebsch-Gordan (CG) matrix for
    transforming complex-valued spherical harmonics.
    The CG matrix is computed as a 3D array of elements
        < l1 m1 l2 m2 | o3_lambda mu >
    where the first axis loops over m1, the second loops over m2,
    and the third one loops over mu. The matrix is real.
    For example, using the relation:
        | l1 l2 o3_lambda mu > =
            \sum_{m1, m2}
            <l1 m1 l2 m2 | o3_lambda mu > | l1 m1 > | l2 m2 >
    (https://en.wikipedia.org/wiki/Clebsch–Gordan_coefficients, section
    "Formal definition of Clebsch-Gordan coefficients", eq 2)
    one can obtain the spherical harmonics o3_lambda from two sets of
    spherical harmonics with l1 and l2 (up to a normalization factor).
    E.g.:
    Args:
        l1: l number for the first set of spherical harmonics
        l2: l number for the second set of spherical harmonics
        o3_lambda: l number For the third set of spherical harmonics
        like: Operations are dispatched to the corresponding this arguments array type
    Returns:
        cg: CG matrix for transforming complex-valued spherical harmonics
    >>> from scipy.special import sph_harm
    >>> import numpy as np
    >>> import wigners
    >>> C_112 = _complex_clebsch_gordan_matrix(1, 1, 2, np.empty(0))
    >>> comp_sph_1 = np.array([sph_harm(m, 1, 0.2, 0.2) for m in range(-1, 1 + 1)])
    >>> comp_sph_2 = np.array([sph_harm(m, 1, 0.2, 0.2) for m in range(-1, 1 + 1)])
    >>> # obtain the (unnormalized) spherical harmonics
    >>> # with l = 2 by contraction over m1 and m2
    >>> comp_sph_2_u = np.einsum("ijk,i,j->k", C_112, comp_sph_1, comp_sph_2)
    >>> # we can check that they differ from the spherical harmonics
    >>> # by a constant factor
    >>> comp_sph_2 = np.array([sph_harm(m, 2, 0.2, 0.2) for m in range(-2, 2 + 1)])
    >>> ratio = comp_sph_2 / comp_sph_2_u
    >>> np.allclose(ratio[0], ratio)
    True
    """
    if abs(l1 - l2) > o3_lambda or abs(l1 + l2) < o3_lambda:
        return _dispatch.zeros_like(like, (2 * l1 + 1, 2 * l2 + 1, 2 * o3_lambda + 1))
    else:
        return wigners.clebsch_gordan_array(l1, l2, o3_lambda)


# ================================================================ #
# =============== Functions for performing CG coupling =========== #
# ================================================================ #


def _cg_couple_sparse(
    arrays: Dict[str, Array],
    l1: int,
    l2: int,
    o3_lambda: int,
    cg_coefficients: TensorMap,
):
    """
    Couple two spherical harmonics (of degree ``l1`` and ``l2``) to a single one (of
    degree ``o3_lambda``) using CG coefficients. This is a "dense" implementation, using
    all CG coefficients at the same time.

    :param l1: degree of the first spherical harmonic
    :param l2: degree of the first spherical harmonic
    :param arrays: dictionary of arrays for the different ``(m1, m2)`` corresponding to
        ``l1`` and ``l2``. Each array in the dictionary is expected to have a shape
        ``[samples, properties]`` and correspond to a single ``(m1, m2)`` pair.
    :param o3_lambda: value of lambda for the output spherical harmonic
    :param cg_coefficients: CG coefficients as returned by
        :py:func:`calculate_cg_coefficients` with ``sparse=True``
    """

    array = list(arrays.values())[0]
    assert len(array.shape) == 2

    output = _dispatch.zeros_like(
        array, (array.shape[0], 2 * o3_lambda + 1, array.shape[1])
    )

    # Fill in each mu component of the output array in turn
    cg_l1l2lam = cg_coefficients.block({"l1": l1, "l2": l2, "lambda": o3_lambda})
    for i in range(len(cg_l1l2lam.samples)):
        m1m2mu = cg_l1l2lam.samples.entry(i)
        m1 = m1m2mu[0]
        m2 = m1m2mu[1]
        mu = m1m2mu[2]
        # Broadcast arrays, multiply together and with CG coeff
        output[:, mu, :] += arrays[str((m1, m2))] * cg_l1l2lam.values[i, 0]

    return output


def _cg_couple_dense(
    array: Array,
    o3_lambda: int,
    cg_coefficients: TensorMap,
) -> Array:
    """
    Couple two spherical harmonics (of degree ``l1`` and ``l2``) to a single one (of
    degree ``o3_lambda``) using CG coefficients. This is a "dense" implementation, using
    all CG coefficients at the same time.

    :param array: input array, we expect a shape of ``[samples, 2*l1 + 1, 2*l2 + 1]``
    :param o3_lambda: value of lambda for the output spherical harmonic
    :param cg_coefficients: CG coefficients as returned by
        :py:func:`calculate_cg_coefficients` with ``sparse=False``
    """
    assert len(array.shape) == 3

    l1 = (array.shape[1] - 1) // 2
    l2 = (array.shape[2] - 1) // 2

    cg_l1l2lam = cg_coefficients.block({"l1": l1, "l2": l2, "lambda": o3_lambda}).values

    # [samples, l1, l2] => [samples, (l1 l2)]
    array = array.reshape(-1, (2 * l1 + 1) * (2 * l2 + 1))

    # [l1, l2, lambda] -> [(l1 l2), lambda]
    cg_l1l2lam = cg_l1l2lam.reshape(-1, 2 * o3_lambda + 1)

    # [samples, (l1 l2)] @ [(l1 l2), lambda] => [samples, lambda]
    return array @ cg_l1l2lam


# ======================================================================= #
# =============== Functions for performing CG tensor products =========== #
# ======================================================================= #


def cg_tensor_product(
    array_1: Array,
    array_2: Array,
    o3_lambdas: List[int],
    cg_coefficients: TensorMap,
    cg_backend: str,
) -> List[Array]:
    """
    Compute the Clebsch-Gordan tensor product of ``array_1`` and ``array_2``.

    ``array_1`` shape should be ``(n_samples, 2 * l1 + 1, n_q)`` and ``array_2`` shape
    should be ``(n_samples, 2 * l2 + 1, n_p)``. ``n_samples`` is the number of samples,
    ``n_q`` and ``n_p`` are the number of properties in each array. The number of
    samples in both array must be the same.

    This function will output a list of arrays, whose shape will be ``[n_samples, (2 *
    o3_lambda+1), n_q * n_p]``, with the specified ``o3_lambdas``. These arrays will
    contain the result of computing a tensor product of ``array_1`` and ``array_2``,
    followed by a projection from the product of spherical harmonic with degree ``l1``
    and ``l2`` to a single spherical harmonic of degree ``o3_lambdas``; using
    Clebsch-Gordan coefficients. The values for ``l1`` and ``l2`` are inferred from the
    size of the middle axis of the input arrays.

    The Clebsch-Gordan coefficients are cached in ``cg_coefficients``. These must be
    computed by :py:func:`calculate_cg_coefficients`.

    The operation is dispatched such that numpy arrays or torch tensors are
    automatically handled.

    ``cg_backend="metadata"`` can be used to return an empty array of the correct shape,
    without performing any calculation. This can be useful for probing the outputs of CG
    iterations in terms of metadata without the computational cost of performing the CG
    combinations — i.e. using :py:func:`DensityCorrelations.compute_metadata`.

    :param array_1: array with shape ``[n_samples, 2 * l1 + 1, n_q]``
    :param array_2: array with shape ``[n_samples, 2 * l2 + 1, n_p]``
    :param o3_lambdas: list of degrees of spherical harmonics to compute
    :param cg_coefficients: a :py:class:`TensorMap` containing CG coefficients in a
        format for either sparse or dense CG tensor products, as returned by
        :py:func:`calculate_cg_coefficients`. Only used if ``cg_backend`` is not
        ``"metadata"``.
    :param cg_backend: specifies the backend to use for the calculation. It can be
        ``"python-dense"``, ``"python-sparse"``, and ``"metadata"``. If
        ``"python-dense"`` or ``"python-sparse"`` is chosen, a dense or sparse
        combination (respectively) of the arrays is performed using either numpy or
        torch, depending on the arrays. If ``"metadata"`` is chosen, no combination is
        performed, and an empty array of the correct shape is returned instead.

    :returns: list of arrays of shape ``[n_samples, (2*o3_lambda+1), n_q * n_p]``
    """
    # If just precomputing metadata, return an empty array
    if cg_backend == "metadata":
        return [
            _cg_tensor_product_empty(array_1, array_2, o3_lambda)
            for o3_lambda in o3_lambdas
        ]

    elif cg_backend == "python-sparse":
        return _cg_tensor_product_sparse(array_1, array_2, o3_lambdas, cg_coefficients)
    elif cg_backend == "python-dense":
        return _cg_tensor_product_dense(array_1, array_2, o3_lambdas, cg_coefficients)
    else:
        raise ValueError(
            f"Wrong cg_backend, got '{cg_backend}', "
            "but only support 'metadata', 'python-dense', or 'python-sparse'"
        )


def _cg_tensor_product_empty(array_1: Array, array_2: Array, o3_lambda: int) -> Array:
    """
    Returns an empty array of the correct shape, imitating the output array shape
    produced by a CG combination of ``array_1`` and ``array_2``.
    """
    # Samples dimensions must be the same
    assert array_1.shape[0] == array_2.shape[0]

    # Define other useful dimensions
    n_s = array_1.shape[0]  # number of samples
    n_p = array_1.shape[2]  # number of properties in array_1
    n_q = array_2.shape[2]  # number of properties in array_2

    return _dispatch.empty_like(array_1, (n_s, 2 * o3_lambda + 1, n_p * n_q))


def _cg_tensor_product_sparse(
    array_1: Array,
    array_2: Array,
    o3_lambdas: List[int],
    cg_coefficients: TensorMap,
) -> List[Array]:
    """compute the Clebsch-Gordan tensor product of 2 arrays using sparse operations"""
    # Samples dimensions must be the same
    assert array_1.shape[0] == array_2.shape[0]

    # Infer l1 and l2 from the len of the length of axis 1 of each tensor
    l1 = (array_1.shape[1] - 1) // 2
    l2 = (array_2.shape[1] - 1) // 2

    # Define other useful dimensions
    n_s = array_1.shape[0]  # number of samples
    n_p = array_1.shape[2]  # number of properties in array_1
    n_q = array_2.shape[2]  # number of properties in array_2

    # Compute the partial tensor products
    partial_tensor_products = {}
    for o3_lambda in o3_lambdas:
        cg_l1l2lam = cg_coefficients.block({"l1": l1, "l2": l2, "lambda": o3_lambda})
        for i in range(len(cg_l1l2lam.samples)):
            m1m2mu = cg_l1l2lam.samples.entry(i)
            m1 = m1m2mu[0]
            m2 = m1m2mu[1]

            # We use a string as dict key since TorchScript does not support (int, int)
            dict_key = str((m1, m2))
            if dict_key not in partial_tensor_products:
                partial_tensor_products[dict_key] = (
                    array_1[:, m1, :, None] * array_2[:, m2, None, :]
                ).reshape(n_s, n_p * n_q)

    # Couple l1,l2 to the different lambda, re-using the partial tensor products
    result = []
    for o3_lambda in o3_lambdas:
        output = _cg_couple_sparse(
            partial_tensor_products, l1, l2, o3_lambda, cg_coefficients
        )
        result.append(output)

    return result


def _cg_tensor_product_dense(
    array_1: Array,
    array_2: Array,
    o3_lambdas: List[int],
    cg_coefficients: TensorMap,
) -> List[Array]:
    """compute the Clebsch-Gordan tensor product of 2 arrays using dense operations"""
    # Infer l1 and l2 from the len of the length of axis 1 of each tensor
    l1 = (array_1.shape[1] - 1) // 2
    l2 = (array_2.shape[1] - 1) // 2

    # Define other useful dimensions
    n_s = array_1.shape[0]  # number of samples
    n_p = array_1.shape[2]  # number of properties in array_1
    n_q = array_2.shape[2]  # number of properties in array_2

    # we broadcast it in this way so we only need to do one swapaxes in the next step
    # The resulting shape of tensor_product is [samples, l2, p, l1, q]
    tensor_product = array_1[:, None, None, :, :] * array_2[:, :, :, None, None]

    # [samples, l2, p, l1, q] -> [samples, q, p, l1, l2]
    tensor_product = _dispatch.swapaxes(tensor_product, 1, 4)
    # [samples, l2, p, l1, q] -> [(samples q p), l1, l2]
    tensor_product = tensor_product.reshape(-1, 2 * l1 + 1, 2 * l2 + 1)

    result = []
    for o3_lambda in o3_lambdas:
        # output shape is [(samples q p), lambda]
        output = _cg_couple_dense(tensor_product, o3_lambda, cg_coefficients)
        #  => [samples, (q p), lambda]
        output = output.reshape(n_s, (n_p * n_q), -1)
        #  => [samples, lambda, (q p)]
        output = _dispatch.swapaxes(output, 1, 2)
        result.append(output)

    return result
