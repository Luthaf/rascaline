"""
This module provides convenience calculators for preforming density correlations, i.e.
the (iterative) CG tensor products of density (body order 2) tensors.

All of these calculators wrap the :py:class:`ClebschGordanProduct` class, handling the
higher-level metadata manipulation to produce the desired output tensors.
"""

from typing import List, Optional

from .. import _dispatch
from .._backend import Device, DType, Labels, TensorMap, TorchModule, operations
from ._cg_product import ClebschGordanProduct


class DensityCorrelations(TorchModule):
    """
    Takes ``n_correlations`` of iterative CG tensor products of a density with itself to
    produce a density auto-correlation tensor of higher correlation order.

    The constructor computes and stores the CG coefficients. The :py:meth:`compute`
    method computes the auto-correlation by CG tensor product of the input density.
    """

    def __init__(
        self,
        *,
        n_correlations: int,
        max_angular: int,
        skip_redundant: bool = True,
        cg_backend: Optional[str] = None,
        arrays_backend: Optional[str] = None,
        dtype: Optional[DType] = None,
        device: Optional[Device] = None,
    ):
        """
        :param n_correlations: :py:class:`int`, the number of iterative CG tensor
                products to perform.
        :param max_angular: :py:class:`int`, the maximum angular momentum to compute CG
            coefficients for. This must be large enough to perform the desired number of
            correlations on the density passed to the :py:meth:`compute` method.
        :param skip_redundant: :py:class:`bool`, whether to skip redundant CG
            combination steps without losing information in the output. Setting to
            ``True`` can save computation time, but does not preserve the norm of the
            output tensors.
        :param cg_backend: :py:class:`str`, the backend to use for the CG tensor
            product. If ``None``, the backend is automatically selected based on the
            arrays backend.
        :param arrays_backend: :py:class:`str`, the backend to use for array operations.
            If ``None``, the backend is automatically selected based on the environment.
            Possible values are "numpy" and "torch".
        :param dtype: the scalar type to use to store coefficients
        :param device: the computational device to use for calculations. This must be
            ``"cpu"`` if ``array_backend="numpy"``.
        """

        super().__init__()

        self._n_correlations = n_correlations
        self._cg_product = ClebschGordanProduct(
            max_angular=max_angular,
            cg_backend=cg_backend,
            keys_filter=_filter_redundant_keys if skip_redundant else None,
            arrays_backend=arrays_backend,
            dtype=dtype,
            device=device,
        )

    def compute(
        self,
        density: TensorMap,
        angular_cutoff: Optional[int] = None,
        selected_keys: Optional[Labels] = None,
    ) -> TensorMap:
        """
        Takes ``n_correlations`` of iterative CG tensor products of a density with
        itself, to generate density auto-correlations of arbitrary correlation order.

        .. math::

            \\rho^{\\nu=n_{corr} + 1} = \\rho^{\\nu=1}
                                        \\otimes \\rho^{\\nu=1} \\ldots \\otimes
                                        \\rho^{\\nu=1}

        where \\rho^{\\nu=1} is the input ``density`` of correlation order 1 (body order
        2), and \\rho^{\\nu=n_{corr} + 1} is the output density of correlation order
        ``n_correlations + 1``

        Before performing any correlations, the properties dimensions of ``density`` are
        modified to carry a "_1" suffix. At each iteration, the dimension names of the
        copy of the density being correlated are incremented by one each time.

        ``selected_keys`` can be passed to select the keys to compute on the final
        iteration. If ``None``, all keys are computed. To limit the maximum angular
        momenta to compute on **intermediate** iterations, pass ``angular_cutoff``.

        If ``angular_cutoff`` and ``selected_keys`` are both passed, ``angular_cutoff``
        is ignored on the final iteration.

        :param density: :py:class:`TensorMap`, the input density tensor of correlation
            order 1.
        :param angular_cutoff: :py:class:`int`, the maximum angular momentum to compute
            output blocks for at each iteration. If ``selected_keys`` is passed, this
            parameter is applied to all intermediate (i.e. prior to the final)
            iterations. If ``selected_keys`` is not passed, this parameter is applied
            globally to all iterations. If ``None``, all angular momenta are computed.
        :param selected_keys: :py:class:`Labels`, the keys to compute on the final
            iteration. If ``None``, all keys are computed. Subsets of key dimensions can
            be passed to compute output blocks that match in these dimensions.

        :return: :py:class:`TensorMap`, the output density auto-correlation tensor of
            correlation order ``n_correlations + 1``.
        """
        return self._density_correlations(
            density,
            angular_cutoff=angular_cutoff,
            selected_keys=selected_keys,
            compute_metadata=False,
        )

    def forward(
        self,
        density: TensorMap,
        angular_cutoff: Optional[int] = None,
        selected_keys: Optional[Labels] = None,
    ) -> TensorMap:
        """
        Calls the :py:meth:`compute` method.

        This is intended for :py:class:`torch.nn.Module` compatibility, and should be
        ignored in pure Python mode.

        See :py:meth:`compute` for a full description of the parameters.
        """
        return self.compute(
            density, angular_cutoff=angular_cutoff, selected_keys=selected_keys
        )

    def compute_metadata(
        self,
        density: TensorMap,
        angular_cutoff: Optional[int] = None,
        selected_keys: Optional[Labels] = None,
    ) -> TensorMap:
        """
        Returns the metadata-only :py:class:`TensorMap` that would be output by the
        function :py:meth:`compute` for the same calculator under the same settings,
        without performing the actual Clebsch-Gordan tensor products.

        See :py:meth:`compute` for a full description of the parameters.
        """
        return self._density_correlations(
            density,
            angular_cutoff=angular_cutoff,
            selected_keys=selected_keys,
            compute_metadata=True,
        )

    def _density_correlations(
        self,
        density: TensorMap,
        angular_cutoff: Optional[int],
        selected_keys: Optional[Labels],
        compute_metadata: bool,
    ) -> TensorMap:
        """
        Computes the density auto-correlations.
        """
        # Prepare the tensors. Before the first iteration, both the 'left' and 'right'
        # densities should have a "_1" suffix in the property names. The property names
        # of the 'right' density is incremented by one at the start of each iteration.
        density_correlations = _increment_property_names(density, 1)
        density = _increment_property_names(density, 1)

        # Perform iterative CG tensor products
        new_lambda_names: List[str] = []
        for i_correlation in range(self._n_correlations):
            # Increment the density property dimension names
            density = _increment_property_names(density, 1)

            # Define new key dimension names for tracking intermediate correlations. The
            # following conditional clause is a matter of convention. "l" indices track
            # blocks of nu=1 densities, and "k" indices track blocks originating from
            # higher order tensors.
            if i_correlation == 0:
                o3_lambda_1_new_name: str = f"l_{i_correlation + 1}"
            else:
                o3_lambda_1_new_name: str = f"k_{i_correlation + 1}"
            o3_lambda_2_new_name: str = f"l_{i_correlation + 2}"
            new_lambda_names.extend([o3_lambda_1_new_name, o3_lambda_2_new_name])

            # Define the selected keys for the current iteration, applying the
            # ``angular_cutoff`` if specified.
            if (
                i_correlation == self._n_correlations - 1 and selected_keys is None
            ) or (i_correlation < self._n_correlations - 1):
                if angular_cutoff is None:
                    selected_keys_iteration = None
                else:
                    selected_keys_iteration: Labels = Labels(
                        names=["o3_lambda"],
                        values=_dispatch.int_range_like(
                            0, angular_cutoff + 1, like=density.keys.values
                        ).reshape(-1, 1),
                    )
            else:
                assert i_correlation == self._n_correlations - 1
                assert selected_keys is not None
                selected_keys_iteration = selected_keys

            # Compute CG tensor product
            if compute_metadata:
                density_correlations = self._cg_product.compute_metadata(
                    density_correlations,
                    density,
                    o3_lambda_1_new_name,
                    o3_lambda_2_new_name,
                    selected_keys=selected_keys_iteration,
                )
            else:
                density_correlations = self._cg_product.compute(
                    density_correlations,
                    density,
                    o3_lambda_1_new_name,
                    o3_lambda_2_new_name,
                    selected_keys=selected_keys_iteration,
                )

        return density_correlations


def _filter_redundant_keys(keys: Labels) -> List[int]:
    """
    Filter redundant keys from the ``keys`` to only keep keys where the l values are
    sorted (i.e. l1 <= l2 <= ... <= ln). These are redundant when handling
    auto-correlations of a density.
    """
    # Infer the correlation order of the output TensorMap by the length of the "l" list
    # present in the keys
    nu_target, present = 1, True
    while present:
        if f"l_{nu_target + 1}" in keys.names:
            nu_target += 1
        else:
            present = False

    # As this function is only valid for the output of a density auto-correlation, we
    # assert the "k" list is present in the keys
    assert all([f"k_{k}" in keys.names for k in range(2, nu_target)])

    # Now find the column idxs of the "l" values in the "l" list
    l_list_idxs = [
        keys.names.index(f"l_{o3_lambda}") for o3_lambda in range(1, nu_target + 1)
    ]

    # Identify keys of redundant blocks and remove them
    keys_to_keep: List[int] = []
    for key_idx in range(len(keys)):
        key = keys.entry(key_idx)

        # Get the list of "l" values
        l_list_values = _dispatch.to_int_list(key.values[l_list_idxs])

        # Keep this key if the l list is already sorted
        if _dispatch.all(
            _dispatch.int_array_like(l_list_values, like=key.values)
            == _dispatch.int_array_like(sorted(l_list_values), like=key.values)
        ):
            keys_to_keep.append(key_idx)

    return keys_to_keep


def _increment_property_names(
    tensor: TensorMap, increment_by: Optional[int] = None
) -> TensorMap:
    """
    Increments all the property dimension names of the input :py:class:`TensorMap`
    ``tensor`` by ``increment_by`` and returns the resulting :py:class:`TensorMap`.

    Any property dimension that is already suffixed by "_{x}", where "x" is an integer,
    will have the integer incremented by ``increment_by``. If the property dimension is
    not suffixed by "_{x}", then "_0" is implied and then this is incremented.

    All property dimensions are assumed to have the format "{property_name}_{x}". The
    only exception are property names that indicate type, and have the format
    "{name}_type". In these cases, the numeric index precedes the "_type" suffix.

    For instance, if the input tensor has property names ["n", "neighbor_type"], the
    property names of the output of ``_increment_property_names(tensor, 2)`` will
    be ["n_2", "neighbor_2_type"]. If the input tensor has property names ["n",
    "neighbor_1_type"], the output property names of the same function call will be
    ["n_2", "neighbor_3_type"].

    :param tensor: :py:class:`TensorMap`, the input tensor.
    :param increment_by: :py:class:`int`, the integer amount to increment the numeric
        suffixes by. Must be a non-negative integer. If ``None``, the default value of 1
        is used.

    :return: :py:class:`TensorMap`, the input tensor with all property dimension names
        incremented by ``increment_by``, according to the rules described above.
    """
    if increment_by is None:
        increment_by = 1
    if increment_by < 0:
        raise ValueError("`increment_by` must be a non-negative integer.")

    for name in tensor.property_names:
        tensor = operations.rename_dimension(
            tensor,
            "properties",
            name,
            _increment_numeric_suffix(name, increment_by),
        )

    return tensor


def _increment_numeric_suffix(name: str, increment_by: Optional[int] = 1) -> str:
    """
    Takes a string name that is suffix by "_{x}", where "x" is an integer,
    and increments the integer by ``increment_by``. Returns the new name.

    If the string is not suffixed by "_{x}", then "_1" is appended to the string.

    The only special cases are names suffixed by "_type", in which case the numeric
    index is inserted and incremented before the "_type" suffix.

    Examples:
        - fn("n", 1) -> "n_1"
        - fn("n", 0) -> "n_0"
        - fn("n_1", 1) -> "n_2"
        - fn("center_type", 3) -> "center_3_type"
        - fn("first_atom_1_type") -> "first_atom_2_type"
        - fn("center_type_1_type", 7) -> "center_type_8_type"
    """
    if not isinstance(increment_by, int):
        raise ValueError("`increment_by` must be an integer.")

    if name.endswith("type"):
        # Special name case: remove the "_type" suffix and add it back on at the end
        prefix: str = name[: name.rfind("_type")]
        suffix: str = "_type"
    else:
        prefix: str = name
        suffix: str = ""

    number: str = ""
    if prefix.rfind("_") == -1:  # no suffix present
        number = "0"
    else:
        number = prefix[prefix.rfind("_") + 1 :]
        if number.isdigit():  # number found
            prefix = prefix[: prefix.rfind("_")]
        else:
            number = "0"

    return prefix + "_" + str(int(number) + increment_by) + suffix
