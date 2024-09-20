"""
This module provides a convenience calculator for computing a single-center equivariant
power spectrum.
"""

from typing import List, Optional, Union

from ...calculators import SphericalExpansion
from ...systems import IntoSystem
from .._backend import Device, DType, Labels, TensorMap, TorchModule, operations
from ._cg_product import ClebschGordanProduct
from ._density_correlations import _filter_redundant_keys


class EquivariantPowerSpectrum(TorchModule):
    """
    Computes an equivariant power spectrum descriptor, or a "lambda-SOAP".

    For a full description of the hyper-parameters, see the corresponding
    :ref:`documentation <soap-power-spectrum>`.
    """

    def __init__(
        self,
        cutoff,
        max_radial,
        max_angular,
        atomic_gaussian_width,
        center_atom_weight,
        radial_basis,
        cutoff_function,
        radial_scaling=None,
        *,
        dtype: Optional[DType] = None,
        device: Optional[Device] = None,
    ):
        """
        :param spherical_expansion_hypers: :py:class:`dict` containing the
            hyper-parameters used to initialize a :py:class:`SphericalExpansion` for
            computing the initial density.
        :param atom_types: :py:class:`list` of :py:class:`str`, the global atom types
            to compute neighbor correlations for. Ensures consistent global properties
            dimensions.
        :param dtype: the scalar type to use to store coefficients
        :param device: the computational device to use for calculations. This must be
            ``"cpu"`` if ``array_backend="numpy"``.
        """

        super().__init__()

        parameters = {
            "cutoff": cutoff,
            "max_radial": max_radial,
            "max_angular": max_angular,
            "atomic_gaussian_width": atomic_gaussian_width,
            "center_atom_weight": center_atom_weight,
            "radial_basis": radial_basis,
            "cutoff_function": cutoff_function,
        }

        if radial_scaling is not None:
            parameters["radial_scaling"] = radial_scaling

        self._spherical_expansion = SphericalExpansion(**parameters)
        self._cg_product = ClebschGordanProduct(
            max_angular=max_angular * 2,
            cg_backend=None,
            keys_filter=_filter_redundant_keys,
            arrays_backend=None,
            dtype=dtype,
            device=device,
        )

    def compute(
        self,
        systems: Union[IntoSystem, List[IntoSystem]],
        *,
        selected_keys: Optional[Labels] = None,
        neighbors_to_properties: bool = False,
    ) -> TensorMap:
        """
        Computes an equivariant power spectrum, or "lambda-SOAP".

        First computes a :py:class:`SphericalExpansion` density descriptor of body order
        2.

        The key dimension 'neighbor_type' is then moved to properties so that they are
        correlated. The global atom types passed in the constructor are taken into
        account so that a consistent output properties dimension is achieved regardless
        of the atomic composition of the systems passed in ``systems``.

        Finally a single Clebsch-Gordan tensor product is taken to produce a body order
        3 equivariant power spectrum, or "lambda-SOAP".

        :param selected_keys: :py:class:`Labels`, the output keys to computed. If
            ``None``, all keys are computed. Subsets of key dimensions can be passed to
            compute output blocks that match in these dimensions.
        :param neighbors_to_properties: :py:class:`bool`, if true, densifies the
            spherical expansion by moving key dimension "neighbor_type" to properties
            prior to performing the Clebsch Gordan product step. Note: typically,
            setting to true results in a speed up of the computation. Defaults to false.

        :return: :py:class:`TensorMap`, the output equivariant power spectrum.
        """
        return self._equivariant_power_spectrum(
            systems=systems,
            selected_keys=selected_keys,
            neighbors_to_properties=neighbors_to_properties,
            compute_metadata=False,
        )

    def forward(
        self,
        systems: Union[IntoSystem, List[IntoSystem]],
        *,
        selected_keys: Optional[Labels] = None,
        neighbors_to_properties: bool = False,
    ) -> TensorMap:
        """
        Calls the :py:meth:`compute` method.

        This is intended for :py:class:`torch.nn.Module` compatibility, and should be
        ignored in pure Python mode.

        See :py:meth:`compute` for a full description of the parameters.
        """
        return self.compute(
            systems=systems,
            selected_keys=selected_keys,
            neighbors_to_properties=neighbors_to_properties,
        )

    def compute_metadata(
        self,
        systems: Union[IntoSystem, List[IntoSystem]],
        *,
        selected_keys: Optional[Labels] = None,
        neighbors_to_properties: bool = False,
    ) -> TensorMap:
        """
        Returns the metadata-only :py:class:`TensorMap` that would be output by the
        function :py:meth:`compute` for the same calculator under the same settings,
        without performing the actual Clebsch-Gordan tensor products in the second step.

        See :py:meth:`compute` for a full description of the parameters.
        """
        return self._equivariant_power_spectrum(
            systems=systems,
            selected_keys=selected_keys,
            neighbors_to_properties=neighbors_to_properties,
            compute_metadata=True,
        )

    def _equivariant_power_spectrum(
        self,
        systems: Union[IntoSystem, List[IntoSystem]],
        selected_keys: Optional[Labels],
        neighbors_to_properties: bool,
        compute_metadata: bool,
    ) -> TensorMap:
        """
        Computes the equivariant power spectrum, either fully or just metadata
        """
        # Compute density
        density = self._spherical_expansion.compute(systems)

        # Rename "neighbor_type" dimension so they are correlated
        density_1 = operations.rename_dimension(
            density, "keys", "neighbor_type", "neighbor_1_type"
        )
        density_2 = operations.rename_dimension(
            density, "keys", "neighbor_type", "neighbor_2_type"
        )
        density_1 = operations.rename_dimension(density_1, "properties", "n", "n_1")
        density_2 = operations.rename_dimension(density_2, "properties", "n", "n_2")

        if neighbors_to_properties:
            density_1 = density_1.keys_to_properties("neighbor_1_type")
            density_2 = density_2.keys_to_properties("neighbor_2_type")

        # Compute the power spectrum
        if compute_metadata:
            pow_spec = self._cg_product.compute_metadata(
                tensor_1=density_1,
                tensor_2=density_2,
                o3_lambda_1_new_name="l_1",
                o3_lambda_2_new_name="l_2",
                selected_keys=selected_keys,
            )
        else:
            pow_spec = self._cg_product.compute(
                tensor_1=density_1,
                tensor_2=density_2,
                o3_lambda_1_new_name="l_1",
                o3_lambda_2_new_name="l_2",
                selected_keys=selected_keys,
            )

        # Move the CG combination info keys to properties
        pow_spec = pow_spec.keys_to_properties(["l_1", "l_2"])

        return pow_spec
