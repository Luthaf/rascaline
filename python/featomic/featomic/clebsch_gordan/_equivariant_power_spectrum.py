"""
This module provides a convenience calculator for computing a single-center equivariant
power spectrum.
"""

import json
from typing import List, Optional, Union

from . import _dispatch
from ._backend import (
    CalculatorBase,
    Device,
    DType,
    IntoSystem,
    Labels,
    TensorMap,
    TorchModule,
    operations,
)
from ._cg_product import ClebschGordanProduct
from ._density_correlations import _filter_redundant_keys


class EquivariantPowerSpectrum(TorchModule):
    r"""
    Computes a general equivariant power spectrum descriptor of two calculators.

    If only ``calculator_1`` is provided, the power spectrum is computed as the density
    auto-correlation of the density produced by the first calculator. If
    ``calculator_2`` is also provided, the power spectrum is computed as the density
    cross-correlation of the densities produced by the two calculators.

    Example
    -------
    As an example we calculate the equivariant power spectrum for a short range (sr)
    spherical expansion and a long-range (lr) LODE spherical expansion for a NaCl
    crystal.

    >>> import featomic
    >>> import ase

    Construct the NaCl crystal

    >>> atoms = ase.Atoms(
    ...     symbols="NaCl",
    ...     positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
    ...     pbc=True,
    ...     cell=[1, 1, 1],
    ... )

    Define the hyper parameters for the short-range spherical expansion

    >>> sr_hypers = {
    ...     "cutoff": {
    ...         "radius": 1.0,
    ...         "smoothing": {"type": "ShiftedCosine", "width": 0.5},
    ...     },
    ...     "density": {
    ...         "type": "Gaussian",
    ...         "width": 0.3,
    ...     },
    ...     "basis": {
    ...         "type": "TensorProduct",
    ...         "max_angular": 2,
    ...         "radial": {"type": "Gto", "max_radial": 5},
    ...     },
    ... }

    Define the hyper parameters for the long-range LODE spherical expansion from the
    hyper parameters of the short-range spherical expansion

    >>> lr_hypers = {
    ...     "density": {
    ...         "type": "SmearedPowerLaw",
    ...         "smearing": 0.3,
    ...         "exponent": 1,
    ...     },
    ...     "basis": {
    ...         "type": "TensorProduct",
    ...         "max_angular": 2,
    ...         "radial": {"type": "Gto", "max_radial": 3, "radius": 1.0},
    ...     },
    ... }

    Construct the calculators

    >>> sr_calculator = featomic.SphericalExpansion(**sr_hypers)
    >>> lr_calculator = featomic.LodeSphericalExpansion(**lr_hypers)

    Construct the power spectrum calculators and compute the spherical expansion

    >>> calculator = featomic.clebsch_gordan.EquivariantPowerSpectrum(
    ...     sr_calculator, lr_calculator
    ... )
    >>> power_spectrum = calculator.compute(atoms, neighbors_to_properties=True)

    The resulting equivariants are stored as :py:class:`metatensor.TensorMap` as for any
    other calculator. The keys contain the symmetry information:

    >>> power_spectrum.keys
    Labels(
        o3_lambda  o3_sigma  center_type
            0         1          11
            1         1          11
            2         1          11
            1         -1         11
            2         -1         11
            3         1          11
            3         -1         11
            4         1          11
            0         1          17
            1         1          17
            2         1          17
            1         -1         17
            2         -1         17
            3         1          17
            3         -1         17
            4         1          17
    )

    The block properties contain the angular order of the combined blocks ("l_1",
    "l_2"), along with the neighbor types ("neighbor_1_type", "neighbor_2_type") and
    radial channel indices.

    >>> power_spectrum[0].properties.names
    ['l_1', 'l_2', 'neighbor_1_type', 'n_1', 'neighbor_2_type', 'n_2']

    .. seealso::
        Faster power spectrum calculator specifically for invariant descriptors can
        be found at :py:class:`featomic.SoapPowerSpectrum` and
        :py:class:`featomic.clebsch_gordan.PowerSpectrum`.
    """

    def __init__(
        self,
        calculator_1: CalculatorBase,
        calculator_2: Optional[CalculatorBase] = None,
        neighbor_types: Optional[List[int]] = None,
        *,
        dtype: Optional[DType] = None,
        device: Optional[Device] = None,
    ):
        """
        Constructs the equivariant power spectrum calculator.

        :param calculator_1: first calculator that computes a density descriptor, either
            a :py:class:`featomic.SphericalExpansion` or
            :py:class:`featomic.LodeSphericalExpansion`.
        :param calculator_2: optional second calculator that computes a density
            descriptor, either a :py:class:`featomic.SphericalExpansion` or
            :py:class:`featomic.LodeSphericalExpansion`. If ``None``, the equivariant
            power spectrum is computed as the auto-correlation of the first calculator.
            Defaults to ``None``.
        :param neighbor_types: List of ``"neighbor_type"`` to use in the properties of
            the output. This option might be useful when running the calculation on
            subset of a whole dataset and trying to join along the ``sample`` dimension
            after the calculation. If ``None``, blocks are filled with
            ``"neighbor_type"`` found in the systems. This parameter is only used if
            ``neighbors_to_properties=True`` is passed to the :py:meth:`compute` method.
        :param dtype: the scalar type to use to store coefficients
        :param device: the computational device to use for calculations.
        """

        super().__init__()
        self.calculator_1 = calculator_1
        self.calculator_2 = calculator_2
        self.neighbor_types = neighbor_types
        self.dtype = dtype
        self.device = device

        supported_calculators = ["lode_spherical_expansion", "spherical_expansion"]

        if self.calculator_1.c_name not in supported_calculators:
            raise ValueError(
                f"Only [{', '.join(supported_calculators)}] are supported for "
                f"`calculator_1`, got '{self.calculator_1.c_name}'"
            )

        parameters_1 = json.loads(calculator_1.parameters)

        if self.calculator_2 is None:
            parameters_2 = parameters_1
        else:
            if self.calculator_2.c_name not in supported_calculators:
                raise ValueError(
                    f"Only [{', '.join(supported_calculators)}] are supported for "
                    f"`calculator_2`, got '{self.calculator_2.c_name}'"
                )

            parameters_2 = json.loads(calculator_2.parameters)
            if parameters_1["basis"]["type"] != "TensorProduct":
                raise ValueError(
                    "only 'TensorProduct' basis is supported for calculator_1"
                )

            if parameters_2["basis"]["type"] != "TensorProduct":
                raise ValueError(
                    "only 'TensorProduct' basis is supported for calculator_2"
                )

        self._cg_product = ClebschGordanProduct(
            max_angular=parameters_1["basis"]["max_angular"]
            + parameters_2["basis"]["max_angular"],
            cg_backend=None,
            keys_filter=_filter_redundant_keys,
            arrays_backend=None,
            dtype=dtype,
            device=device,
        )

    @property
    def name(self):
        """Name of this calculator."""
        return "EquivariantPowerSpectrum"

    def compute(
        self,
        systems: Union[IntoSystem, List[IntoSystem]],
        selected_keys: Optional[Labels] = None,
        neighbors_to_properties: bool = False,
    ) -> TensorMap:
        """
        Computes an equivariant power spectrum, also called "Lambda-SOAP" when doing a
        self-correlation of the SOAP density.

        First computes a :py:class:`SphericalExpansion` density descriptor of body order
        2.

        Before performing the Clebsch-Gordan tensor product, the spherical expansion
        density can be densified by moving the key dimension "neighbor_type" to the
        block properties. This is controlled by the ``neighbors_to_properties``
        parameter. Depending on the specific systems descriptors are being computed for,
        the sparsity or density of the density can affect the computational cost of the
        Clebsch-Gordan tensor product.

        If ``neighbors_to_properties=True`` and ``neighbor_types`` have been passed to
        the constructor, property dimensions are created for all of these global atom
        types when moving the key dimension to properties. This ensures that the output
        properties dimension is of consistent size across all systems passed in
        ``systems``.

        Finally a single Clebsch-Gordan tensor product is taken to produce a body order
        3 equivariant power spectrum.

        :param selected_keys: :py:class:`Labels`, the output keys to computed. If
            ``None``, all keys are computed. Subsets of key dimensions can be passed to
            compute output blocks that match in these dimensions.
        :param neighbors_to_properties: :py:class:`bool`, if true, densifies the
            spherical expansion by moving key dimension "neighbor_type" to properties
            prior to performing the Clebsch Gordan product step. Defaults to false.

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
        density_1 = self.calculator_1.compute(systems)

        if self.calculator_2 is None:
            density_2 = density_1
        else:
            density_2 = self.calculator_2.compute(systems)

        # Rename "neighbor_type" dimension so they are correlated
        density_1 = operations.rename_dimension(
            density_1, "keys", "neighbor_type", "neighbor_1_type"
        )
        density_2 = operations.rename_dimension(
            density_2, "keys", "neighbor_type", "neighbor_2_type"
        )
        density_1 = operations.rename_dimension(density_1, "properties", "n", "n_1")
        density_2 = operations.rename_dimension(density_2, "properties", "n", "n_2")

        if neighbors_to_properties:
            if self.neighbor_types is None:  # just move neighbor type
                keys_to_move_1 = "neighbor_1_type"
                keys_to_move_2 = "neighbor_2_type"
            else:  # use the user-specified types
                values = _dispatch.list_to_array(
                    array=density_1.keys.values,
                    data=[[t] for t in self.neighbor_types],
                )
                keys_to_move_1 = Labels(names="neighbor_1_type", values=values)
                keys_to_move_2 = Labels(names="neighbor_2_type", values=values)

            density_1 = density_1.keys_to_properties(keys_to_move_1)
            density_2 = density_2.keys_to_properties(keys_to_move_2)

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
