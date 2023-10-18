import json
from math import sqrt
from typing import List, Optional, Union

from . import _dispatch
from ._classes import CalculatorBase, IntoSystem, Labels, TensorBlock, TensorMap


class PowerSpectrum:
    def __init__(
        self,
        calculator_1: CalculatorBase,
        calculator_2: Optional[CalculatorBase] = None,
    ):
        r"""General power spectrum of one or of two calculators.

        If ``calculator_2`` is provided, the invariants :math:`p_{nl}` are generated by
        taking quadratic combinations of ``calculator_1``'s spherical expansion
        :math:`\rho_{nlm}` and ``calculator_2``'s spherical expansion :math:`\nu_{nlm}`
        according to `Bartók et. al
        <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.87.184115>`_.

        .. math::
            p_{nl} = \rho_{nlm}^\dagger \cdot \nu_{nlm}

        where we use the Einstein summation convention. If gradients are present the
        invariants of those are constructed as

        .. math::
            \nabla p_{nl} = \nabla \rho_{nlm}^\dagger \cdot \nu_{nlm} +
                            \rho_{nlm}^\dagger \cdot \nabla \nu_{nlm}

        .. note::
            Currently only supports gradients with respect to ``positions``.

        If ``calculator_2=None`` invariants are generated by combining the coefficients
        of the spherical expansion of ``calculator_1``. The spherical expansions given
        as input can only be :py:class:`rascaline.SphericalExpansion` or
        :py:class:`rascaline.LodeSphericalExpansion`.

        :param calculator_1: first calculator
        :param calculator_1: second calculator
        :raises ValueError: If other calculators than
            :py:class:`rascaline.SphericalExpansion` or
            :py:class:`rascaline.LodeSphericalExpansion` are used.
        :raises ValueError: If ``'max_angular'`` of both calculators is different

        Example
        -------
        As an example we calculate the power spectrum for a short range (sr) spherical
        expansion and a long-range (lr) LODE spherical expansion for a NaCl crystal.

        >>> import rascaline
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
        ...     "cutoff": 1.0,
        ...     "max_radial": 6,
        ...     "max_angular": 2,
        ...     "atomic_gaussian_width": 0.3,
        ...     "center_atom_weight": 1.0,
        ...     "radial_basis": {
        ...         "Gto": {},
        ...     },
        ...     "cutoff_function": {
        ...         "ShiftedCosine": {"width": 0.5},
        ...     },
        ... }

        Define the hyper parameters for the long-range LODE spherical expansion from the
        hyper parameters of the short-range spherical expansion

        >>> lr_hypers = sr_hypers.copy()
        >>> lr_hypers.pop("cutoff_function")
        {'ShiftedCosine': {'width': 0.5}}
        >>> lr_hypers["potential_exponent"] = 1

        Construct the calculators

        >>> sr_calculator = rascaline.SphericalExpansion(**sr_hypers)
        >>> lr_calculator = rascaline.LodeSphericalExpansion(**lr_hypers)

        Construct the power spectrum calculators and compute the spherical expansion

        >>> calculator = rascaline.utils.PowerSpectrum(sr_calculator, lr_calculator)
        >>> power_spectrum = calculator.compute(atoms)

        The resulting invariants are stored as :py:class:`metatensor.TensorMap` as for any other calculator

        >>> power_spectrum.keys
        Labels(
            species_center
                  11
                  17
        )
        >>> power_spectrum[0]
        TensorBlock
            samples (1): ['structure', 'center']
            components (): []
            properties (432): ['l', 'species_neighbor_1', 'n1', 'species_neighbor_2', 'n2']
            gradients: None


        .. seealso::
            If you are interested in the SOAP power spectrum you can the use the
            faster :py:class:`rascaline.SoapPowerSpectrum`.
        """  # noqa E501
        self.calculator_1 = calculator_1
        self.calculator_2 = calculator_2

        supported_calculators = ["lode_spherical_expansion", "spherical_expansion"]

        if self.calculator_1.c_name not in supported_calculators:
            raise ValueError(
                f"Only {','.join(supported_calculators)} are supported for "
                "calculator_1!"
            )

        if self.calculator_2 is not None:
            if self.calculator_2.c_name not in supported_calculators:
                raise ValueError(
                    f"Only {','.join(supported_calculators)} are supported for "
                    "calculator_2!"
                )

            parameters_1 = json.loads(calculator_1.parameters)
            parameters_2 = json.loads(calculator_2.parameters)
            if parameters_1["max_angular"] != parameters_2["max_angular"]:
                raise ValueError("'max_angular' of both calculators must be the same!")

    @property
    def name(self):
        """Name of this calculator."""
        return "PowerSpectrum"

    def compute(
        self,
        systems: Union[IntoSystem, List[IntoSystem]],
        gradients: Optional[List[str]] = None,
        use_native_system: bool = True,
    ) -> TensorMap:
        """Runs a calculation with this calculator on the given ``systems``.

        See :py:func:`rascaline.calculators.CalculatorBase.compute()` for details on the
        parameters.

        :raises NotImplementedError: If a spherical expansions contains a gradient with
            respect to an unknwon parameter.
        """
        if gradients is not None:
            for parameter in gradients:
                if parameter != "positions":
                    raise NotImplementedError(
                        "PowerSpectrum currently only supports gradients "
                        "w.r.t. to positions"
                    )

        spherical_expansion_1 = self.calculator_1.compute(
            systems=systems, gradients=gradients, use_native_system=use_native_system
        )

        expected_key_names = [
            "spherical_harmonics_l",
            "species_center",
            "species_neighbor",
        ]

        assert spherical_expansion_1.keys.names == expected_key_names
        assert spherical_expansion_1.property_names == ["n"]

        # Fill blocks with `species_neighbor` from ALL blocks. If we don't do this
        # merging blocks along the ``sample`` direction might be not possible.
        array = spherical_expansion_1.keys.column("species_neighbor")
        keys_to_move = Labels(
            names="species_neighbor",
            values=_dispatch.unique(array).reshape(-1, 1),
        )

        spherical_expansion_1 = spherical_expansion_1.keys_to_properties(keys_to_move)

        if self.calculator_2 is None:
            spherical_expansion_2 = spherical_expansion_1
        else:
            spherical_expansion_2 = self.calculator_2.compute(
                systems=systems,
                gradients=gradients,
                use_native_system=use_native_system,
            )
            assert spherical_expansion_2.keys.names == expected_key_names
            assert spherical_expansion_2.property_names == ["n"]

            array = spherical_expansion_2.keys.column("species_neighbor")
            keys_to_move = Labels(
                names="species_neighbor",
                values=_dispatch.unique(array).reshape(-1, 1),
            )

            spherical_expansion_2 = spherical_expansion_2.keys_to_properties(
                keys_to_move
            )

        new_blocks: List[TensorBlock] = []
        new_keys_values: List[List[int]] = []

        for key, block_1 in spherical_expansion_1.items():
            ell = key[0]
            species_center = key[1]

            factor = (-1) ** ell / sqrt(2 * ell + 1)

            # Find that block indices that have the same spherical_harmonics_l and
            # species_center
            selection = Labels(
                names=["spherical_harmonics_l", "species_center"],
                values=_dispatch.list_to_array(
                    array=spherical_expansion_1.keys.values,
                    data=[[ell, species_center]],
                ),
            )
            blocks_2 = spherical_expansion_2.blocks(selection)
            for block_2 in blocks_2:
                # Make sure that samples are the same. This should not happen.
                assert block_1.samples == block_2.samples

                properties_1 = block_1.properties
                properties_2 = block_2.properties

                n_keys_dimensions = (
                    properties_1.values.shape[1] + properties_2.values.shape[1]
                )

                new_property_values = _dispatch.empty_like(
                    array=properties_1.values,
                    shape=[
                        properties_1.values.shape[0],
                        properties_2.values.shape[0],
                        n_keys_dimensions,
                    ],
                )

                for i, values_1 in enumerate(properties_1.values):
                    for j, values_2 in enumerate(properties_2.values):
                        new_property_values[i, j, : len(values_1)] = values_1
                        new_property_values[i, j, len(values_1) :] = values_2

                properties = Labels(
                    names=["species_neighbor_1", "n1", "species_neighbor_2", "n2"],
                    values=new_property_values.reshape(-1, n_keys_dimensions),
                )

                # Compute the invariants by summation and store the results this is
                # equivalent to an einsum with: ima, imb -> iab
                data = factor * _dispatch.matmul(
                    block_1.values.swapaxes(1, 2), block_2.values
                )

                new_block = TensorBlock(
                    values=data.reshape(data.shape[0], -1),
                    samples=block_1.samples,
                    components=[],
                    properties=properties,
                )

                for parameter in block_1.gradients_list():
                    if parameter == "positions":
                        _positions_gradients(new_block, block_1, block_2, factor)

                new_keys_values.append([ell, species_center])
                new_blocks.append(new_block)

        new_keys = Labels(
            names=["l", "species_center"],
            values=_dispatch.list_to_array(
                array=spherical_expansion_1.keys.values, data=new_keys_values
            ),
        )

        return TensorMap(new_keys, new_blocks).keys_to_properties("l")


def _positions_gradients(
    new_block: TensorBlock, block_1: TensorBlock, block_2: TensorBlock, factor: float
):
    gradient_1 = block_1.gradient("positions")
    gradient_2 = block_2.gradient("positions")

    if len(gradient_1.samples) == 0 or len(gradient_2.samples) == 0:
        gradients_samples = Labels.empty(["sample", "structure", "atom"])
        gradient_values = _dispatch.list_to_array(
            array=gradient_1.values, data=[]
        ).reshape(0, 1, len(new_block.properties))
    else:
        # The "sample" dimension in the power spectrum gradient samples do
        # not necessarily matches the "sample" dimension in the spherical
        # expansion gradient samples. We create new samples by creating a
        # union between the two gradient samples.
        (
            gradients_samples,
            grad1_sample_idxs,
            grad2_sample_idxs,
        ) = gradient_1.samples.union_and_mapping(gradient_2.samples)

        gradient_values = _dispatch.zeros_like(
            array=gradient_1.values,
            shape=[gradients_samples.values.shape[0], 3, len(new_block.properties)],
        )

        # the operation below is equivalent to an einsum with: ixma, imb -> ixab
        sample_indices_1 = _dispatch.to_index_array(gradient_1.samples.column("sample"))
        block_2_values = block_2.values[sample_indices_1]
        new_shape = block_2_values.shape[:1] + (-1,) + block_2_values.shape[1:]

        gradient_1_values = factor * _dispatch.matmul(
            gradient_1.values.swapaxes(2, 3),
            block_2_values.reshape(new_shape),
        )

        gradient_values[grad1_sample_idxs] += gradient_1_values.reshape(
            gradient_1.samples.values.shape[0], 3, -1
        )

        # the operation below is equivalent to an einsum with: ima, ixmb -> ixab
        sample_indices_2 = _dispatch.to_index_array(gradient_2.samples.column("sample"))
        block_1_values = block_1.values[sample_indices_2]
        new_shape = block_1_values.shape[:1] + (-1,) + block_1_values.shape[1:]

        gradient_values_2 = factor * _dispatch.matmul(
            block_1_values.reshape(new_shape).swapaxes(2, 3),
            gradient_2.values,
        )

        gradient_values[grad2_sample_idxs] += gradient_values_2.reshape(
            gradient_2.samples.values.shape[0], 3, -1
        )

    gradient = TensorBlock(
        values=gradient_values,
        samples=gradients_samples,
        components=[gradient_1.components[0]],
        properties=new_block.properties,
    )

    new_block.add_gradient("positions", gradient)
