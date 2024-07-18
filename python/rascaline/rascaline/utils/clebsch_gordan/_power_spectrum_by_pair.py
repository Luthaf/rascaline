"""
Module containing Power Spectrum calculators.
"""

from typing import List, Optional, Tuple

from ...calculators import SphericalExpansion, SphericalExpansionByPair
from .. import _dispatch
from .._backend import (
    Labels,
    TensorMap,
    TorchModule,
    operations,
    torch_jit_export,
)
from . import _utils
from ._density_correlations import DensityCorrelations
from ._tensor_correlator import TensorCorrelator


class _PowerSpectrumByPair(TorchModule):
    """
    Base class for generating a power spectrum from two generic densities.
    """

    def __init__(
        self,
        spherical_expansion_hypers: dict,
        spherical_expansion_by_pair_hypers: dict,
        atom_types: Optional[List[int]],
        angular_cutoff: Optional[int],
        neighbor_type_to_properties: bool,
        combine_keys_to_properties: bool,
        *,
        tensor_correlator: Optional[TensorCorrelator],
        arrays_backend: Optional[str],
        cg_backend: Optional[str],
    ) -> None:

        super().__init__()

        self._n_correlations = 1  # by definition for power spectrum
        self._atom_types = atom_types
        self._angular_cutoff = angular_cutoff
        self._neighbor_type_to_properties = neighbor_type_to_properties
        self._combine_keys_to_properties = combine_keys_to_properties

        # Initialize the spherical expansion calculator
        self._spherical_expansion_calculator = SphericalExpansion(
            **spherical_expansion_hypers
        )

        # Initialize the spherical expansion by pair calculator
        self._spherical_expansion_by_pair_calculator = SphericalExpansionByPair(
            **spherical_expansion_by_pair_hypers
        )

        # Initialize the TensorCorrelator calculator if not provided
        if tensor_correlator is None:
            max_angular = (  # determine max_angular for CG coefficients
                spherical_expansion_hypers["max_angular"]
                + spherical_expansion_by_pair_hypers["max_angular"]
            )
            if angular_cutoff is not None:
                max_angular = min(max_angular, angular_cutoff)

            self._tensor_correlator = TensorCorrelator(
                max_angular=max_angular,
                arrays_backend=arrays_backend,
                cg_backend=cg_backend,
            )
        else:
            if _dispatch.any(
                [param is not None for param in [arrays_backend, cg_backend]]
            ):
                raise ValueError(
                    "If ``tensor_correlator`` is provided, ``arrays_backend`` and "
                    " ``cg_backend`` should be None."
                )
            self._tensor_correlator = tensor_correlator

        # Initialize the DensityCorrelations calculator. Pass the TensorCorrelator to
        # avoid re-computing the Clebsch-Gordan coefficients.
        self._density_correlations_calculator = DensityCorrelations(
            n_correlations=self._n_correlations,
            angular_cutoff=self._angular_cutoff,
            skip_redundant=False,  # only for density auto-correlations
            tensor_correlator=self._tensor_correlator,
        )

    def _prepare_densities(self, frames) -> Tuple[TensorMap]:
        """
        Generates a :py:class:`SphericalExpansion` and a
        :py:class:`SphericalExpansionByPair` and manipulates their metadata ready for CG
        tensor products.
        """
        # Compute ``density`` as the spherical expansion of the frames
        density = self._spherical_expansion_calculator.compute(frames)

        # Rename dimensions of ``density`` to match those of the pair density generated
        # later:
        #
        # Dimension  |  Original name  |  New name
        # -----------|-----------------|------------------
        # Keys       |  center_type    |  first_atom_type
        # Keys       |  neighbor_type  |  second_atom_type
        # Samples    |  atom           |  first_atom
        density = operations.rename_dimension(
            density, "keys", old="center_type", new="first_atom_type"
        )
        density = operations.rename_dimension(
            density, "keys", old="neighbor_type", new="second_atom_type"
        )
        density = operations.rename_dimension(
            density, "samples", old="atom", new="first_atom"
        )

        # Move "second_atom_type" keys to properties if requested
        if self._neighbor_type_to_properties:
            if self._atom_types is None:
                density = density.keys_to_properties("second_atom_type")
            else:
                density = density.keys_to_properties(
                    Labels(
                        names=["second_atom_type"],
                        values=_dispatch.int_array_like(
                            self._atom_types, like=density.keys.values
                        ).reshape(-1, 1),
                    )
                )

        # Ensure ``density`` has a "_1" prefix for all property dimensions
        density = _utils._increment_property_name_suffices(density, 1)

        # Compute ``density_to_combine`` as the SphericalExpansionByPair of the frames
        density_to_combine = self._spherical_expansion_by_pair_calculator.compute(
            frames
        )

        # Move "second_atom_type" keys to properties if requested
        if self._neighbor_type_to_properties:
            if self._atom_types is None:
                density_to_combine = density_to_combine.keys_to_properties(
                    "second_atom_type"
                )
            else:
                density_to_combine = density_to_combine.keys_to_properties(
                    Labels(
                        names=["second_atom_type"],
                        values=_dispatch.int_array_like(
                            self._atom_types, like=density_to_combine.keys.values
                        ).reshape(-1, 1),
                    )
                )

        # Ensure ``density`` has a "_2" prefix for all property dimensions
        density_to_combine = _utils._increment_property_name_suffices(
            density_to_combine, 2
        )

        return density, density_to_combine

    def _power_spectrum(
        self, frames, selected_keys: Optional[Labels], compute_metadata: bool
    ) -> TensorMap:
        """Generate the power spectrum by pair for a set of frames."""

        # Get the staring densities
        density, density_to_combine = self._prepare_densities(frames)

        # Compute the power spectrum
        power_spectrum_by_pair = (
            self._density_correlations_calculator._density_correlations(
                density,
                density_to_combine,
                selected_keys=selected_keys,
                compute_metadata=compute_metadata,
            )
        )

        if self._combine_keys_to_properties:

            # Move the combination keys to properties
            power_spectrum_by_pair = power_spectrum_by_pair.keys_to_properties(
                [f"l_{x}" for x in range(1, self._n_correlations + 2)]
                + [f"k_{x}" for x in range(2, self._n_correlations + 1)]
            )

        return power_spectrum_by_pair


class EquivariantPowerSpectrumByPair(_PowerSpectrumByPair):
    """
    Equivariant body order 3 tensor produced by the tensor product of a single-center
    spherical expansiona and a pair-wise spherical expansion.
    """

    def __init__(
        self,
        spherical_expansion_hypers: dict,
        spherical_expansion_by_pair_hypers: dict,
        atom_types: Optional[List[int]] = None,
        angular_cutoff: Optional[int] = None,
        neighbor_type_to_properties: bool = True,
        combine_keys_to_properties: bool = True,
        *,
        tensor_correlator: Optional[TensorCorrelator] = None,
        arrays_backend: Optional[str] = None,
        cg_backend: Optional[str] = None,
    ) -> None:

        super().__init__(
            spherical_expansion_hypers=spherical_expansion_hypers,
            spherical_expansion_by_pair_hypers=spherical_expansion_by_pair_hypers,
            atom_types=atom_types,
            angular_cutoff=angular_cutoff,
            neighbor_type_to_properties=neighbor_type_to_properties,
            combine_keys_to_properties=combine_keys_to_properties,
            tensor_correlator=tensor_correlator,
            arrays_backend=arrays_backend,
            cg_backend=cg_backend,
        )

    def forward(self, frames, selected_keys: Optional[Labels] = None) -> TensorMap:
        """
        Calls the :py:meth:`compute` method.

        This is intended for :py:class:`torch.nn.Module` compatibility, and should be
        ignored in pure Python mode.

        See :py:meth:`compute` for a full description of the parameters.
        """
        return self.compute(frames, selected_keys)

    @torch_jit_export
    def compute_metadata(
        self, frames, selected_keys: Optional[Labels] = None
    ) -> TensorMap:
        """
        Returns the metadata-only :py:class:`TensorMap` that would be output by the
        function :py:meth:`compute` for the same calculator under the same settings,
        without performing the actual Clebsch-Gordan tensor products.

        See :py:meth:`compute` for a full description of the parameters.
        """
        return self._power_spectrum(
            frames,
            selected_keys,
            compute_metadata=True,
        )

    def compute(self, frames, selected_keys: Optional[Labels] = None) -> TensorMap:
        """
        Compute the equivariant power spectrum for a set of frames.
        """
        return self._power_spectrum(
            frames,
            selected_keys,
            compute_metadata=False,
        )


class InvariantPowerSpectrumByPair(_PowerSpectrumByPair):
    """
    Invariant body order 3 tensor, i.e. SOAP
    """

    def __init__(
        self,
        spherical_expansion_hypers: dict,
        spherical_expansion_by_pair_hypers: dict,
        atom_types: Optional[List[int]] = None,
        angular_cutoff: Optional[int] = None,
        neighbor_type_to_properties: bool = True,
        combine_keys_to_properties: bool = True,
        *,
        tensor_correlator: Optional[TensorCorrelator] = None,
        arrays_backend: Optional[str] = None,
        cg_backend: Optional[str] = None,
    ) -> None:

        super().__init__(
            spherical_expansion_hypers=spherical_expansion_hypers,
            spherical_expansion_by_pair_hypers=spherical_expansion_by_pair_hypers,
            atom_types=atom_types,
            angular_cutoff=angular_cutoff,
            neighbor_type_to_properties=neighbor_type_to_properties,
            combine_keys_to_properties=combine_keys_to_properties,
            tensor_correlator=tensor_correlator,
            arrays_backend=arrays_backend,
            cg_backend=cg_backend,
        )

        # Define the selected_keys: these are blocks where "o3_lambda" is
        # 0 and "o3_sigma" is 1, by definition for an invariant descriptor
        self._selected_keys = Labels(
            names=["o3_lambda", "o3_sigma"],
            values=_dispatch.int_array_like(
                [0, 1],
                like=self._tensor_correlator._cg_coefficients.keys.values,
            ).reshape(1, 2),
        )

    def forward(self, frames) -> TensorMap:
        """
        Calls the :py:meth:`compute` method.

        This is intended for :py:class:`torch.nn.Module` compatibility, and should be
        ignored in pure Python mode.

        See :py:meth:`compute` for a full description of the parameters.
        """
        return self.compute(frames)

    @torch_jit_export
    def compute_metadata(self, frames) -> TensorMap:
        """
        Returns the metadata-only :py:class:`TensorMap` that would be output by the
        function :py:meth:`compute` for the same calculator under the same settings,
        without performing the actual Clebsch-Gordan tensor products.

        See :py:meth:`compute` for a full description of the parameters.
        """
        return self._power_spectrum(
            frames,
            self._selected_keys,
            compute_metadata=True,
        )

    def compute(self, frames) -> TensorMap:
        """
        Compute the equivariant power spectrum for a set of frames.
        """
        return self._power_spectrum(
            frames,
            self._selected_keys,
            compute_metadata=False,
        )
