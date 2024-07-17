"""
Module containing Power Spectrum calculators.
"""

from typing import List, Optional, Tuple

from ...calculators import SphericalExpansion
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


class _PowerSpectrum(TorchModule):
    """
    Base class for generating a power spectrum from two generic densities.
    """

    def __init__(
        self,
        spherical_expansion_hypers: dict,
        atom_types: Optional[List[int]],
        angular_cutoff: Optional[int],
        skip_redundant: bool,
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
        self._skip_redundant = skip_redundant
        self._neighbor_type_to_properties = neighbor_type_to_properties
        self._combine_keys_to_properties = combine_keys_to_properties

        # Initialize the spherical expansion calculator
        self._spherical_expansion_calculator = SphericalExpansion(
            **spherical_expansion_hypers
        )

        # Initialize the TensorCorrelator calculator if not provided
        if tensor_correlator is None:
            max_angular = (  # determine max_angular for CG coefficients
                spherical_expansion_hypers["max_angular"] * 2
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
            skip_redundant=self._skip_redundant,
            tensor_correlator=self._tensor_correlator,
        )

    def _prepare_density(self, frames) -> TensorMap:
        """
        Generates a :py:class:`SphericalExpansion` and manipulates its metadata ready
        for CG tensor products.
        """
        # Compute density
        density = self._spherical_expansion_calculator.compute(frames)

        # Move "neighbor_type" keys to properties
        if self._neighbor_type_to_properties:
            if self._atom_types is None:
                density = density.keys_to_properties("neighbor_type")
            else:
                density = density.keys_to_properties(
                    Labels(
                        names=["neighbor_type"],
                        values=_dispatch.int_array_like(
                            self._atom_types, like=density.keys.values
                        ).reshape(-1, 1),
                    )
                )

        # Modify the names of property dimensions to carry a "_1" suffix
        density = _utils._increment_property_name_suffices(density, 1)

        return density

    def _power_spectrum(
        self, frames, selected_keys: Optional[Labels], compute_metadata: bool
    ) -> TensorMap:
        """Generate the power spectrum for a set of frames."""

        # Get the staring densities
        density = self._prepare_densities(frames)

        # Compute the power spectrum
        power_spectrum = self._density_correlations_calculator._density_correlations(
            tensor=density,
            selected_keys=selected_keys,
            compute_metadata=compute_metadata,
        )

        if self._combine_keys_to_properties:
            # Move the combination keys to properties
            power_spectrum = power_spectrum.keys_to_properties(
                [f"l_{x}" for x in range(1, self._n_correlations + 2)]
                + [f"k_{x}" for x in range(2, self._n_correlations + 1)]
            )

        return power_spectrum


class EquivariantPowerSpectrum(_PowerSpectrum):
    """
    Equivariant body order 3 tensor, i.e. equivariant-SOAP
    """

    def __init__(
        self,
        spherical_expansion_hypers: dict,
        atom_types: Optional[List[int]] = None,
        angular_cutoff: Optional[int] = None,
        skip_redundant: bool = True,
        neighbor_type_to_properties: bool = True,
        combine_keys_to_properties: bool = True,
        *,
        tensor_correlator: Optional[TensorCorrelator] = None,
        arrays_backend: Optional[str] = None,
        cg_backend: Optional[str] = None,
    ) -> None:

        super().__init__(
            spherical_expansion_hypers=spherical_expansion_hypers,
            atom_types=atom_types,
            angular_cutoff=angular_cutoff,
            skip_redundant=skip_redundant,
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


class InvariantPowerSpectrum(_PowerSpectrum):
    """
    Invariant body order 3 tensor, i.e. SOAP
    """

    def __init__(
        self,
        spherical_expansion_hypers: dict,
        atom_types: Optional[List[int]] = None,
        angular_cutoff: Optional[int] = None,
        skip_redundant: bool = True,
        neighbor_type_to_properties: bool = True,
        combine_keys_to_properties: bool = True,
        *,
        tensor_correlator: Optional[TensorCorrelator] = None,
        arrays_backend: Optional[str] = None,
        cg_backend: Optional[str] = None,
    ) -> None:

        super().__init__(
            spherical_expansion_hypers=spherical_expansion_hypers,
            atom_types=atom_types,
            angular_cutoff=angular_cutoff,
            skip_redundant=skip_redundant,
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
