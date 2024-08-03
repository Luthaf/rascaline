"""
Module containing a convenience calculator that returns the output of a CG tensor
product between :py:call:`EquivariantPowerSpectrum` and :py:class:`SphericalExpansionByPair`
equivariant descriptors.
"""

from typing import List, Optional, Union

import numpy as np

from ...calculators import SphericalExpansion, SphericalExpansionByPair
from .. import _dispatch
from .._backend import (
    Labels,
    TensorBlock,
    TensorMap,
    TorchModule,
    TorchScriptClass,
    operations,
    torch_jit_export,
    torch_jit_is_scripting,
)
from . import _coefficients, _utils
from ._correlate_density import DensityCorrelations
from ._correlate_tensors import CorrelateTensorWithDensity


try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ======================================================================
# ===== Public API functions
# ======================================================================


class EquivariantPowerSpectrum(TorchModule):
    """
    Computes a Lambda-SOAP equivariant descriptor.
    """

    def __init__(
        self,
        spherical_expansion_hypers: dict,
        atom_types: Optional[List[int]] = None,
        angular_cutoff: Optional[int] = None,
        selected_keys: Optional[Labels] = None,
    ) -> None:

        self._spherical_expansion_hypers = spherical_expansion_hypers
        self._atom_types = atom_types
        self._spherical_expansion_calc = SphericalExpansion(
            **spherical_expansion_hypers
        )

        density_correlations_hypers = {
            "max_angular": self._spherical_expansion_hypers["max_angular"] * 2,
            "body_order": 3,
            "angular_cutoff": angular_cutoff,
            "selected_keys": selected_keys,
            "match_keys": ["center_type"],
            "skip_redundant": True,
            "output_selection": None,
            "keep_l_in_keys": False,
            "arrays_backend": None,
            "cg_backend": None,
        }
        self._density_correlations_hypers = density_correlations_hypers
        self._density_correlations_calc = DensityCorrelations(
            **density_correlations_hypers
        )

    def forward(self, frames) -> TensorMap:
        """
        Calls the :py:meth:`EquivariantPowerSpectrum.compute` function.

        This is intended for :py:class:`torch.nn.Module` compatibility, and should be
        ignored in pure Python mode.
        """
        return self.compute(frames)

    def compute(self, frames) -> TensorMap:
        """
        Computes the Lambda-SOAP descriptor for the given ``frames``.

        :param frames: The input frames, i.e. ase Atoms objects.
        """
        # Compute the nu = 1 density and move "neighbor_type" to properties
        density = self._spherical_expansion_calc.compute(frames)
        if self._atom_types is None:
            keys_to_move = ["neighbor_type"]
        else:
            keys_to_move = Labels(
                names=["neighbor_type"],
                values=_dispatch.int_array_like(
                    self._atom_types, like=density.keys.values
                ).reshape(-1, 1),
            )
        density = density.keys_to_properties(keys_to_move)

        # Compute the nu = 2 lambda-SOAP descriptor
        lsoap = self._density_correlations_calc.compute(density)

        return lsoap


class EquivariantPowerSpectrumByPair(TorchModule):
    """
    Computes a Lambda-SOAP equivariant descriptor and takes the CG tensor product with a
    SphericalExansionByPair descriptor.
    """

    def __init__(
        self,
        spherical_expansion_hypers: dict,
        spherical_expansion_by_pair_hypers: dict,
        atom_types: Optional[List[int]] = None,
        angular_cutoff: Optional[int] = None,
        selected_keys: Optional[Labels] = None,
    ) -> None:
        super().__init__()

        self._spherical_expansion_hypers = spherical_expansion_hypers
        self._spherical_expansion_by_pair_hypers = spherical_expansion_by_pair_hypers
        self._atom_types = atom_types
        self._angular_cutoff = angular_cutoff
        self._selected_keys = selected_keys
        self._spherical_expansion_calc = SphericalExpansion(
            **spherical_expansion_hypers
        )
        self._spherical_expansion_by_pair_calc = SphericalExpansionByPair(
            **spherical_expansion_by_pair_hypers
        )

        density_correlations_hypers = {
            # We will re-use the CG coefficients computed by the DensityCorrelations
            # constructor when initializing the CorrelateTensorWithDensity calculator.
            # As such, we need to increase the max_angular to cover two iterations of CG
            # tensor products.
            "max_angular": self._spherical_expansion_hypers["max_angular"] * 3,
            "body_order": 3,
            "angular_cutoff": self._angular_cutoff,
            "selected_keys": None,  # only apply key selection when doing rho_i x g_ij
            # The "center_type" key dimension will be renamed to "first_atom_type"
            # for matching with the dimension of the same name in the pair density
            "match_keys": ["first_atom_type"],
            "skip_redundant": True,
            "output_selection": None,
            "keep_l_in_keys": True,
            "arrays_backend": None,
            "cg_backend": None,
        }
        self._density_correlations_hypers = density_correlations_hypers
        self._density_correlations_calc = DensityCorrelations(
            **density_correlations_hypers
        )

    def forward(self, frames) -> TensorMap:
        """
        Calls the :py:meth:`EquivariantPowerSpectrumByPair.compute` function.

        This is intended for :py:class:`torch.nn.Module` compatibility, and should be
        ignored in pure Python mode.
        """
        return self.compute(frames)

    def compute(self, frames) -> TensorMap:
        """
        Computes the Lambda-SOAP descriptor for the given ``frames`` and takes the CG
        tensor product with a SphericalExpansionByPair descriptor.

        :param frames: The input frames, i.e. ase Atoms objects.
        """
        return self._correlate_tensor_with_density(frames, compute_metadata=False)

    @torch_jit_export
    def compute_metadata(self, frames) -> TensorMap:
        """
        Returns the metadata-only :py:class:`TensorMap` that would be output by the
        function :py:meth:`compute` for the same calculator under the same settings,
        without performing the actual Clebsch-Gordan tensor products.

        :param density: A density descriptor of body order 2 (correlation order 1), in
            :py:class:`TensorMap` format. This may be, for example, a rascaline
            :py:class:`SphericalExpansion` or :py:class:`LodeSphericalExpansion`.
            Alternatively, this could be multi-center descriptor, such as a pair
            density.
        """
        return self._correlate_tensor_with_density(
            frames,
            compute_metadata=True,
        )

    def _correlate_tensor_with_density(
        self, frames, compute_metadata: bool
    ) -> TensorMap:

        # Compute the nu = 1 density and move "neighbor_type" to properties
        density = self._spherical_expansion_calc.compute(frames)

        # Rename "center_type" -> "first_atom_type" in the keys, and "atom" ->
        # "first_atom" in the samples for consistency with the pair density to be
        # computed later.
        density = operations.rename_dimension(
            density, "keys", "center_type", "first_atom_type"
        )
        density = operations.rename_dimension(density, "samples", "atom", "first_atom")

        # Move "neighbor_type" to "properties
        if self._atom_types is None:
            keys_to_move = ["neighbor_type"]
        else:
            keys_to_move = Labels(
                names=["neighbor_type"],
                values=_dispatch.int_array_like(
                    self._atom_types, like=density.keys.values
                ).reshape(-1, 1),
            )
        density = density.keys_to_properties(keys_to_move)

        density = operations.insert_dimension(density, "keys", 0, "order_nu", 1)

        # # Compute the nu = 2 lambda-SOAP descriptor
        # lsoap = self._density_correlations_calc.compute(density)

        # # Change "body_order" to "order_nu"
        # lsoap = operations.insert_dimension(lsoap, "keys", 0, "order_nu", 2)
        # lsoap = operations.remove_dimension(lsoap, "keys", "body_order")

        # Compute the nu = 2 SphericalExpansionByPair descriptor
        pair_density = self._spherical_expansion_by_pair_calc.compute(frames)

        # Copy the "second_atom_type" key dimension to a new properties dimension called
        # "second_atom_type" so that it is correlated.
        pair_density = operations.append_dimension(
            pair_density,
            "keys",
            "neighbor_type",
            values=pair_density.keys["second_atom_type"],
        )

        # Manipulate the pair density metadata ready for CG tensor product
        pair_density = pair_density.keys_to_properties(
            keys_to_move=Labels(
                names=["neighbor_type"],
                values=_dispatch.int_array_like(
                    self._atom_types, pair_density.keys.values
                ).reshape(-1, 1),
            )
        )
        # pair_density = operations.insert_dimension(
        #     pair_density, axis="keys", index=0, name="order_nu", values=1
        # )
        # pair_density = operations.rename_dimension(
        #     pair_density, "properties", "neighbor_type", "neighbor_3_type"
        # )
        # pair_density = operations.rename_dimension(
        #     pair_density, "properties", "n", "n_3"
        # )
        pair_density = operations.insert_dimension(
            pair_density, axis="keys", index=0, name="order_nu", values=1
        )
        pair_density = operations.rename_dimension(
            pair_density, "properties", "neighbor_type", "neighbor_2_type"
        )
        pair_density = operations.rename_dimension(
            pair_density, "properties", "n", "n_2"
        )

        # Symmetrise permutations. TODO: to remove
        pair_density = _utils._symmetrise_permutations(pair_density)

        # Initialize the CorrelateTensorWithDensity calculator. Re-use the CG
        # coefficients computed by the DensityCorrelations constructor in when this
        # class was initialized.
        tensor_correlator_hypers = {
            "max_angular": self._density_correlations_hypers["max_angular"],
            "angular_cutoff": self._angular_cutoff,
            "selected_keys": self._selected_keys,
            "match_keys": ["first_atom_type"],
            "match_samples": ["system", "first_atom"],
            "keep_l_in_keys": False,
            "arrays_backend": None,
            "cg_backend": None,
        }
        tensor_correlator_calc = CorrelateTensorWithDensity(
            **tensor_correlator_hypers,
            # Re-use CG coefficients computed by DensityCorrelations
            cg_coefficients=self._density_correlations_calc._cg_coefficients,
        )

        # # Compute the CG tensor product of the two descriptors
        # if compute_metadata:
        #     tensor_correlation = tensor_correlator_calc.compute_metadata(lsoap, pair_density)
        # else:
        #     tensor_correlation = tensor_correlator_calc.compute(lsoap, pair_density)

        # Compute the CG tensor product of the two descriptors
        if compute_metadata:
            tensor_correlation = tensor_correlator_calc.compute_metadata(density, pair_density)
        else:
            tensor_correlation = tensor_correlator_calc.compute(density, pair_density)

        return tensor_correlation



