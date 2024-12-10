from typing import List

import metatensor
import numpy as np
import pytest
from metatensor import Labels, TensorBlock, TensorMap

from featomic import SphericalExpansion
from featomic.clebsch_gordan import EquivariantPowerSpectrum, PowerSpectrum


# Try to import some modules
ase = pytest.importorskip("ase")
import ase.io  # noqa: E402, F811


MAX_ANGULAR = 2
SPHEX_HYPERS_SMALL = {
    "cutoff": {
        "radius": 2.5,
        "smoothing": {"type": "ShiftedCosine", "width": 0.5},
    },
    "density": {
        "type": "Gaussian",
        "width": 0.2,
    },
    "basis": {
        "type": "TensorProduct",
        # use a small basis to make the tests faster
        # FIXME: setting max_angular=1 breaks the tests
        "max_angular": MAX_ANGULAR,
        "radial": {"type": "Gto", "max_radial": 1},
    },
}

# ============ Helper functions ============


def h2o_isolated():
    return [
        ase.Atoms(
            symbols=["O", "H", "H"],
            positions=[
                [2.56633400, 2.50000000, 2.50370100],
                [1.97361700, 1.73067300, 2.47063400],
                [1.97361700, 3.26932700, 2.47063400],
            ],
        )
    ]


def h2o_periodic():
    return [
        ase.Atoms(
            symbols=["O", "H", "H"],
            positions=[
                [2.56633400, 2.50000000, 2.50370100],
                [1.97361700, 1.73067300, 2.47063400],
                [1.97361700, 3.26932700, 2.47063400],
            ],
            cell=[5, 5, 5],
            pbc=[True, True, True],
        )
    ]


def power_spectrum(frames: List[ase.Atoms]):
    """Returns a featomic PowerSpectrum constructed from a
    SphericalExpansion"""
    return PowerSpectrum(SphericalExpansion(**SPHEX_HYPERS_SMALL)).compute(frames)


# ============ Test EquivariantPowerSpectrum vs PowerSpectrum ============


@pytest.mark.parametrize("frames", [h2o_isolated(), h2o_periodic()])
def test_equivariant_power_spectrum_vs_powerspectrum(frames):
    """
    Tests for exact equivalence between the invariant block of a generated
    EquivariantPowerSpectrum the Python implementation of PowerSpectrum in featomic
    utils.
    """
    # Build a PowerSpectrum
    ps_1 = power_spectrum(frames)

    # Build an EquivariantPowerSpectrum
    ps_2 = EquivariantPowerSpectrum(SphericalExpansion(**SPHEX_HYPERS_SMALL)).compute(
        frames,
        selected_keys=Labels(names=["o3_lambda"], values=np.array([0]).reshape(-1, 1)),
        neighbors_to_properties=False,
    )

    # Manipulate metadata to match
    ps_2 = ps_2.keys_to_properties(["neighbor_1_type", "neighbor_2_type"])
    keys = ps_2.keys.remove(name="o3_lambda")  # redundant as all 0
    keys = keys.remove("o3_sigma")  # redundant as all 1

    blocks = []
    for block in ps_2.blocks():
        n_samples, n_props = block.values.shape[0], block.values.shape[2]
        new_props = block.properties
        new_props = new_props.remove(name="l_1")
        new_props = new_props.rename(old="l_2", new="l")
        blocks.append(
            TensorBlock(
                values=block.values.reshape((n_samples, n_props)),
                samples=block.samples,
                components=[],
                properties=new_props,
            )
        )
    ps_2 = TensorMap(keys=keys, blocks=blocks)

    # Permute properties dimension to match ps_1 and sort
    ps_2 = metatensor.sort(
        metatensor.permute_dimensions(ps_2, "properties", [2, 0, 3, 1, 4]),
        "properties",
    )

    metatensor.equal_metadata_raise(ps_1, ps_2)
    metatensor.allclose_raise(ps_1, ps_2)


@pytest.mark.parametrize("frames", [h2o_isolated(), h2o_periodic()])
def test_equivariant_power_spectrum_neighbors_to_properties(frames):
    """
    Tests that computing an EquivariantPowerSpectrum is equivalent when passing
    `neighbors_to_properties` as both True and False (after metadata manipulation).
    """
    # Build an EquivariantPowerSpectrum
    powspec_calc = EquivariantPowerSpectrum(SphericalExpansion(**SPHEX_HYPERS_SMALL))

    # Compute the first. Move keys after CG step
    powspec_1 = powspec_calc.compute(
        frames,
        neighbors_to_properties=False,
    )
    powspec_1 = powspec_1.keys_to_properties(["neighbor_1_type", "neighbor_2_type"])

    # Compute the second.  Move keys before the CG step
    powspec_2 = powspec_calc.compute(
        frames,
        neighbors_to_properties=True,
    )

    # Permute properties dimensions to match ``powspec_1`` and sort
    powspec_2 = metatensor.sort(
        metatensor.permute_dimensions(powspec_2, "properties", [2, 4, 0, 1, 3, 5])
    )

    # Check equivalent
    metatensor.equal_metadata_raise(powspec_1, powspec_2)
    metatensor.equal_raise(powspec_1, powspec_2)
