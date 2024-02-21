# -*- coding: utf-8 -*-
import io
import os
from typing import Any, List

import ase.io
import metatensor.torch
import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap  # noqa

import rascaline.torch
from rascaline.torch.utils.clebsch_gordan.correlate_density import DensityCorrelations


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


@torch.jit.script
def is_tensor_map(obj: Any):
    return isinstance(obj, TensorMap)


is_tensor_map = torch.jit.script(is_tensor_map)

SPHERICAL_EXPANSION_HYPERS = {
    "cutoff": 2.5,
    "max_radial": 3,
    "max_angular": 3,
    "atomic_gaussian_width": 0.2,
    "radial_basis": {"Gto": {}},
    "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
    "center_atom_weight": 1.0,
}

SELECTED_KEYS = Labels(
    names=["spherical_harmonics_l"], values=torch.tensor([1, 3]).reshape(-1, 1)
)


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


def spherical_expansion(frames: List[ase.Atoms]):
    """Returns a rascaline SphericalExpansion"""
    calculator = rascaline.torch.SphericalExpansion(**SPHERICAL_EXPANSION_HYPERS)
    return calculator.compute(rascaline.torch.systems_to_torch(frames))


# copy of def test_correlate_density_angular_selection(
@pytest.mark.parametrize("selected_keys", [None, SELECTED_KEYS])
@pytest.mark.parametrize("skip_redundant", [True, False])
def test_torch_script_correlate_density_angular_selection(
    selected_keys: Labels,
    skip_redundant: bool,
):
    """
    Tests that the correct angular channels are output based on the specified
    ``selected_keys``.
    """
    frames = h2o_isolated()
    nu_1 = spherical_expansion(frames)
    correlation_order = 2
    corr_calculator = DensityCorrelations(
        max_angular=SPHERICAL_EXPANSION_HYPERS["max_angular"] * correlation_order,
        correlation_order=correlation_order,
        angular_cutoff=None,
        selected_keys=selected_keys,
        skip_redundant=skip_redundant,
    )

    ref_nu_2 = corr_calculator.compute(nu_1)
    scripted_corr_calculator = torch.jit.script(corr_calculator)

    # Test compute
    scripted_nu_2 = scripted_corr_calculator.compute(nu_1)
    assert metatensor.torch.equal_metadata(scripted_nu_2, ref_nu_2)
    assert metatensor.torch.allclose(scripted_nu_2, ref_nu_2)

    # Test compute_metadata
    scripted_nu_2 = scripted_corr_calculator.compute_metadata(nu_1)
    assert metatensor.torch.equal_metadata(scripted_nu_2, ref_nu_2)


def test_jit_save_load():
    corr_calculator = DensityCorrelations(
        max_angular=2,
        correlation_order=2,
        angular_cutoff=1,
    )
    scripted_correlate_density = torch.jit.script(corr_calculator)
    with io.BytesIO() as buffer:
        torch.jit.save(scripted_correlate_density, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)


def test_save_load():
    """Tests for saving and loading with cg_backend="python-dense",
    which makes the DensityCorrelations object non-scriptable due to
    a non-contiguous CG cache."""
    corr_calculator = DensityCorrelations(
        max_angular=2,
        correlation_order=2,
        angular_cutoff=1,
        cg_backend="python-dense",
    )
    with io.BytesIO() as buffer:
        torch.save(corr_calculator, buffer)
        buffer.seek(0)
        torch.load(buffer)
