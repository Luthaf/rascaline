# -*- coding: utf-8 -*-
import io
import os
from typing import List

import ase.io
import metatensor.torch
import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap  # noqa

import rascaline.torch
from rascaline.torch.utils.clebsch_gordan.correlate_density import correlate_density


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")

SPHEX_HYPERS = {
    "cutoff": 2.5,  # Angstrom
    "max_radial": 3,  # Exclusive
    "max_angular": 3,  # Inclusive
    "atomic_gaussian_width": 0.2,
    "radial_basis": {"Gto": {}},
    "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
    "center_atom_weight": 1.0,
}


def h2o_isolated():
    return ase.io.read(os.path.join(DATA_ROOT, "h2o_isolated.xyz"), ":")


def spherical_expansion(frames: List[ase.Atoms]):
    """Returns a rascaline SphericalExpansion"""
    calculator = rascaline.torch.SphericalExpansion(**SPHEX_HYPERS)
    return calculator.compute(rascaline.torch.systems_to_torch(frames))


# copy of def test_correlate_density_angular_selection(
@pytest.mark.parametrize(
    "selected_keys",
    [
        None,
        Labels(
            names=["spherical_harmonics_l"], values=torch.tensor([1, 3]).reshape(-1, 1)
        ),
    ],
)
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
    scripted_correlate_density = torch.jit.script(correlate_density)
    scripted_nu_2 = scripted_correlate_density(
        density=nu_1,
        correlation_order=2,
        angular_cutoff=None,
        selected_keys=selected_keys,
        skip_redundant=skip_redundant,
    )
    nu_2 = correlate_density(
        density=nu_1,
        correlation_order=2,
        angular_cutoff=None,
        selected_keys=selected_keys,
        skip_redundant=skip_redundant,
    )
    assert metatensor.torch.equal_metadata(scripted_nu_2, nu_2)
    # The test below cannot pass for the moment until we can script wigners or extract
    # cg_cache out of the scripting. For the moment the output is only zeros
    # assert metatensor.torch.allclose(scripted_nu_2, nu_2)


def test_save_load():
    scripted_correlate_density = torch.jit.script(correlate_density)
    buffer = io.BytesIO()
    torch.jit.save(scripted_correlate_density, buffer)
    buffer.seek(0)
    torch.jit.load(buffer)
    buffer.close()
