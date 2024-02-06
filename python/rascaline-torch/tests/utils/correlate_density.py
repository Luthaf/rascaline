# -*- coding: utf-8 -*-
from typing import List
import os
import torch
import rascaline.torch
from metatensor.torch import Labels, TensorBlock, TensorMap  # noqa
import ase.io
from rascaline.torch.utils.clebsch_gordan.correlate_density import (
    correlate_density,
    correlate_density_metadata,
)


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")

def h2o_isolated():
    return ase.io.read(os.path.join(DATA_ROOT, "h2o_isolated.xyz"), ":")

def spherical_expansion(frames: List[ase.Atoms]):
    """Returns a rascaline SphericalExpansion"""
    calculator = rascaline.torch.SphericalExpansion(**SPHEX_HYPERS)
    return calculator.compute(frames)

# copy of def test_correlate_density_angular_selection(
def test_scritability(
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
    assert metatensor.torch.allclose(scripted_nu_2, nu_2)


def test_save_load():
    scripted_correlate_density = torch.jit.script(correlate_density)
    buffer = io.BytesIO()
    torch.jit.save(scripted_correlate_density, buffer)
    buffer.seek(0)
    torch.jit.load(buffer)
    buffer.close()
