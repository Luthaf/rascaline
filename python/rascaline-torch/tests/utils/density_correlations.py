# -*- coding: utf-8 -*-
import io

import metatensor.torch
import pytest
import torch
from metatensor.torch import Labels
from metatensor.torch.atomistic import System

import rascaline.torch
from rascaline.torch.utils import DensityCorrelations


SPHERICAL_EXPANSION_HYPERS = {
    "cutoff": 2.5,
    "max_radial": 3,
    "max_angular": 3,
    "atomic_gaussian_width": 0.2,
    "radial_basis": {"Gto": {}},
    "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
    "center_atom_weight": 1.0,
}

SELECTED_KEYS = Labels(names=["o3_lambda"], values=torch.tensor([1, 3]).reshape(-1, 1))


def system():
    return System(
        types=torch.tensor([8, 1, 1]),
        positions=torch.tensor(
            [
                [2.56633400, 2.50000000, 2.50370100],
                [1.97361700, 1.73067300, 2.47063400],
                [1.97361700, 3.26932700, 2.47063400],
            ]
        ),
        cell=torch.zeros((3, 3)),
    )


def spherical_expansion():
    """Returns a rascaline SphericalExpansion"""
    calculator = rascaline.torch.SphericalExpansion(**SPHERICAL_EXPANSION_HYPERS)
    return calculator.compute(system())


@pytest.mark.parametrize("selected_keys", [None, SELECTED_KEYS])
@pytest.mark.parametrize("skip_redundant", [True, False])
def test_torch_script_correlate_density_angular_selection(
    selected_keys: Labels,
    skip_redundant: bool,
):
    nu_1 = spherical_expansion()

    # Initialize the calculator and scripted calculator
    calculator = DensityCorrelations(
        n_correlations=1,
        max_angular=SPHERICAL_EXPANSION_HYPERS["max_angular"] * 2,
        skip_redundant=skip_redundant,
    )
    scripted_calculator = torch.jit.script(calculator)

    # Compute the reference and scripted results
    ref_nu_2 = calculator.compute(nu_1, selected_keys=selected_keys)
    scripted_nu_2 = scripted_calculator.compute(nu_1, selected_keys=selected_keys)
    metatensor.torch.equal_metadata_raise(scripted_nu_2, ref_nu_2)
    assert metatensor.torch.allclose(scripted_nu_2, ref_nu_2)


@pytest.mark.parametrize("cg_backend", ["python-dense", "python-sparse"])
def test_jit_save_load(cg_backend: str):
    calculator = torch.jit.script(
        DensityCorrelations(
            n_correlations=1,
            max_angular=2,
            cg_backend=cg_backend,
            # FIXME: we should be able to save/load modules with other dtypes, but they
            # currently crash in metatensor serialization
            dtype=torch.float64,
        )
    )
    with io.BytesIO() as buffer:
        torch.jit.save(calculator, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
