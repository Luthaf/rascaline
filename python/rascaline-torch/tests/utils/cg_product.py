# -*- coding: utf-8 -*-
import io

import metatensor.torch as mts
import pytest
import torch
from metatensor.torch import Labels
from metatensor.torch.atomistic import System

import rascaline.torch
from rascaline.torch.utils.clebsch_gordan import ClebschGordanProduct


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
            ],
        ),
        cell=torch.zeros(
            (3, 3),
        ),
    )


def spherical_expansion():
    """Returns a rascaline SphericalExpansion"""
    calculator = rascaline.torch.SphericalExpansion(**SPHERICAL_EXPANSION_HYPERS)
    return calculator.compute(system())


def keys_filter_example(keys: Labels):
    return [0, 1, 3, 5]


@pytest.mark.parametrize("selected_keys", [None, SELECTED_KEYS])
@pytest.mark.parametrize("keys_filter", [None, keys_filter_example])
def test_torch_script_tensor_compute(selected_keys: Labels, keys_filter):
    nu_1 = spherical_expansion()

    # Initialize the calculator and scripted calculator
    calculator = ClebschGordanProduct(
        max_angular=SPHERICAL_EXPANSION_HYPERS["max_angular"] * 2,
        keys_filter=keys_filter,
    )
    scripted_calculator = torch.jit.script(calculator)

    # Compute the reference and scripted results
    ref_nu_2 = calculator.compute(
        mts.rename_dimension(nu_1, "properties", "n", "n_1"),
        mts.rename_dimension(nu_1, "properties", "n", "n_2"),
        o3_lambda_1_new_name="l_1",
        o3_lambda_2_new_name="l_2",
        selected_keys=selected_keys,
    )
    scripted_nu_2 = scripted_calculator.compute(
        mts.rename_dimension(nu_1, "properties", "n", "n_1"),
        mts.rename_dimension(nu_1, "properties", "n", "n_2"),
        o3_lambda_1_new_name="l_1",
        o3_lambda_2_new_name="l_2",
        selected_keys=selected_keys,
    )
    mts.equal_metadata_raise(scripted_nu_2, ref_nu_2)
    assert mts.allclose(scripted_nu_2, ref_nu_2)


def test_save_load():
    calculator = torch.jit.script(
        ClebschGordanProduct(
            max_angular=2,
            # FIXME: we should be able to save/load modules with other dtypes, but they
            # currently crash in metatensor serialization
            dtype=torch.float64,
        )
    )
    with io.BytesIO() as buffer:
        torch.jit.save(calculator, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
