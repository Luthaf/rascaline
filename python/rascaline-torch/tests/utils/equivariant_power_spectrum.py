# -*- coding: utf-8 -*-
import io

import torch

from rascaline.torch.utils import EquivariantPowerSpectrum


SPHERICAL_EXPANSION_HYPERS = {
    "cutoff": 2.5,
    "max_radial": 3,
    "max_angular": 3,
    "atomic_gaussian_width": 0.2,
    "radial_basis": {"Gto": {}},
    "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
    "center_atom_weight": 1.0,
}


def test_jit_save_load():
    calculator = torch.jit.script(
        EquivariantPowerSpectrum(
            **SPHERICAL_EXPANSION_HYPERS,
            dtype=torch.float64,
        )
    )

    with io.BytesIO() as buffer:
        torch.jit.save(calculator, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
