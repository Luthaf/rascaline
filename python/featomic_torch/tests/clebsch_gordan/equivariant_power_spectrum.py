import io

import torch

from featomic.torch import SphericalExpansion
from featomic.torch.clebsch_gordan import EquivariantPowerSpectrum


SPHEX_HYPERS_SMALL = {
    "cutoff": {
        "radius": 5.5,
        "smoothing": {"type": "ShiftedCosine", "width": 0.5},
    },
    "density": {
        "type": "Gaussian",
        "width": 0.2,
    },
    "basis": {
        "type": "TensorProduct",
        "max_angular": 6,
        "radial": {"type": "Gto", "max_radial": 4},
    },
}


def test_jit_save_load():
    calculator = torch.jit.script(
        EquivariantPowerSpectrum(
            SphericalExpansion(**SPHEX_HYPERS_SMALL),
            dtype=torch.float64,
        )
    )

    with io.BytesIO() as buffer:
        torch.jit.save(calculator, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
