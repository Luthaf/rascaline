import io

import torch

from featomic.torch import SphericalExpansion
from featomic.torch.clebsch_gordan import EquivariantPowerSpectrum


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
