"""Use internal profiler for timing data on SOAP calculations."""
import sys

import ase.io

import rascaline
from rascaline import SoapPowerSpectrum


def compute_soap(path):
    """Compute SOAP power spectrum.

    This is the same code as the 'compute-soap' example
    """
    frames = ase.io.read(sys.argv[1], ":")

    HYPER_PARAMETERS = {
        "cutoff": 5.0,
        "max_radial": 6,
        "max_angular": 4,
        "atomic_gaussian_width": 0.3,
        "gradients": False,
        "radial_basis": {
            "Gto": {},
        },
        "cutoff_function": {
            "ShiftedCosine": {"width": 0.5},
        },
    }

    calculator = SoapPowerSpectrum(**HYPER_PARAMETERS)
    descriptor = calculator.compute(frames)
    descriptor.densify(["species_neighbor_1", "species_neighbor_2"])

    return descriptor


if __name__ == "__main__":
    # run the calculation with profiling enabled
    with rascaline.Profiler() as profiler:
        descriptor = compute_soap(sys.argv[1])

    # display the recorded profiling data
    print(profiler.as_short_table())

    # Or save this data as json for future usage
    print(profiler.as_json())
