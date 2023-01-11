"""
Profiling calculation
=====================

.. start-body
"""

import chemfiles

import rascaline
from rascaline import SoapPowerSpectrum


def compute_soap(path):
    """Compute SOAP power spectrum.

    This is the same code as the 'compute-soap' example
    """
    with chemfiles.Trajectory(path) as trajectory:
        frames = [f for f in trajectory]

    HYPER_PARAMETERS = {
        "cutoff": 5.0,
        "max_radial": 6,
        "max_angular": 4,
        "atomic_gaussian_width": 0.3,
        "center_atom_weight": 1.0,
        "radial_basis": {
            "Gto": {},
        },
        "cutoff_function": {
            "ShiftedCosine": {"width": 0.5},
        },
    }

    calculator = SoapPowerSpectrum(**HYPER_PARAMETERS)
    descriptor = calculator.compute(frames, gradients=["positions"])
    descriptor = descriptor.keys_to_samples("species_center")
    descriptor = descriptor.keys_to_properties(
        ["species_neighbor_1", "species_neighbor_2"]
    )

    return descriptor


# %%
#
# Run the calculation with profiling enabled.

with rascaline.Profiler() as profiler:
    descriptor = compute_soap("dataset.xyz")
# %%
#
# Display the recorded profiling data as table.

print(profiler.as_short_table())

# %%
#
# You can also save this data as json for future usage
print(profiler.as_json())

# %%
#
# .. end-body
